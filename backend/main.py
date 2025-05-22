





# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from dotenv import load_dotenv # <<<--- CORRECTED
import logging
from typing import Optional, List, Union, Any, Dict
import asyncio
import json

# --- LOAD DOTENV AT THE VERY BEGINNING ---
load_dotenv() # <<<--- CORRECTED

# --- Import your services ---
from services.media_processor import process_video_for_extraction # Ensure this is async def if it awaits
from services import database_service
from services import search_service
from services import clip_builder_service
from services.segment_processor_service import refine_segments_for_clip
from services import rag_service
from services import orchestration_service
from moviepy.editor import VideoFileClip # For getting video duration

from pydantic import BaseModel, Field

# --- Centralized Logging Configuration ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(), # Allow configuring log level via .env
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - [%(video_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Add a NullHandler to the root logger to prevent duplicate messages if uvicorn also configures logging
# logging.getLogger().addHandler(logging.NullHandler())
# logging.getLogger().propagate = False

main_logger = logging.getLogger(__name__)
# Add a filter to add video_id to log records if not present
class VideoIDLogFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'video_id'):
            record.video_id = 'N/A' # Default if no video_id is set
        return True
main_logger.addFilter(VideoIDLogFilter())
# For all loggers created via logging.getLogger(...)
logging.getLogger().addFilter(VideoIDLogFilter())


app = FastAPI(title="ClipPilot.ai Backend")

# --- CORS Middleware ---
origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
main_logger.info(f"CORS allowed origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Configuration ---
TEMP_VIDEO_DIR = os.getenv("TEMP_VIDEO_DIR", "/tmp/clippilot_uploads_dev")
if not os.path.exists(TEMP_VIDEO_DIR):
    try:
        os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
        main_logger.info(f"Successfully created TEMP_VIDEO_DIR at: {TEMP_VIDEO_DIR}")
    except OSError as e:
        main_logger.critical(f"CRITICAL: Failed to create TEMP_VIDEO_DIR at {TEMP_VIDEO_DIR}: {e}. Application may not function correctly.")
else:
    main_logger.info(f"TEMP_VIDEO_DIR found at: {TEMP_VIDEO_DIR}")

# --- Pydantic Models ---
class AgentHighlightRequest(BaseModel):
    video_uuid: str
    user_query: str
    output_filename: Optional[str] = None


class HighlightSegmentRequest(BaseModel):
    start_time: float
    end_time: float
    text_content: Optional[str] = None

class GenerateHighlightRequest(BaseModel):
    video_uuid: str = Field(..., description="UUID of the original video to generate highlights from.")
    segments: Optional[List[HighlightSegmentRequest]] = Field(None, description="List of specific segments to include in the highlight.")
    search_query_text: Optional[str] = Field(None, description="If segments are not provided, use this query to find segments.")
    search_type: Optional[str] = Field("text", pattern="^(text|visual)$", description="Type of search if using search_query_text.")
    search_top_k: Optional[int] = Field(5, gt=0, le=20, description="How many top search results to consider for the clip.")
    
    padding_start_seconds: float = Field(1.0, ge=0, description="Padding to add to the start of each segment (seconds).")
    padding_end_seconds: float = Field(1.5, ge=0, description="Padding to add to the end of each segment (seconds).")
    max_merge_gap_seconds: float = Field(0.75, ge=0, description="Maximum gap between segments to consider them for merging (seconds).")
    
    output_filename: Optional[str] = Field(None, description="Optional custom filename for the highlight clip.")

class GenerateHighlightResponse(BaseModel):
    video_uuid: str
    message: str
    highlight_job_id: Optional[str] = None
    estimated_highlight_path: Optional[str] = None

class SearchQueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="The text query for semantic search.")
    video_uuid: Optional[str] = Field(None, description="Optional: UUID of a specific video to search within.")
    top_k: int = Field(5, gt=0, le=20, description="Number of top results to return.")
    search_type: str = Field("text", pattern="^(text|visual)$", description="Type of search: 'text' or 'visual'.")
    use_rag_refinement: bool = Field(False, description="Whether to use RAG to refine/summarize search results.")

class TextSearchResultItem(BaseModel):
    id: str
    video_uuid: Optional[str] = None
    segment_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    score: Optional[float] = None
    score_type: Optional[str] = None
    _raw_distance: Optional[float] = None

class VisionSearchResultItem(BaseModel):
    id: str
    video_uuid: Optional[str] = None
    frame_filename: Optional[str] = None
    frame_timestamp_sec: Optional[float] = None
    score: Optional[float] = None
    score_type: Optional[str] = None
    _raw_distance: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    search_type: str
    results: List[Union[TextSearchResultItem, VisionSearchResultItem]]
    rag_summary: Optional[str] = None

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def on_startup():
    log_extra = {'video_id': 'SYSTEM'}
    main_logger.info("Application startup sequence initiated...", extra=log_extra)
    if database_service.DATABASE_URL and database_service.async_engine:
        main_logger.info("Attempting to initialize database tables...", extra=log_extra)
        await database_service.init_db_tables()
        main_logger.info("Database initialization sequence complete (tables checked/created).", extra=log_extra)
    else:
        main_logger.error("DATABASE_URL not configured or async_engine failed in database_service. Database features will be UNAVAILABLE.", extra=log_extra)
    main_logger.info("Application startup complete.", extra=log_extra)


# --- API Endpoints ---
@app.get("/ping", summary="Health check")
async def ping():
    main_logger.info("Ping endpoint called", extra={'video_id': 'HEALTH_CHECK'})
    return {"message": "pong from ClipPilot.ai backend"}

@app.post("/search", summary="Perform semantic search on processed videos", response_model=SearchResponse)
async def search_videos(query_request: SearchQueryRequest = Body(...)):
    log_extra = {'video_id': query_request.video_uuid or "all_videos_search"}
    main_logger.info(f"Search request: query='{query_request.query_text}', type='{query_request.search_type}', k='{query_request.top_k}'", extra=log_extra)
    
    results_data: List[Union[TextSearchResultItem, VisionSearchResultItem]] = []
    search_results_raw: List[Dict[str, Any]] = []
    rag_refined_output: Optional[str] = None
    esults_data_pydantic: List[Union[TextSearchResultItem, VisionSearchResultItem]] = []

    if query_request.search_type == "text":
        if not search_service.text_embeddings_collection or not search_service.get_text_embedding_model():
            main_logger.error("Search (text): Text embedding service components not ready.", extra=log_extra)
            raise HTTPException(status_code=503, detail="Text search service components not ready.")
        
        search_results_raw = await search_service.perform_semantic_text_search(
            query_text=query_request.query_text, video_uuid=query_request.video_uuid, top_k=query_request.top_k
        )
        results_data = [TextSearchResultItem(**item) for item in search_results_raw]

    elif query_request.search_type == "visual":
        if not search_service.vision_embeddings_collection or not search_service.get_vision_embedding_model_and_processor(): 
            main_logger.error("Search (visual): Vision embedding service components not ready.", extra=log_extra)
            raise HTTPException(status_code=503, detail="Vision search service components not ready.")

        search_results_raw = await search_service.perform_semantic_visual_search(
            query_text_for_visual=query_request.query_text, video_uuid=query_request.video_uuid, top_k=query_request.top_k
        )
        results_data = [VisionSearchResultItem(**item) for item in search_results_raw]
    else:
        raise HTTPException(status_code=400, detail="Invalid search_type. Must be 'text' or 'visual'.")

    ### RAG begins
    if query_request.use_rag_refinement and search_results_raw: # Only if requested and results exist
        main_logger.info(f"video_id: {query_request.video_uuid or 'all'} - RAG refinement requested. Calling RAG service.", extra=log_extra)
        rag_refined_output = await rag_service.refine_search_results_with_rag(
            query_str=query_request.query_text,
            search_results=search_results_raw, # Pass the raw dictionaries
            max_context_segments=min(len(search_results_raw), 3), # e.g., use top 3 for context
            video_id_for_log=query_request.video_uuid
        )
        if rag_refined_output:
             main_logger.info(f"video_id: {query_request.video_uuid or 'all'} - RAG refinement successful.", extra=log_extra)
        else:
             main_logger.warning(f"video_id: {query_request.video_uuid or 'all'} - RAG refinement did not produce output or failed.", extra=log_extra)
    # --- END RAG Refinement Step ---

    return SearchResponse(query=query_request.query_text,
                          search_type=query_request.search_type,
                          results=results_data,
                          rag_summary=rag_refined_output)

@app.post("/upload", summary="Upload a video for processing")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    main_logger.info(f"Upload request received for file: {file.filename or 'N/A'}", extra={'video_id': 'PRE_UUID_UPLOAD'})

    if not file.content_type or not file.content_type.startswith("video/"):
        main_logger.warning(f"Invalid file type: Content-Type='{file.content_type}' for file: {file.filename or 'N/A'}", extra={'video_id': 'INVALID_UPLOAD'})
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
    video_id = str(uuid.uuid4())
    log_req_extra = {'video_id': video_id} 

    main_logger.info(f"Generated for file: {file.filename or 'N/A'}", extra=log_req_extra)
    
    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, video_id)
    try:
        os.makedirs(video_processing_base_path, exist_ok=True)
        main_logger.info(f"Created processing directory: {video_processing_base_path}", extra=log_req_extra)
    except OSError as e:
        main_logger.error(f"Failed to create video processing directory {video_processing_base_path}: {e}", extra=log_req_extra)
        raise HTTPException(status_code=500, detail="Server error: Could not create processing directory.")
            
    original_filename_sanitized = "uploaded_video.mp4" # Default
    if file.filename:
        temp_name = "".join(c for c in file.filename if c.isalnum() or c in ['.', '_', '-']).strip()
        if temp_name: original_filename_sanitized = temp_name
            
    original_video_file_path = os.path.join(video_processing_base_path, original_filename_sanitized)
    
    try:
        async with database_service.get_db_session() as session: # Ensure session is available
            try:
                with open(original_video_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                main_logger.info(f"Video file saved to: {original_video_file_path}", extra=log_req_extra)
            except Exception as e_save:
                main_logger.exception("Error saving uploaded file physically.", extra=log_req_extra)
                raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e_save)}")

            if not database_service.AsyncSessionLocal: # Check if DB is configured
                main_logger.error("Database not configured. Cannot create video record.", extra=log_req_extra)
                raise HTTPException(status_code=503, detail="Database service not available.") # 503 Service Unavailable

            video_record = await database_service.add_new_video_record(
                session=session, video_uuid=video_id,
                original_filename_server=original_filename_sanitized,
                original_video_file_path=original_video_file_path
            )
            if not video_record:
                main_logger.error("Failed to create database record for new video (returned None).", extra=log_req_extra)
                raise HTTPException(status_code=500, detail="Failed to create video record in database.")
            main_logger.info(f"Database record created with DB ID: {video_record.id}", extra=log_req_extra)

        # process_video_for_extraction should be an async function if it awaits DB calls etc.
        background_tasks.add_task(
            process_video_for_extraction, video_id=video_id,
            original_video_path=original_video_file_path,
            video_processing_base_path=video_processing_base_path
        )
        main_logger.info("Background processing task added.", extra=log_req_extra)
        
        return JSONResponse(status_code=202, content={
            "video_id": video_id, "message": "Video upload accepted. Processing has been queued.",
            "original_filename_on_server": original_filename_sanitized,
        })
    except HTTPException: raise # Re-raise HTTPExceptions directly
    except Exception as e:
        main_logger.exception("Unexpected error during upload processing.", extra=log_req_extra)
        if os.path.exists(video_processing_base_path): # Cleanup attempt
            try: shutil.rmtree(video_processing_base_path)
            except Exception as e_rm: main_logger.error(f"Error cleaning directory {video_processing_base_path}: {e_rm}", extra=log_req_extra)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during upload: {str(e)}")
    finally:
        if file: await file.close()

@app.post("/generate_highlight", summary="Generate a highlight clip from a video", response_model=GenerateHighlightResponse)
async def generate_highlight_endpoint(request_data: GenerateHighlightRequest, background_tasks: BackgroundTasks):
    log_req_extra = {'video_id': request_data.video_uuid}
    main_logger.info(f"Highlight generation request: {request_data.model_dump(exclude_none=True)}", extra=log_req_extra)
    
    segments_for_clip_raw: List[Dict[str, Any]] = []

    if request_data.segments:
        segments_for_clip_raw = [segment.model_dump() for segment in request_data.segments]
        main_logger.info(f"Using {len(segments_for_clip_raw)} user-provided segments.", extra=log_req_extra)
    elif request_data.search_query_text:
        main_logger.info(f"Deriving segments from search: '{request_data.search_query_text}', type: '{request_data.search_type}'", extra=log_req_extra)
        search_results_from_service: List[Dict[str, Any]] = []
        if request_data.search_type == "text":
            search_results_from_service = await search_service.perform_semantic_text_search(
                query_text=request_data.search_query_text, video_uuid=request_data.video_uuid, top_k=request_data.search_top_k
            )
        elif request_data.search_type == "visual":
            search_results_from_service = await search_service.perform_semantic_visual_search(
                query_text_for_visual=request_data.search_query_text, video_uuid=request_data.video_uuid, top_k=request_data.search_top_k
            )
        
        if not search_results_from_service:
            raise HTTPException(status_code=404, detail="No search results found for the query to generate highlights.")

        for res in search_results_from_service:
            start_time = res.get("start_time") if request_data.search_type == "text" else res.get("frame_timestamp_sec")
            end_time = res.get("end_time") if request_data.search_type == "text" else (start_time + 3.0 if start_time is not None else None) # Default 3s for visual
            text_content = res.get("segment_text") if request_data.search_type == "text" else f"Visual hit around {start_time:.2f}s"

            if start_time is not None and end_time is not None:
                segments_for_clip_raw.append({"start_time": start_time, "end_time": end_time, "text_content": text_content})
        main_logger.info(f"Derived {len(segments_for_clip_raw)} raw segments from search.", extra=log_req_extra)
    else:
        raise HTTPException(status_code=400, detail="Either 'segments' or 'search_query_text' must be provided.")

    if not segments_for_clip_raw:
        raise HTTPException(status_code=400, detail="No segments determined for highlight generation.")

    video_duration: Optional[float] = None
    original_video_path_for_duration: Optional[str] = None
    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, request_data.video_uuid)

    async with database_service.get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, request_data.video_uuid)
        if not video_record or not video_record.original_video_file_path:
            main_logger.error("Original video record or path not found in DB.", extra=log_req_extra)
            raise HTTPException(status_code=404, detail=f"Original video {request_data.video_uuid} not found or path missing.")
        if not os.path.exists(video_record.original_video_file_path):
            main_logger.error(f"Original video file MISSING at {video_record.original_video_file_path}", extra=log_req_extra)
            raise HTTPException(status_code=404, detail="Original video file is missing on server.")
        original_video_path_for_duration = video_record.original_video_file_path

    # try:
    #     if original_video_path_for_duration:
    #         with VideoFileClip(original_video_path_for_duration) as temp_clip: # This is blocking
    #             video_duration = temp_clip.duration
    #         main_logger.info(f"Fetched video duration: {video_duration:.2f}s for segment refinement.", extra=log_req_extra)
    # except Exception as e_dur:
    #     main_logger.warning(f"Could not get video duration using MoviePy: {e_dur}. Segment capping may not be accurate.", extra=log_req_extra)

    if original_video_path_for_duration:
        try:
            def get_duration_sync(path): # Helper sync function
                with VideoFileClip(path) as clip:
                    return clip.duration
            video_duration = await asyncio.to_thread(get_duration_sync, original_video_path_for_duration)
            main_logger.info(f"Fetched video duration: {video_duration:.2f}s for segment refinement.", extra=log_req_extra)
        except Exception as e_dur:
            main_logger.warning(f"Could not get video duration using MoviePy: {e_dur}. Segment capping may not be accurate.", extra=log_req_extra)


    segments_to_pass_to_builder = refine_segments_for_clip(
        segments_for_clip_raw, padding_start_sec=request_data.padding_start_seconds,
        padding_end_sec=request_data.padding_end_seconds, max_gap_to_merge_sec=request_data.max_merge_gap_seconds,
        video_duration=video_duration, video_id=request_data.video_uuid
    )
    
    if not segments_to_pass_to_builder:
        main_logger.error("No valid segments remained after refinement process.", extra=log_req_extra)
        raise HTTPException(status_code=400, detail="No valid segments found after refinement for highlight generation.")
    
    main_logger.info(f"Using {len(segments_to_pass_to_builder)} refined segments for clip builder.", extra=log_req_extra)

    if not os.path.isdir(video_processing_base_path): # Should exist from upload
        main_logger.error(f"Processing base path for highlights not found: {video_processing_base_path}.", extra=log_req_extra)
        raise HTTPException(status_code=404, detail=f"Processing directory for video {request_data.video_uuid} not found.")

    async with database_service.get_db_session() as session:
        await database_service.update_video_status_and_error(
            session, request_data.video_uuid, database_service.VideoProcessingStatus.HIGHLIGHT_GENERATING
        )

    # generate_highlight_clip should be async if it awaits DB calls etc.
    background_tasks.add_task(
        clip_builder_service.generate_highlight_clip, video_id=request_data.video_uuid,
        segments_to_include=segments_to_pass_to_builder, processing_base_path=video_processing_base_path,
        output_filename=request_data.output_filename
    )
    main_logger.info("Highlight generation task added to background.", extra=log_req_extra)

    effective_output_filename = request_data.output_filename or f"highlight_{request_data.video_uuid}_{str(uuid.uuid4())[:8]}.mp4"
    estimated_path = os.path.join(video_processing_base_path, clip_builder_service.HIGHLIGHT_CLIPS_SUBDIR, effective_output_filename)
    
    return GenerateHighlightResponse(
        video_uuid=request_data.video_uuid, message="Highlight clip generation has been queued.",
        estimated_highlight_path=estimated_path
    )

# @app.post("/agent/generate_highlight", summary="Generate a highlight clip using an LLM agent for orchestration", response_model=GenerateHighlightResponse)
# async def agent_generate_highlight_endpoint(
#     request_data: AgentHighlightRequest, 
#     background_tasks: BackgroundTasks
# ):
#     log_req_extra = {'video_id': request_data.video_uuid}
#     main_logger.info(f"AGENT highlight request for query: '{request_data.user_query}'", extra=log_req_extra)

#     # --- This call would now be to your new agent/orchestration service ---
#     # The orchestrate_clip_generation_with_agent should return the list of segments
#     # For this example, let's assume it returns the segments list directly.
    
#     # Conceptual call to a more advanced RAG/Agent service that returns segments
#     # segments_from_agent = await rag_service.orchestrate_clip_generation_with_agent(
#     #     user_query=request_data.user_query,
#     #     video_uuid=request_data.video_uuid
#     # )
#     # For now, let's use the simpler RAG refine and then process it,
#     # as the full agent is a larger step.
#     # If your refine_search_results_with_rag was adapted to select segments:

#     # Step 1: Basic search to get initial candidates (agent might do this internally)
#     initial_segments_raw = await search_service.perform_semantic_text_search(
#         query_text=request_data.user_query,
#         video_uuid=request_data.video_uuid,
#         top_k=10 # Get more candidates for the agent/LLM to choose from
#     )

#     if not initial_segments_raw:
#         raise HTTPException(status_code=404, detail="Agent: Initial search found no segments.")

#     # Step 2: Use LLM (via rag_service, but with a different prompt/goal) to select/refine segments for a clip
#     # This is where a dedicated function in rag_service would be needed, e.g., select_segments_for_clip_with_llm
#     # For now, we'll simulate that the RAG summary IS the segments or provides guidance.
#     # THIS IS A SIMPLIFICATION - A real agent would return a structured list of segments.
    
#     # Let's assume for now the "agent" just means using RAG to get better text and then we process those.
#     # A true agent would be more iterative.
#     # For now, we'll use the existing /generate_highlight logic path for search-driven segments
#     # but one could imagine the `segments_for_clip_raw` being decided by a more intelligent LLM process.
    
#     # For this example, we'll mimic the current /generate_highlight search path
#     # to show where an agent's output (list of segments) would plug in.
    
#     main_logger.info(f"Agent: Simulating segment selection using existing search for query: '{request_data.user_query}'", extra=log_req_extra)
#     segments_for_clip_raw = []
#     search_results_from_service = await search_service.perform_semantic_text_search(
#         query_text=request_data.user_query, video_uuid=request_data.video_uuid, top_k=5 # Agent might decide this top_k
#     )
#     if not search_results_from_service:
#         raise HTTPException(status_code=404, detail="Agent-simulated search returned no results.")

#     # ... (Same logic as in /generate_highlight to extract start_time, end_time, text_content from search_results_from_service)
#     # ... (This part would be replaced by the direct output of an advanced agent if it returns the structured segment list)
#     for res in search_results_from_service: # Example
#         segments_for_clip_raw.append({
#             "start_time": res.get("start_time"), 
#             "end_time": res.get("end_time"), 
#             "text_content": res.get("segment_text")
#         })
#     # --- End of simulated agent segment selection ---


#     if not segments_for_clip_raw:
#         raise HTTPException(status_code=400, detail="Agent could not determine any segments for highlight.")

#     video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, request_data.video_uuid)
#     original_video_path_for_duration = None
#     video_duration = None
#     async with database_service.get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, request_data.video_uuid)
#         if not video_record or not video_record.original_video_file_path:
#             raise HTTPException(status_code=404, detail="Original video not found for agent processing.")
#         original_video_path_for_duration = video_record.original_video_file_path
    
#     if original_video_path_for_duration:
#         try:
#             def get_duration_sync(path): 
#                 with VideoFileClip(path) as clip: return clip.duration
#             video_duration = await asyncio.to_thread(get_duration_sync, original_video_path_for_duration)
#         except Exception as e_dur:
#             main_logger.warning(f"Agent: Could not get video duration: {e_dur}", extra=log_req_extra)
    
#     # Use default padding/merge values or allow them in AgentHighlightRequest
#     segments_to_pass_to_builder = refine_segments_for_clip(
#         segments_for_clip_raw, 
#         video_duration=video_duration, 
#         video_id=request_data.video_uuid
#     )

#     if not segments_to_pass_to_builder:
#         raise HTTPException(status_code=400, detail="Agent: No valid segments after refinement.")

#     async with database_service.get_db_session() as session:
#         await database_service.update_video_status_and_error(
#             session, request_data.video_uuid, database_service.VideoProcessingStatus.HIGHLIGHT_GENERATING
#         )

#     background_tasks.add_task(
#         clip_builder_service.generate_highlight_clip, video_id=request_data.video_uuid,
#         segments_to_include=segments_to_pass_to_builder, processing_base_path=video_processing_base_path,
#         output_filename=request_data.output_filename
#     )
#     # ... (return GenerateHighlightResponse) ...
#     effective_output_filename = request_data.output_filename or f"highlight_agent_{request_data.video_uuid}_{str(uuid.uuid4())[:8]}.mp4"
#     estimated_path = os.path.join(video_processing_base_path, clip_builder_service.HIGHLIGHT_CLIPS_SUBDIR, effective_output_filename)
    
#     return GenerateHighlightResponse(
#         video_uuid=request_data.video_uuid, message="Agent-driven highlight generation has been queued.",
#         estimated_highlight_path=estimated_path
#     )

# @app.post("/agent/generate_highlight", summary="[AGENT] Generate highlight clip via LLM orchestration", response_model=GenerateHighlightResponse)
# async def agent_generate_highlight_endpoint(
#     request_data: AgentHighlightRequest, 
#     background_tasks: BackgroundTasks
# ):
#     log_req_extra = {'video_id': request_data.video_uuid}
#     main_logger.info(f"AGENT-DRIVEN highlight generation request. Query: '{request_data.user_query}'", extra=log_req_extra)

#     # 1. Call the orchestration agent to get the list of segments
#     # This agent will use tools (search, refine, etc.) and LLM reasoning
#     # to decide on the final segments.
#     agent_selected_segments: Optional[List[Dict[str, Any]]] = None
#     try:
#         agent_selected_segments = await orchestration_service.run_clip_orchestration_agent(
#             user_query=request_data.user_query,
#             video_uuid=request_data.video_uuid
#         )
#     except Exception as e_agent:
#         main_logger.exception("Exception during agent orchestration call.", extra=log_req_extra)
#         raise HTTPException(status_code=500, detail=f"Agent orchestration failed: {str(e_agent)}")

#     if agent_selected_segments is None: # Agent failed to produce a result
#         main_logger.error("Agent orchestration service returned None (error or no segments).", extra=log_req_extra)
#         raise HTTPException(status_code=500, detail="Agent failed to determine segments for the highlight clip.")
#     if not agent_selected_segments: # Agent returned an empty list
#         main_logger.info("Agent determined no suitable segments for highlight.", extra=log_req_extra)
#         # Return a specific message or an empty success, depending on desired behavior
#         # For now, let's treat "no segments" as a valid outcome that doesn't generate a clip.
#         return GenerateHighlightResponse(
#             video_uuid=request_data.video_uuid,
#             message="Agent determined no suitable segments for a highlight based on the query.",
#             estimated_highlight_path=None 
#         )
#         # Alternatively, raise HTTPException(status_code=404, detail="Agent found no suitable segments.")


#     main_logger.info(f"Agent selected {len(agent_selected_segments)} segments for clip builder: {json.dumps(agent_selected_segments, indent=2)}", extra=log_req_extra)
    
#     video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, request_data.video_uuid)
#     if not os.path.isdir(video_processing_base_path):
#         main_logger.error(f"Processing base path for highlights not found: {video_processing_base_path}.", extra=log_req_extra)
#         raise HTTPException(status_code=404, detail=f"Processing directory for video {request_data.video_uuid} not found.")

#     # Update DB status to HIGHLIGHT_GENERATING before starting task
#     async with database_service.get_db_session() as session:
#         await database_service.update_video_status_and_error(
#             session, request_data.video_uuid, database_service.VideoProcessingStatus.HIGHLIGHT_GENERATING,
#             error_msg=f"Agent initiated highlight for query: {request_data.user_query}" # Add context to error_msg field
#         )

#     # Add clip generation to background tasks with the agent-selected segments
#     background_tasks.add_task(
#         clip_builder_service.generate_highlight_clip, 
#         video_id=request_data.video_uuid,
#         segments_to_include=agent_selected_segments, # <<< Segments from the agent
#         processing_base_path=video_processing_base_path,
#         output_filename=request_data.output_filename
#     )
#     main_logger.info("Agent-driven highlight generation task added to background.", extra=log_req_extra)

#     effective_output_filename = request_data.output_filename or f"highlight_agent_{request_data.video_uuid}_{str(uuid.uuid4())[:8]}.mp4"
#     estimated_path = os.path.join(video_processing_base_path, clip_builder_service.HIGHLIGHT_CLIPS_SUBDIR, effective_output_filename)
    
#     return GenerateHighlightResponse(
#         video_uuid=request_data.video_uuid, 
#         message="Agent-driven highlight generation has been queued.",
#         estimated_highlight_path=estimated_path
#     )


@app.post("/agent/generate_highlight", summary="[LLM-Assisted] Generate highlight clip", response_model=GenerateHighlightResponse)
async def agent_generate_highlight_endpoint(
    request_data: AgentHighlightRequest, # AgentHighlightRequest only needs video_uuid, user_query
    background_tasks: BackgroundTasks
):
    log_req_extra = {'video_id': request_data.video_uuid}
    main_logger.info(f"LLM-Assisted highlight generation request. Query: '{request_data.user_query}'", extra=log_req_extra)

    # 1. Perform initial broad semantic search to get candidates for the LLM
    #    You might want to fetch more candidates than the final clip will have.
    search_top_k_candidates = 10 # How many initial segments to feed to the LLM
    
    initial_segments_raw = await search_service.perform_semantic_text_search(
        query_text=request_data.user_query,
        video_uuid=request_data.video_uuid,
        top_k=search_top_k_candidates 
    )
    # Optionally, also perform visual search and combine/interleave candidates if query implies visuals

    if not initial_segments_raw:
        main_logger.warning("Initial search for LLM selection found no segments.", extra=log_req_extra)
        raise HTTPException(status_code=404, detail="No initial segments found to select from for the highlight.")

    main_logger.info(f"Fetched {len(initial_segments_raw)} candidate segments for LLM selection.", extra=log_req_extra)

    # 2. Call the LLM-powered segment selection service
    llm_selected_segments: Optional[List[Dict[str, Any]]] = None
    try:
        llm_selected_segments = await rag_service.select_segments_for_highlight_with_llm(
            query_str=request_data.user_query,
            candidate_segments=initial_segments_raw,
            max_output_segments=5, # Ask LLM to pick up to 5 best segments
            video_id_for_log=request_data.video_uuid
        )
    except Exception as e_llm_select:
        main_logger.exception("Exception during LLM segment selection call.", extra=log_req_extra)
        raise HTTPException(status_code=500, detail=f"LLM segment selection failed: {str(e_llm_select)}")

    if llm_selected_segments is None:
        main_logger.error("LLM segment selection service returned None (error).", extra=log_req_extra)
        raise HTTPException(status_code=500, detail="LLM failed to select segments for the highlight clip.")
    if not llm_selected_segments:
        main_logger.info("LLM selected no suitable segments for the highlight.", extra=log_req_extra)
        return GenerateHighlightResponse(
            video_uuid=request_data.video_uuid,
            message="LLM determined no suitable segments for a highlight based on the query.",
            estimated_highlight_path=None 
        )

    main_logger.info(f"LLM selected {len(llm_selected_segments)} segments. Raw from LLM: {json.dumps(llm_selected_segments, indent=2)}", extra=log_req_extra)
    
    # --- These segments from LLM are now treated like "user-provided" segments ---
    # We still need to get video duration and then pass them to segment_processor_service

    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, request_data.video_uuid)
    original_video_path_for_duration: Optional[str] = None
    video_duration: Optional[float] = None

    async with database_service.get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, request_data.video_uuid)
        if not video_record or not video_record.original_video_file_path:
            main_logger.error("Original video record or path not found in DB for LLM-assisted highlight.", extra=log_req_extra)
            raise HTTPException(status_code=404, detail=f"Original video {request_data.video_uuid} not found.")
        if not os.path.exists(video_record.original_video_file_path):
            main_logger.error(f"Original video file MISSING at {video_record.original_video_file_path}", extra=log_req_extra)
            raise HTTPException(status_code=404, detail="Original video file is missing on server.")
        original_video_path_for_duration = video_record.original_video_file_path

    if original_video_path_for_duration:
        try:
            def get_duration_sync(path): 
                with VideoFileClip(path) as clip: return clip.duration
            video_duration = await asyncio.to_thread(get_duration_sync, original_video_path_for_duration)
            main_logger.info(f"Fetched video duration: {video_duration:.2f}s for segment refinement.", extra=log_req_extra)
        except Exception as e_dur:
            main_logger.warning(f"Could not get video duration: {e_dur}. Segment capping may not be accurate.", extra=log_req_extra)

    # Use default padding/merging values or make them configurable in AgentHighlightRequest
    # For now, using some sensible defaults.
    segments_to_pass_to_builder = refine_segments_for_clip(
        segments=llm_selected_segments, # Use LLM selected segments
        padding_start_sec=0.25, # Minimal padding as LLM already selected context
        padding_end_sec=0.25,
        max_gap_to_merge_sec=0.1, # Minimal merging
        video_duration=video_duration, 
        video_id=request_data.video_uuid
    )
    
    if not segments_to_pass_to_builder:
        main_logger.error("No valid segments remained after refinement of LLM-selected segments.", extra=log_req_extra)
        raise HTTPException(status_code=400, detail="No valid segments after LLM selection and refinement.")
    
    main_logger.info(f"Using {len(segments_to_pass_to_builder)} LLM-selected & refined segments for clip builder.", extra=log_req_extra)

    if not os.path.isdir(video_processing_base_path):
        main_logger.error(f"Processing base path for highlights not found: {video_processing_base_path}.", extra=log_req_extra)
        raise HTTPException(status_code=404, detail=f"Processing directory for video {request_data.video_uuid} not found.")

    async with database_service.get_db_session() as session:
        await database_service.update_video_status_and_error(
            session, request_data.video_uuid, database_service.VideoProcessingStatus.HIGHLIGHT_GENERATING,
            error_msg=f"LLM-assisted highlight for query: {request_data.user_query}"
        )

    background_tasks.add_task(
        clip_builder_service.generate_highlight_clip, video_id=request_data.video_uuid,
        segments_to_include=segments_to_pass_to_builder, 
        processing_base_path=video_processing_base_path,
        output_filename=request_data.output_filename
    )
    main_logger.info("LLM-assisted highlight generation task added to background.", extra=log_req_extra)

    effective_output_filename = request_data.output_filename or f"highlight_llm_{request_data.video_uuid}_{str(uuid.uuid4())[:8]}.mp4"
    estimated_path = os.path.join(video_processing_base_path, clip_builder_service.HIGHLIGHT_CLIPS_SUBDIR, effective_output_filename)
    
    return GenerateHighlightResponse(
        video_uuid=request_data.video_uuid, 
        message="LLM-assisted highlight generation has been queued.",
        estimated_highlight_path=estimated_path
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001")) # Allow configuring port via .env
    reload_flag = os.getenv("UVICORN_RELOAD", "true").lower() == "true"
    main_logger.info(f"Starting Uvicorn server on port {port} with reload={reload_flag}. TEMP_VIDEO_DIR: {TEMP_VIDEO_DIR}", extra={'video_id': 'SYSTEM_INIT'})
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_flag)