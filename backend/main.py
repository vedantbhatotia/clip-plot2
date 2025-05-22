





# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
import os
import shutil
import uuid
from dotenv import load_dotenv # <<<--- CORRECTED
import logging
from typing import Optional, List, Union, Any, Dict
import asyncio
import json
import math

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

class VideoSummaryResponse(BaseModel):
    video_uuid: str
    summary: Optional[str]
    message: str

class SummaryHighlightResponse(BaseModel):
    video_uuid: str
    text_summary: Optional[str]
    message: str
    estimated_highlight_path: Optional[str] = None

def get_transcript_text_for_segment(
    transcript_data: Dict[str, Any], 
    segment_start_time: float, 
    segment_end_time: float,
    video_id_for_log: str = "N/A"
) -> Optional[str]:
    # ... (your existing implementation of this function)
    log_extra = {'video_id': video_id_for_log}
    relevant_texts = []
    whisper_segments = transcript_data.get("segments", [])
    if not whisper_segments:
        main_logger.warning(f"No 'segments' key found in transcript data.", extra=log_extra)
        return None
    for ws_segment in whisper_segments:
        ws_start = ws_segment.get("start")
        ws_end = ws_segment.get("end")
        ws_text = ws_segment.get("text", "").strip()
        if ws_start is None or ws_end is None or not ws_text:
            continue
        if ws_start < segment_end_time and segment_start_time < ws_end: # Overlap check
            relevant_texts.append(ws_text)
    if relevant_texts:
        full_text = " ".join(relevant_texts).strip()
        main_logger.debug(f"Found transcript text for {segment_start_time:.2f}-{segment_end_time:.2f}: '{full_text[:100]}...'", extra=log_extra)
        return full_text
    else:
        main_logger.warning(f"No overlapping Whisper segments found for {segment_start_time:.2f}-{segment_end_time:.2f}", extra=log_extra)
        return None



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


# @app.post("/agent/generate_highlight", summary="[LLM-Assisted] Generate highlight clip", response_model=GenerateHighlightResponse)
# async def agent_generate_highlight_endpoint(
#     request_data: AgentHighlightRequest, # AgentHighlightRequest only needs video_uuid, user_query
#     background_tasks: BackgroundTasks
# ):
#     log_req_extra = {'video_id': request_data.video_uuid}
#     main_logger.info(f"LLM-Assisted highlight generation request. Query: '{request_data.user_query}'", extra=log_req_extra)

#     # 1. Perform initial broad semantic search to get candidates for the LLM
#     #    You might want to fetch more candidates than the final clip will have.
#     search_top_k_candidates = 10 # How many initial segments to feed to the LLM
    
#     initial_segments_raw = await search_service.perform_semantic_text_search(
#         query_text=request_data.user_query,
#         video_uuid=request_data.video_uuid,
#         top_k=search_top_k_candidates 
#     )
#     # Optionally, also perform visual search and combine/interleave candidates if query implies visuals

#     if not initial_segments_raw:
#         main_logger.warning("Initial search for LLM selection found no segments.", extra=log_req_extra)
#         raise HTTPException(status_code=404, detail="No initial segments found to select from for the highlight.")

#     main_logger.info(f"Fetched {len(initial_segments_raw)} candidate segments for LLM selection.", extra=log_req_extra)

#     # 2. Call the LLM-powered segment selection service
#     llm_selected_segments: Optional[List[Dict[str, Any]]] = None
#     try:
#         llm_selected_segments = await rag_service.select_segments_for_highlight_with_llm(
#             query_str=request_data.user_query,
#             candidate_segments=initial_segments_raw,
#             max_output_segments=5, # Ask LLM to pick up to 5 best segments
#             video_id_for_log=request_data.video_uuid
#         )
#     except Exception as e_llm_select:
#         main_logger.exception("Exception during LLM segment selection call.", extra=log_req_extra)
#         raise HTTPException(status_code=500, detail=f"LLM segment selection failed: {str(e_llm_select)}")

#     if llm_selected_segments is None:
#         main_logger.error("LLM segment selection service returned None (error).", extra=log_req_extra)
#         raise HTTPException(status_code=500, detail="LLM failed to select segments for the highlight clip.")
#     if not llm_selected_segments:
#         main_logger.info("LLM selected no suitable segments for the highlight.", extra=log_req_extra)
#         return GenerateHighlightResponse(
#             video_uuid=request_data.video_uuid,
#             message="LLM determined no suitable segments for a highlight based on the query.",
#             estimated_highlight_path=None 
#         )

#     main_logger.info(f"LLM selected {len(llm_selected_segments)} segments. Raw from LLM: {json.dumps(llm_selected_segments, indent=2)}", extra=log_req_extra)
    
#     # --- These segments from LLM are now treated like "user-provided" segments ---
#     # We still need to get video duration and then pass them to segment_processor_service

#     video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, request_data.video_uuid)
#     original_video_path_for_duration: Optional[str] = None
#     video_duration: Optional[float] = None

#     async with database_service.get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, request_data.video_uuid)
#         if not video_record or not video_record.original_video_file_path:
#             main_logger.error("Original video record or path not found in DB for LLM-assisted highlight.", extra=log_req_extra)
#             raise HTTPException(status_code=404, detail=f"Original video {request_data.video_uuid} not found.")
#         if not os.path.exists(video_record.original_video_file_path):
#             main_logger.error(f"Original video file MISSING at {video_record.original_video_file_path}", extra=log_req_extra)
#             raise HTTPException(status_code=404, detail="Original video file is missing on server.")
#         original_video_path_for_duration = video_record.original_video_file_path

#     if original_video_path_for_duration:
#         try:
#             def get_duration_sync(path): 
#                 with VideoFileClip(path) as clip: return clip.duration
#             video_duration = await asyncio.to_thread(get_duration_sync, original_video_path_for_duration)
#             main_logger.info(f"Fetched video duration: {video_duration:.2f}s for segment refinement.", extra=log_req_extra)
#         except Exception as e_dur:
#             main_logger.warning(f"Could not get video duration: {e_dur}. Segment capping may not be accurate.", extra=log_req_extra)

#     # Use default padding/merging values or make them configurable in AgentHighlightRequest
#     # For now, using some sensible defaults.
#     segments_to_pass_to_builder = refine_segments_for_clip(
#         segments=llm_selected_segments, # Use LLM selected segments
#         padding_start_sec=0.25, # Minimal padding as LLM already selected context
#         padding_end_sec=0.25,
#         max_gap_to_merge_sec=0.1, # Minimal merging
#         video_duration=video_duration, 
#         video_id=request_data.video_uuid
#     )
    
#     if not segments_to_pass_to_builder:
#         main_logger.error("No valid segments remained after refinement of LLM-selected segments.", extra=log_req_extra)
#         raise HTTPException(status_code=400, detail="No valid segments after LLM selection and refinement.")
    
#     main_logger.info(f"Using {len(segments_to_pass_to_builder)} LLM-selected & refined segments for clip builder.", extra=log_req_extra)

#     if not os.path.isdir(video_processing_base_path):
#         main_logger.error(f"Processing base path for highlights not found: {video_processing_base_path}.", extra=log_req_extra)
#         raise HTTPException(status_code=404, detail=f"Processing directory for video {request_data.video_uuid} not found.")

#     async with database_service.get_db_session() as session:
#         await database_service.update_video_status_and_error(
#             session, request_data.video_uuid, database_service.VideoProcessingStatus.HIGHLIGHT_GENERATING,
#             error_msg=f"LLM-assisted highlight for query: {request_data.user_query}"
#         )

#     background_tasks.add_task(
#         clip_builder_service.generate_highlight_clip, video_id=request_data.video_uuid,
#         segments_to_include=segments_to_pass_to_builder, 
#         processing_base_path=video_processing_base_path,
#         output_filename=request_data.output_filename
#     )
#     main_logger.info("LLM-assisted highlight generation task added to background.", extra=log_req_extra)

#     effective_output_filename = request_data.output_filename or f"highlight_llm_{request_data.video_uuid}_{str(uuid.uuid4())[:8]}.mp4"
#     estimated_path = os.path.join(video_processing_base_path, clip_builder_service.HIGHLIGHT_CLIPS_SUBDIR, effective_output_filename)
    
#     return GenerateHighlightResponse(
#         video_uuid=request_data.video_uuid, 
#         message="LLM-assisted highlight generation has been queued.",
#         estimated_highlight_path=estimated_path
#     )

@app.post("/agent/generate_highlight", summary="[LLM-Assisted] Generate highlight clip", response_model=GenerateHighlightResponse)
async def agent_generate_highlight_endpoint(
    request_data: AgentHighlightRequest, 
    background_tasks: BackgroundTasks
):
    log_req_extra = {'video_id': request_data.video_uuid}
    main_logger.info(f"LLM-Assisted highlight generation request. Query: '{request_data.user_query}'", extra=log_req_extra)

    # 1. Perform initial broad semantic search
    search_top_k_candidates = 10
    initial_segments_raw = await search_service.perform_semantic_text_search(
        query_text=request_data.user_query,
        video_uuid=request_data.video_uuid,
        top_k=search_top_k_candidates 
    )
    if not initial_segments_raw:
        main_logger.warning("Initial search for LLM selection found no segments.", extra=log_req_extra)
        raise HTTPException(status_code=404, detail="No initial segments found to select from for the highlight.")
    main_logger.info(f"Fetched {len(initial_segments_raw)} candidate segments for LLM selection.", extra=log_req_extra)

    # 2. Call LLM-powered segment selection
    llm_selected_segments_with_placeholder_text: Optional[List[Dict[str, Any]]] = None
    try:
        llm_selected_segments_with_placeholder_text = await rag_service.select_segments_for_highlight_with_llm(
            query_str=request_data.user_query,
            candidate_segments=initial_segments_raw, # These have real segment_text
            max_output_segments=5,
            video_id_for_log=request_data.video_uuid
        )
    except Exception as e_llm_select:
        main_logger.exception("Exception during LLM segment selection call.", extra=log_req_extra)
        raise HTTPException(status_code=500, detail=f"LLM segment selection failed: {str(e_llm_select)}")

    if llm_selected_segments_with_placeholder_text is None:
        main_logger.error("LLM segment selection service returned None (error).", extra=log_req_extra)
        raise HTTPException(status_code=500, detail="LLM failed to determine segments for the highlight clip.")
    if not llm_selected_segments_with_placeholder_text:
        main_logger.info("LLM selected no suitable segments for the highlight.", extra=log_req_extra)
        return GenerateHighlightResponse(
            video_uuid=request_data.video_uuid,
            message="LLM determined no suitable segments for a highlight based on the query.",
            estimated_highlight_path=None 
        )
    main_logger.info(f"LLM selected {len(llm_selected_segments_with_placeholder_text)} segments (potentially with placeholder text). Raw from LLM: {json.dumps(llm_selected_segments_with_placeholder_text, indent=2)}", extra=log_req_extra)
    
    # --- 3. Fetch Full Transcript Text for LLM-Selected Segments ---
    segments_for_clip_raw_with_actual_text: List[Dict[str, Any]] = []
    transcript_file_path_for_video: Optional[str] = None
    transcript_data_cache: Dict[str, Any] = {}

    async with database_service.get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, request_data.video_uuid)
        if video_record:
            transcript_file_path_for_video = video_record.transcript_file_path
            original_video_path_for_duration = video_record.original_video_file_path # Already fetched
        else: # Should not happen if video_uuid is valid from request
            main_logger.error("Video record not found when trying to fetch transcript path.", extra=log_req_extra)
            raise HTTPException(status_code=404, detail="Video record disappeared.")

    if transcript_file_path_for_video and os.path.exists(transcript_file_path_for_video):
        try:
            with open(transcript_file_path_for_video, "r", encoding="utf-8") as f:
                transcript_data_cache = json.load(f)
            main_logger.info(f"Loaded transcript data from: {transcript_file_path_for_video} for populating LLM selected segments.", extra=log_req_extra)
        except Exception as e_load_transcript:
            main_logger.error(f"Failed to load transcript file {transcript_file_path_for_video}: {e_load_transcript}", extra=log_req_extra)
    else:
        main_logger.warning(f"Transcript file path not found or file does not exist: {transcript_file_path_for_video}. Subtitles for LLM-selected segments might be placeholders.", extra=log_req_extra)

    for llm_seg in llm_selected_segments_with_placeholder_text:
        start_time = llm_seg.get("start_time")
        end_time = llm_seg.get("end_time")
        # The text_content from LLM might be a placeholder like "This is the first selected segment text."
        # We should prioritize fetching the actual transcript.
        actual_text_content = None
        if transcript_data_cache and start_time is not None and end_time is not None:
            actual_text_content = get_transcript_text_for_segment(
                transcript_data_cache, start_time, end_time, request_data.video_uuid
            )
        
        if not actual_text_content: # Fallback if no transcript or no overlap
            actual_text_content = llm_seg.get("text_content", "") # Use LLM's text if no better found
            main_logger.warning(f"Using LLM-provided/placeholder text for segment {start_time}-{end_time} as full transcript fetch failed or yielded nothing.", extra=log_req_extra)
        
        if start_time is not None and end_time is not None:
             segments_for_clip_raw_with_actual_text.append({
                 "start_time": start_time,
                 "end_time": end_time,
                 "text_content": actual_text_content
             })
    
    if not segments_for_clip_raw_with_actual_text:
        # This should ideally not happen if llm_selected_segments_with_placeholder_text was not empty
        # and contained valid start/end times.
        main_logger.error("Failed to populate actual text for any LLM-selected segments.", extra=log_req_extra)
        raise HTTPException(status_code=500, detail="Error processing LLM-selected segments for subtitles.")

    # --- 4. Get video duration and Refine these LLM-selected (and now text-populated) segments ---
    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, request_data.video_uuid)
    video_duration: Optional[float] = None
    # original_video_path_for_duration already fetched via video_record earlier

    if original_video_path_for_duration: # This path was fetched when getting transcript_file_path
        try:
            def get_duration_sync(path): 
                with VideoFileClip(path) as clip: return clip.duration
            video_duration = await asyncio.to_thread(get_duration_sync, original_video_path_for_duration)
            main_logger.info(f"Fetched video duration: {video_duration:.2f}s for segment refinement.", extra=log_req_extra)
        except Exception as e_dur:
            main_logger.warning(f"Could not get video duration: {e_dur}. Segment capping may not be accurate.", extra=log_req_extra)

    segments_to_pass_to_builder = refine_segments_for_clip(
        segments=segments_for_clip_raw_with_actual_text, # Use segments with actual text
        padding_start_sec=0.25, 
        padding_end_sec=0.25,
        max_gap_to_merge_sec=0.1, 
        video_duration=video_duration, 
        video_id=request_data.video_uuid
    )
    
    if not segments_to_pass_to_builder:
        main_logger.error("No valid segments after refinement of LLM-selected segments.", extra=log_req_extra)
        raise HTTPException(status_code=400, detail="No valid segments after LLM selection and refinement for highlight generation.")
    
    main_logger.info(f"Using {len(segments_to_pass_to_builder)} LLM-selected & refined segments for clip builder.", extra=log_req_extra)

    # ... (rest of the endpoint: check video_processing_base_path, update DB status, add background task, return response) ...
    # (This part remains the same as your existing /agent/generate_highlight)
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


@app.post("/videos/{video_uuid}/summarize", 
            summary="Generate a summary of the entire video content",
            response_model=VideoSummaryResponse)
async def summarize_video_content(video_uuid: str):
    log_req_extra = {'video_id': video_uuid}
    main_logger.info(f"Full video summary request received.", extra=log_req_extra)

    transcript_file_path: Optional[str] = None
    async with database_service.get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, video_uuid)
        if not video_record:
            main_logger.warning(f"Video record not found in DB.", extra=log_req_extra)
            raise HTTPException(status_code=404, detail="Video not found.")
        if not video_record.transcript_file_path or not os.path.exists(video_record.transcript_file_path):
            main_logger.warning(f"Transcript file not found for video. Path: {video_record.transcript_file_path}", extra=log_req_extra)
            raise HTTPException(status_code=404, detail="Transcript not found for this video, cannot summarize.")
        transcript_file_path = video_record.transcript_file_path

    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
    except Exception as e:
        main_logger.exception(f"Failed to load or parse transcript file: {transcript_file_path}", extra=log_req_extra)
        raise HTTPException(status_code=500, detail="Error reading transcript file.")

    whisper_segments = transcript_data.get("segments", [])
    if not whisper_segments:
        main_logger.info(f"No segments found in transcript for video. Cannot summarize.", extra=log_req_extra)
        return VideoSummaryResponse(
            video_uuid=video_uuid, 
            summary=None, 
            message="No content found in transcript to summarize."
        )

    all_segments_as_search_results: List[Dict[str, Any]] = []
    for seg in whisper_segments:
        all_segments_as_search_results.append({
            "segment_text": seg.get("text","").strip(), 
            "text_content": seg.get("text","").strip(), 
            "start_time": seg.get("start"),
            "end_time": seg.get("end")
        })

    if not all_segments_as_search_results:
        main_logger.info("No processable text segments found after formatting.", extra=log_req_extra)
        return VideoSummaryResponse(video_uuid=video_uuid, summary=None, message="No text content available to summarize.")

    summary_query = "Provide a concise summary of the following video content."
    
    segments_to_summarize = []
    concatenated_text_for_summary = ""
    max_chars_for_summary_context = 3000 
    
    for seg in whisper_segments:
        text_to_add = seg.get("text", "").strip()
        if text_to_add:
            if len(concatenated_text_for_summary) + len(text_to_add) + 2 < max_chars_for_summary_context:
                concatenated_text_for_summary += text_to_add + " "
            else:
                break 
    
    if not concatenated_text_for_summary.strip():
        main_logger.info("No text content to summarize after attempting concatenation.", extra=log_req_extra)
        return VideoSummaryResponse(video_uuid=video_uuid, summary=None, message="No text content available for summary.")

    mock_search_result_for_summary = [{
        "segment_text": concatenated_text_for_summary.strip(),
        "start_time": whisper_segments[0].get("start") if whisper_segments else 0, 
        "end_time": whisper_segments[-1].get("end") if whisper_segments else 0, 
    }]

    main_logger.info(f"Calling RAG service for full video summary with concatenated text (length: {len(concatenated_text_for_summary)}).", extra=log_req_extra)
    
    video_summary = await rag_service.refine_search_results_with_rag(
        query_str=summary_query,
        search_results=mock_search_result_for_summary,
        max_context_segments=1, 
        video_id_for_log=video_uuid
    )

    if video_summary and "unavailable" not in video_summary.lower() and "error" not in video_summary.lower() :
        main_logger.info(f"Full video summary generated successfully.", extra=log_req_extra)
        return VideoSummaryResponse(video_uuid=video_uuid, summary=video_summary, message="Summary generated successfully.")
    else:
        main_logger.error(f"Failed to generate full video summary or RAG service reported an issue. LLM Output: {video_summary}", extra=log_req_extra)
        return VideoSummaryResponse(video_uuid=video_uuid, summary=None, message=video_summary or "Failed to generate summary.")

async def get_full_transcript_text_and_segments(
    video_uuid: str, 
    log_req_extra: Dict[str, Any]
) -> tuple[Optional[str], Optional[List[Dict[str,Any]]], Optional[str]]:
    """Helper to load full transcript text and structured segments."""
    transcript_file_path: Optional[str] = None
    async with database_service.get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, video_uuid)
        if not video_record:
            main_logger.warning(f"Video record not found in DB.", extra=log_req_extra)
            return None, None, None # Text, Segments, Original Video Path
        if not video_record.transcript_file_path or not os.path.exists(video_record.transcript_file_path):
            main_logger.warning(f"Transcript file not found. Path: {video_record.transcript_file_path}", extra=log_req_extra)
            return None, None, video_record.original_video_file_path
        transcript_file_path = video_record.transcript_file_path
        original_video_path = video_record.original_video_file_path

    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        
        whisper_segments = transcript_data.get("segments", [])
        full_text = " ".join([seg.get("text", "").strip() for seg in whisper_segments if seg.get("text")])
        return full_text.strip(), whisper_segments, original_video_path
    except Exception as e:
        main_logger.exception(f"Failed to load/parse transcript: {transcript_file_path}", extra=log_req_extra)
        return None, None, None


@app.post("/videos/{video_uuid}/summary_highlight_clip",
            summary="Generate a text summary and a highlight clip based on the summary",
            response_model=SummaryHighlightResponse)
async def generate_summary_and_clip(
    video_uuid: str,
    background_tasks: BackgroundTasks,
    # Optional: Add request body for target_duration_for_clip, max_segments_for_clip
    target_clip_duration_approx: Optional[int] = Body(60, description="Approximate target duration for the highlight clip in seconds."),
    max_segments_in_clip: Optional[int] = Body(10, description="Maximum number of distinct segments to try and include.")
):
    log_req_extra = {'video_id': video_uuid}
    main_logger.info(f"Full video summary & highlight clip request received.", extra=log_req_extra)

    full_transcript_text, whisper_segments_for_mapping, original_video_path = await get_full_transcript_text_and_segments(video_uuid, log_req_extra)

    if not full_transcript_text or not whisper_segments_for_mapping: # whisper_segments needed for mapping summary back
        raise HTTPException(status_code=404, detail="Transcript content not available to generate summary or map segments.")

    max_chars_for_summary_context = 3500 # Adjust based on LLM
    summary_context_text = full_transcript_text[:max_chars_for_summary_context]
    
    mock_search_result_for_summary = [{ # Format for existing RAG function
        "segment_text": summary_context_text,
        "start_time": 0, "end_time": 0 # Not crucial for this summary prompt
    }]
    summary_query = "Provide a concise summary of the key points in the following video content. Remember your output would be used to make a highlight or summary video, so try retaining information. Try to join multiple short clips, instead of selecting a single long clip."
    
    text_summary = await rag_service.refine_search_results_with_rag(
        query_str=summary_query,
        search_results=mock_search_result_for_summary,
        max_context_segments=1, # We are passing one large chunk
        video_id_for_log=video_uuid
    )

    if not text_summary or "unavailable" in text_summary.lower() or "error" in text_summary.lower():
        main_logger.error(f"Failed to generate text summary. LLM Output: {text_summary}", extra=log_req_extra)
        # Return summary error but still try to make a generic highlight? Or fail here?
        # For now, let's allow proceeding to clip generation even if summary fails, using a different strategy for segments.
        # Or, better, raise error if summary is critical for segment selection.
        raise HTTPException(status_code=500, detail=f"Failed to generate text summary: {text_summary}")

    main_logger.info(f"Text summary generated: {text_summary[:200]}...", extra=log_req_extra)

    # --- 2. Identify Segments Based on the Text Summary (Using Option B: Semantic Search) ---
    segments_for_clip_raw: List[Dict[str, Any]] = []
    if text_summary:
        # Split summary into sentences (simple split, can be improved with NLP library like spaCy/NLTK)
        summary_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_summary) if s.strip()]
        if not summary_sentences and text_summary: # If no sentence splits, use whole summary as one query
            summary_sentences = [text_summary]
            
        main_logger.info(f"Attempting to find segments for {len(summary_sentences)} key points/sentences from summary.", extra=log_req_extra)

        # For each key sentence in the summary, find the best matching segment in the original video
        # Limit the number of segments to avoid overly long processing for search
        # And ensure we don't get too many tiny, fragmented results from ChromaDB for each summary sentence
        
        all_candidate_segments_from_summary: List[Dict[str, Any]] = []
        for i, sentence in enumerate(summary_sentences[:max_segments_in_clip]): # Limit queries from summary
            main_logger.info(f"Searching for video segments related to summary sentence {i+1}: '{sentence[:100]}...'", extra=log_req_extra)
            # Search for this sentence in the video's text embeddings
            # We want the top 1-2 most relevant original segments for this summary sentence
            sentence_search_results = await search_service.perform_semantic_text_search(
                query_text=sentence,
                video_uuid=video_uuid,
                top_k=1 # Get the single best matching original segment for this summary sentence
            )
            if sentence_search_results:
                # Add these to a list of candidates, include original summary sentence for context if needed
                for res in sentence_search_results:
                    res["source_summary_sentence"] = sentence # Keep track of which summary part it relates to
                    all_candidate_segments_from_summary.append(res)
        
        # De-duplicate and sort candidates (e.g., if multiple summary sentences point to same video segment)
        # A simple de-duplication based on start_time can be done.
        unique_segments_dict: Dict[float, Dict[str, Any]] = {}
        for seg in all_candidate_segments_from_summary:
            st = seg.get("start_time")
            if st is not None:
                if st not in unique_segments_dict: # Keep first one found for a start time
                     unique_segments_dict[st] = seg
                # Optionally, if a new segment for same start time has higher score, replace.
                # elif seg.get("score",0) > unique_segments_dict[st].get("score",0):
                #    unique_segments_dict[st] = seg

        segments_for_clip_raw = sorted(list(unique_segments_dict.values()), key=lambda x: x.get("start_time", 0))
        
        # Now, ensure the text_content for these segments is the full transcript text
        # (as done in /agent/generate_highlight)
        segments_with_full_text: List[Dict[str, Any]] = []
        if segments_for_clip_raw and whisper_segments_for_mapping: # whisper_segments_for_mapping loaded earlier
            for seg_from_search in segments_for_clip_raw:
                s_start = seg_from_search.get("start_time")
                s_end = seg_from_search.get("end_time")
                full_seg_text = get_transcript_text_for_segment(
                    {"segments": whisper_segments_for_mapping}, # Pass in correct format
                    s_start, 
                    s_end,
                    video_uuid
                )
                segments_with_full_text.append({
                    "start_time": s_start,
                    "end_time": s_end,
                    "text_content": full_seg_text or seg_from_search.get("segment_text", "") # Fallback
                })
            segments_for_clip_raw = segments_with_full_text
        
        main_logger.info(f"Identified {len(segments_for_clip_raw)} raw segments based on summary sentences.", extra=log_req_extra)

    if not segments_for_clip_raw:
        main_logger.warning("No segments identified from summary to generate clip. Clip generation might be empty or fail.", extra=log_req_extra)
        # Fallback: maybe create a clip of first N seconds of video if summary-based selection fails?
        # For now, let it proceed, segment_processor might return empty.
        # Or raise HTTPException here.
        raise HTTPException(status_code=404, detail="Could not identify relevant video segments based on the generated summary.")


    # --- 3. Refine and Build Clip ---
    video_duration: Optional[float] = None
    if original_video_path: # Fetched earlier
        try:
            def get_duration_sync(path): 
                with VideoFileClip(path) as clip: return clip.duration
            video_duration = await asyncio.to_thread(get_duration_sync, original_video_path)
        except Exception as e_dur:
            main_logger.warning(f"Could not get video duration for summary clip: {e_dur}", extra=log_req_extra)

    # Apply segment processing (padding, merging, capping)
    # Use different padding/merging for summary clips if desired, e.g., tighter.
    segments_to_pass_to_builder = refine_segments_for_clip(
        segments=segments_for_clip_raw,
        padding_start_sec=0.2, # Less padding for summary segments
        padding_end_sec=0.2,
        max_gap_to_merge_sec=0.5, # Allow some merging
        video_duration=video_duration,
        video_id=video_uuid
    )

    if not segments_to_pass_to_builder:
        main_logger.error("No valid segments remained after refinement for summary clip.", extra=log_req_extra)
        raise HTTPException(status_code=400, detail="No valid segments after refinement for summary clip generation.")

    main_logger.info(f"Using {len(segments_to_pass_to_builder)} refined segments for summary clip.", extra=log_req_extra)
    
    # --- Queue Clip Generation ---
    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, video_uuid)
    output_clip_filename = f"summary_highlight_{video_uuid}_{str(uuid.uuid4())[:8]}.mp4"

    async with database_service.get_db_session() as session: # New session for this status update
        await database_service.update_video_status_and_error(
            session, video_uuid, database_service.VideoProcessingStatus.HIGHLIGHT_GENERATING,
            error_msg=f"Summary clip generation initiated."
        )

    background_tasks.add_task(
        clip_builder_service.generate_highlight_clip,
        video_id=video_uuid,
        segments_to_include=segments_to_pass_to_builder,
        processing_base_path=video_processing_base_path,
        output_filename=output_clip_filename
    )
    main_logger.info("Summary highlight clip generation task added to background.", extra=log_req_extra)

    estimated_path = os.path.join(video_processing_base_path, clip_builder_service.HIGHLIGHT_CLIPS_SUBDIR, output_clip_filename)
    return SummaryHighlightResponse(
        video_uuid=video_uuid,
        text_summary=text_summary,
        message="Text summary generated and highlight clip generation has been queued.",
        estimated_highlight_path=estimated_path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001")) 
    reload_flag = os.getenv("UVICORN_RELOAD", "true").lower() == "true"
    main_logger.info(f"Starting Uvicorn server on port {port} with reload={reload_flag}. TEMP_VIDEO_DIR: {TEMP_VIDEO_DIR}", extra={'video_id': 'SYSTEM_INIT'})
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_flag)