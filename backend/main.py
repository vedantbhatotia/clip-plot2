# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from dotenv import load_dotenv
import logging

load_dotenv()
# --- Import your services ---
from services.media_processor import process_video_for_extraction # Ensure this becomes async def
from services import database_service # For database operations

# --- Centralized Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
main_logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="ClipPilot.ai Backend")

# --- CORS Middleware ---
origins = [
    "http://localhost:3000", # Your frontend
    # Add any other origins if needed, or use "*" for development (less secure for prod)
]
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
        # Consider exiting if this directory is essential for all operations
else:
    main_logger.info(f"TEMP_VIDEO_DIR found at: {TEMP_VIDEO_DIR}")


# --- FastAPI Startup Event ---
@app.on_event("startup")
async def on_startup():
    main_logger.info("Application startup sequence initiated...")
    if database_service.DATABASE_URL and database_service.async_engine:
        main_logger.info("Attempting to initialize database tables...")
        await database_service.init_db_tables()
        main_logger.info("Database initialization sequence complete (tables checked/created).")
    else:
        main_logger.error("DATABASE_URL not configured or async_engine failed in database_service. Database features will be UNAVAILABLE.")
    main_logger.info("Application startup complete.")


# --- API Endpoints ---
@app.get("/ping", summary="Health check")
async def ping():
    main_logger.info("Ping endpoint called")
    return {"message": "pong from ClipPilot.ai backend"}


@app.post("/upload", summary="Upload a video for processing")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    main_logger.info(f"UPLOAD_DEBUG: Received file='{file.filename}', Content-Type='{file.content_type}', Size='{file.size or 0}'") # Added default for size
    main_logger.info(f"Upload request received for file: {file.filename if file.filename else 'N/A'}")

    if not file.content_type or not file.content_type.startswith("video/"):
        main_logger.warning(f"Invalid file type uploaded: Content-Type='{file.content_type}' for file: {file.filename if file.filename else 'N/A'}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
    video_id = str(uuid.uuid4())
    log_req_extra = {'video_id': video_id} 

    main_logger.info(f"video_id: {video_id} - Generated for file: {file.filename if file.filename else 'N/A'}", extra=log_req_extra)
    
    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, video_id)
    try:
        os.makedirs(video_processing_base_path, exist_ok=True)
        main_logger.info(f"video_id: {video_id} - Created processing directory: {video_processing_base_path}", extra=log_req_extra)
    except OSError as e:
        main_logger.error(f"video_id: {video_id} - Failed to create video processing directory {video_processing_base_path}: {e}", extra=log_req_extra)
        raise HTTPException(status_code=500, detail="Server error: Could not create processing directory.")
            
    original_filename_sanitized = "uploaded_video.mp4"
    if file.filename:
        temp_name = "".join(c for c in file.filename if c.isalnum() or c in ['.', '_', '-']).strip()
        if temp_name:
            original_filename_sanitized = temp_name
            
    original_video_file_path = os.path.join(video_processing_base_path, original_filename_sanitized)
    
    # Get a database session using the async context manager
    try:
        async with database_service.get_db_session() as session:
            # 1. Save the uploaded file
            try:
                with open(original_video_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                main_logger.info(f"video_id: {video_id} - Video file saved to: {original_video_file_path}", extra=log_req_extra)
            except Exception as e_save:
                main_logger.exception(f"video_id: {video_id} - Error saving uploaded file physically.", extra=log_req_extra)
                raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e_save)}")

            # 2. Create database record for the video
            if not database_service.AsyncSessionLocal: # Check if DB was configured properly
                main_logger.error(f"video_id: {video_id} - Database not configured. Cannot create video record.", extra=log_req_extra)
                raise HTTPException(status_code=500, detail="Database service not available.")

            video_record = await database_service.add_new_video_record(
                session=session, # Pass the active session
                video_uuid=video_id,
                original_filename_server=original_filename_sanitized,
                original_video_file_path=original_video_file_path
            )
            if not video_record: # Could be due to duplicate UUID (highly unlikely) or DB error
                main_logger.error(f"video_id: {video_id} - Failed to create database record for new video (returned None).", extra=log_req_extra)
                raise HTTPException(status_code=500, detail="Failed to create video record in database.")
            main_logger.info(f"video_id: {video_id} - Database record created with DB ID: {video_record.id}", extra=log_req_extra)

        # 3. Add the extraction task to run in the background
        # This task will run AFTER the 'async with' block for the session has exited and committed.
        background_tasks.add_task(
            process_video_for_extraction, # This function MUST be async def
            video_id=video_id,
            original_video_path=original_video_file_path,
            video_processing_base_path=video_processing_base_path
        )
        main_logger.info(f"video_id: {video_id} - Background processing task added.", extra=log_req_extra)
        
        # 4. Return a 202 Accepted response immediately
        return JSONResponse(
            status_code=202,
            content={
                "video_id": video_id,
                "message": "Video upload accepted. Processing has been queued.",
                "original_filename_on_server": original_filename_sanitized,
            }
        )
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e: # Catch other unexpected errors
        main_logger.exception(f"video_id: {video_id} - Unexpected error during upload processing.", extra=log_req_extra)
        # Attempt to clean up directory if it exists from a partial save
        if os.path.exists(video_processing_base_path):
            try:
                shutil.rmtree(video_processing_base_path)
                main_logger.info(f"video_id: {video_id} - Cleaned up directory {video_processing_base_path} after error.", extra=log_req_extra)
            except Exception as e_rm:
                main_logger.error(f"video_id: {video_id} - Error cleaning up directory {video_processing_base_path}: {e_rm}", extra=log_req_extra)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during upload: {str(e)}")
    finally:
        if file: 
            await file.close()

if __name__ == "__main__":
    main_logger.info(f"Starting Uvicorn server. TEMP_VIDEO_DIR is set to: {TEMP_VIDEO_DIR}")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)