from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks # Import BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from dotenv import load_dotenv

from services.media_processor import process_video_for_extraction

load_dotenv()
app = FastAPI(title="ClipPilot.ai Backend")

origins = [
    "http://localhost:3000",
    "0.0.0.0"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_VIDEO_DIR = os.getenv("TEMP_VIDEO_DIR", "/tmp/clippilot_uploads_dev")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True) 


# os.makedirs(TEMP_VIDEO_DIR, exist_ok=True) 
@app.get("/ping", summary="Health check")
async def ping():
    """
    Simple health check endpoint.
    """
    return {"message": "pong from ClipPilot.ai backend"}

# @app.post("/upload", summary="Upload a video for processing")
# async def upload_video(file: UploadFile = File(...)):

#     if not file.content_type or not file.content_type.startswith("video/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
#     video_id = str(uuid.uuid4())

#     # store the video to temp dir
#     video_processing_path = os.path.join(TEMP_VIDEO_DIR, video_id)
#     os.makedirs(video_processing_path, exist_ok=True)
#     video_file_path = os.path.join(video_processing_path, file.filename if file.filename else "uploaded_video.mp4")


#     try:
#         with open(video_file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)        
#         video_clip = VideoFileClip(video_file_path)        
#         audio_file_name = "audio.wav"
#         audio_path = os.path.join(video_processing_path, audio_file_name)
#         video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
#         frames_output_dir = os.path.join(video_processing_path, "frames")
#         os.makedirs(frames_output_dir, exist_ok=True)
#         duration = video_clip.duration
#         for t_sec in range(int(duration)): 
#             frame_filename = os.path.join(frames_output_dir, f"frame_{t_sec+1:04d}.jpg")
#             video_clip.save_frame(frame_filename, t=t_sec)
#         video_clip.close()
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "video_id": video_id,
#                 "message": "Video uploaded and initial processing (audio/frames) complete.",
#                 "original_filename": file.filename,
#                 "processed_video_path": video_processing_path,
#                 "audio_file": os.path.join(video_id, audio_file_name),
#                 "frames_directory": os.path.join(video_id, "frames")
#             }
#         )
#     except Exception as e:
#         if os.path.exists(video_processing_path):
#             shutil.rmtree(video_processing_path)
#         print(f"Error processing video {video_id}: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
#     finally:
#         if file: 
#             await file.close()

@app.post("/upload", summary="Upload a video for processing")
async def upload_video(
    background_tasks: BackgroundTasks, # FastAPI will inject this
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
    video_id = str(uuid.uuid4())
    
    # Base path for all artifacts related to this video_id
    video_processing_base_path = os.path.join(TEMP_VIDEO_DIR, video_id)
    os.makedirs(video_processing_base_path, exist_ok=True)
    
    # Sanitize filename and ensure it's not empty
    original_filename_sanitized = "uploaded_video.mp4" # Default
    if file.filename:
        # Basic sanitization: replace non-alphanumeric (excluding ., _, -) with nothing
        temp_name = "".join(c for c in file.filename if c.isalnum() or c in ['.', '_', '-']).strip()
        if temp_name: # Ensure it's not empty after sanitization
            original_filename_sanitized = temp_name
            
    original_video_file_path = os.path.join(video_processing_base_path, original_filename_sanitized)
    
    try:
        # 1. Save the uploaded file to the path madde in teh beginning
        with open(original_video_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Add the extraction task to run in the background
        background_tasks.add_task(
            process_video_for_extraction, #  from services.media_extractor
            video_id=video_id,
            original_video_path=original_video_file_path,
            video_processing_base_path=video_processing_base_path
        )
        
        # 3. Return a 202 Accepted response immediately
        return JSONResponse(
            status_code=202, # HTTP 202 Accepted: Request accepted for processing, not yet completed
            content={
                "video_id": video_id,
                "message": "Video upload accepted. Frame and audio extraction has been queued.",
                "original_filename_on_server": original_filename_sanitized,
                "processing_base_path": video_processing_base_path # For reference/debugging
            }
        )
    except Exception as e:
        # If initial saving fails, attempt to clean up
        if os.path.exists(video_processing_base_path):
            shutil.rmtree(video_processing_base_path)

        print(f"ERROR during initial video save for video_id {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving uploaded video: {str(e)}")
    finally:
        if file: 
            await file.close()

if __name__ == "__main__":
    print(f"Starting Uvicorn server. TEMP_VIDEO_DIR is set to: {TEMP_VIDEO_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8001)