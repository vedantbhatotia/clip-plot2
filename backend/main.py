from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import uuid
from moviepy import VideoFileClip
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="ClipPilot.ai Backend")
# TEMP_VIDEO_DIR = os.getenv("TEMP_VIDEO_DIR", "/tmp/clippilot_uploads_dev")
TEMP_VIDEO_DIR="/tmp/clippilot_uploads_dev"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True) 
@app.get("/ping", summary="Health check")
async def ping():
    """
    Simple health check endpoint.
    """
    return {"message": "pong from ClipPilot.ai backend"}

@app.post("/upload", summary="Upload a video for processing")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    video_id = str(uuid.uuid4())
    video_processing_path = os.path.join(TEMP_VIDEO_DIR, video_id)
    os.makedirs(video_processing_path, exist_ok=True)
    video_file_path = os.path.join(video_processing_path, file.filename if file.filename else "uploaded_video.mp4")
    try:
        with open(video_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)        
        video_clip = VideoFileClip(video_file_path)        
        audio_file_name = "audio.wav"
        audio_path = os.path.join(video_processing_path, audio_file_name)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        frames_output_dir = os.path.join(video_processing_path, "frames")
        os.makedirs(frames_output_dir, exist_ok=True)
        duration = video_clip.duration
        for t_sec in range(int(duration)): 
            frame_filename = os.path.join(frames_output_dir, f"frame_{t_sec+1:04d}.jpg")
            video_clip.save_frame(frame_filename, t=t_sec)
        video_clip.close()
        return JSONResponse(
            status_code=200,
            content={
                "video_id": video_id,
                "message": "Video uploaded and initial processing (audio/frames) complete.",
                "original_filename": file.filename,
                "processed_video_path": video_processing_path,
                "audio_file": os.path.join(video_id, audio_file_name),
                "frames_directory": os.path.join(video_id, "frames")
            }
        )
    except Exception as e:
        if os.path.exists(video_processing_path):
            shutil.rmtree(video_processing_path)
        print(f"Error processing video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if file: 
            await file.close()

if __name__ == "__main__":
    print(f"Starting Uvicorn server. TEMP_VIDEO_DIR is set to: {TEMP_VIDEO_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)