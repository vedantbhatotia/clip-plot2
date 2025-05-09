# services/media_extractor.py
import os
import subprocess
import logging
from .transcription_service import run_transcription_pipeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - video_id:[%(video_id)s] - %(message)s')
logger = logging.getLogger(__name__)


FRAME_RATE_STR = os.getenv("FRAME_RATE", "1/1")  # Default to 1 frame per second
AUDIO_SAMPLE_RATE = os.getenv("AUDIO_SAMPLE_RATE", "16000") # Whisper
AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1") 

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def process_video_for_extraction(video_id: str, original_video_path: str, video_processing_base_path: str):
    """
    Extracts frames and audio from the given video file using FFmpeg.
    Outputs are saved into 'frames' and 'audio' subdirectories within video_processing_base_path.

    Args:
        video_id (str): Unique identifier for the video, used for logging and context.
        original_video_path (str): Absolute path to the input video file.
        video_processing_base_path (str): Absolute path to the base directory for this specific video's
                                          processed artifacts (e.g., /tmp/clippilot_uploads_dev/<video_id>/).
    Returns:
        tuple: (absolute_frames_directory_path, absolute_audio_file_path) or (None, None) on failure.
    """
    log_extra = {'video_id': video_id}

    if not os.path.exists(original_video_path):
        logger.error(f"Video file not found at {original_video_path}", extra=log_extra)
        return None, None

    logger.info(f"Starting media extraction for: {original_video_path}", extra=log_extra)

    # standardized output subdirectories for frames and audio
    frames_subdir_name = "frames"
    audio_subdir_name = "audio"
    
    frames_output_dir = os.path.join(video_processing_base_path, frames_subdir_name)
    audio_output_dir = os.path.join(video_processing_base_path, audio_subdir_name)
    
    ensure_dir(frames_output_dir)
    ensure_dir(audio_output_dir)

    # Standardized audio output name
    audio_filename = "audio.wav"
    audio_file_path = os.path.join(audio_output_dir, audio_filename)

    logger.info(f"Output frames to: {frames_output_dir}", extra=log_extra)
    logger.info(f"Output audio to: {audio_file_path}", extra=log_extra)

    try:
        # Extract Frames
        frames_command = [
            "ffmpeg", "-i", original_video_path,
            "-vf", f"fps={FRAME_RATE_STR}",
            "-q:v", "2", # Quality for JPG output (2-5 is good)
            os.path.join(frames_output_dir, "frame_%04d.jpg"),
            "-loglevel", "error",
            "-hide_banner"
        ]
        logger.info(f"Executing frames command: {' '.join(frames_command)}", extra=log_extra)
        subprocess.run(frames_command, check=True, capture_output=True, text=True)
        logger.info(f"Frame extraction successful.", extra=log_extra)

        # Extract Audio
        audio_command = [
            "ffmpeg", "-i", original_video_path,
            "-vn", # no video output
            "-acodec", "pcm_s16le", # standard WAV codec
            "-ar", AUDIO_SAMPLE_RATE,
            "-ac", AUDIO_CHANNELS,
            audio_file_path,
            "-y", # Overwrite output file without asking
            "-loglevel", "error",
            "-hide_banner"
        ]
        logger.info(f"Executing audio command: {' '.join(audio_command)}", extra=log_extra)
        subprocess.run(audio_command, check=True, capture_output=True, text=True)
        logger.info(f"Audio extraction successful.", extra=log_extra)
        
       
        logger.info(f"Media extraction completed for video {video_id}.", extra=log_extra)

        logger.info(f"Audio extraction successful.", extra=log_extra)
    
    # Update DB for media extraction completion
        # update_db_status(video_id, "extraction", "completed", 
        #                 data={"frames_path": frames_output_dir, "audio_path": audio_file_path}) 
        logger.info(f"Media extraction completed. Frames: {frames_output_dir}, Audio: {audio_file_path}", extra=log_extra)

        # --- CHAIN TO TRANSCRIPTION ---
        logger.info(f"Triggering transcription pipeline for video_id: {video_id}", extra=log_extra)
        transcript_result = run_transcription_pipeline(
            video_id=video_id,
            audio_file_path=audio_file_path, # Path to the .wav file just created
            processing_output_base_path=video_processing_base_path # The base path like /tmp/.../<video_id>/ (uniqueness checked from fastAPI, maybe?)
        )



        if transcript_result:
            logger.info(f"Transcription pipeline completed successfully for video_id: {video_id}", extra=log_extra)
        else:
            logger.error(f"Transcription pipeline failed for video_id: {video_id}", extra=log_extra)


        return frames_output_dir, audio_file_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during FFmpeg execution for {original_video_path}: {e.stderr}", extra=log_extra)
        return None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during media extraction for {original_video_path}", extra=log_extra)
        return None, None

process_video_for_extraction("12","../rec.mp4","../test_output")