
import os
import json
import whisper 
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(video_id)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en") 
DEFAULT_WHISPER_MODEL = "base.en"
TRANSCRIPTS_SUBDIR = "transcripts"
TRANSCRIPT_FILENAME = "transcript_whisper.json"

def update_db_transcription_status(video_id: str, status: str, transcript_path: str = None, error_message: str = None):
    log_extra = {'video_id': video_id}
    db_update_payload = {
        "video_id": video_id,
        "step": "transcription",
        "status": status,
        "transcript_path": transcript_path,
        "error_message": error_message
    }
    logger.info(f"DB Update: {db_update_payload}", extra=log_extra)

def queue_text_embedding_task(video_id: str, transcript_data: dict):
    log_extra = {'video_id': video_id}
    logger.info(f"Placeholder: Queuing text embedding task for video_id: {video_id}", extra=log_extra)

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def run_transcription_pipeline(video_id: str, audio_file_path: str, processing_output_base_path: str, model_name: str = DEFAULT_WHISPER_MODEL):
    """
    Transcribes the given audio file using Whisper and saves the result.

    Args:
        video_id (str): Unique identifier for the video.
        audio_file_path (str): Absolute path to the input audio file (e.g., .wav).
        processing_output_base_path (str): Base directory for this video's processed artifacts
                                           (e.g., /tmp/clippilot_uploads_dev/<video_id>/).
        model_name (str): The name of the Whisper model to use.

    Returns:
        Optional[dict]: The structured transcript data if successful, otherwise None.
    """
    log_extra = {'video_id': video_id}
    logger.info(f"Starting transcription pipeline for audio: {audio_file_path} using model: {model_name}", extra=log_extra)
    update_db_transcription_status(video_id, "started")

    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found at {audio_file_path}", extra=log_extra)
        update_db_transcription_status(video_id, "failed", error_message="Audio file not found")
        return None

    transcripts_dir = os.path.join(processing_output_base_path, TRANSCRIPTS_SUBDIR)
    ensure_dir(transcripts_dir)
    transcript_output_path = os.path.join(transcripts_dir, TRANSCRIPT_FILENAME)

    try:
        logger.info(f"Loading Whisper model: {model_name}...", extra=log_extra)
        start_time = time.time()

        model = whisper.load_model(model_name)
        load_time = time.time() - start_time
        logger.info(f"Whisper model '{model_name}' loaded in {load_time:.2f} seconds.", extra=log_extra)

        logger.info(f"Starting transcription for: {audio_file_path}", extra=log_extra)
        start_time = time.time()

        result = model.transcribe(audio_file_path, fp16=False, verbose=False) # Set verbose=True for debugging
        transcribe_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcribe_time:.2f} seconds.", extra=log_extra)

        # Save the full transcript result to a JSON file
        with open(transcript_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Transcript saved to: {transcript_output_path}", extra=log_extra)
        
        update_db_transcription_status(video_id, "completed", transcript_path=transcript_output_path)


        # queue_text_embedding_task(video_id, result) # Pass the full result dictionary

        return result # Return the full transcript dictionary

    except Exception as e:
        logger.exception(f"Error during Whisper model loading or transcription for {audio_file_path}", extra=log_extra)
        update_db_transcription_status(video_id, "failed", error_message=str(e))
        return None

