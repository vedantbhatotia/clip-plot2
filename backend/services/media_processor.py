
# backend/services/media_processor.py
import os
import subprocess
import logging
import asyncio # For asyncio.run in __main__ if testing directly

from . import database_service 
from . import storage_service
from .database_service import VideoProcessingStatus, get_db_session

from .transcription_service import run_transcription_pipeline
from .embedding_service import run_vision_embedding_pipeline

logger = logging.getLogger(__name__) # Will be 'services.media_processor'

FRAME_RATE_STR = os.getenv("FRAME_RATE", "1/1")
AUDIO_SAMPLE_RATE = os.getenv("AUDIO_SAMPLE_RATE", "16000")
AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1")

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise # Re-raise to be handled by caller

def parse_frame_rate_to_fps(fr_str: str, video_id_for_log: str = "N/A") -> float:
    """ Parses FFmpeg frame rate string (e.g., "25", "30/1", "1/2") to FPS float. """
    log_extra = {'video_id': video_id_for_log}
    if '/' in fr_str:
        try:
            num, den = map(float, fr_str.split('/'))
            if den == 0:
                logger.warning(f"video_id: {video_id_for_log} - Invalid frame rate denominator 0 in '{fr_str}'. Defaulting to 1.0 FPS.", extra=log_extra)
                return 1.0
            return num / den
        except ValueError:
            logger.warning(f"video_id: {video_id_for_log} - Could not parse complex frame rate '{fr_str}'. Defaulting to 1.0 FPS.", extra=log_extra)
            return 1.0
    else:
        try:
            return float(fr_str)
        except ValueError:
            logger.warning(f"video_id: {video_id_for_log} - Could not parse simple frame rate '{fr_str}'. Defaulting to 1.0 FPS.", extra=log_extra)
            return 1.0

# --- Make this function ASYNC ---
# async def process_video_for_extraction(video_id: str, original_video_supabase_key: str, video_processing_base_path: str):
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting full processing chain for: {original_video_supabase_key}", extra=log_extra)
    
#     frames_output_dir = os.path.join(video_processing_base_path, "frames")
#     audio_output_dir = os.path.join(video_processing_base_path, "audio")
#     audio_filename = "audio.wav"
#     audio_file_path = os.path.join(audio_output_dir, audio_filename)

#     media_extraction_successful = False
#     # These flags will be determined by the return values of the async service calls
#     transcription_chain_success = False
#     vision_embedding_chain_success = False

#     # --- Media Extraction Part ---
#     try:
#         # --- Download original video from Supabase Storage ---
#         ensure_dir(video_processing_base_path) # Ensure the main temp processing dir exists
#         logger.info(f"Downloading original video from Supabase. Key: '{original_video_supabase_key}' to Local Temp: '{local_original_video_download_path}'", extra=log_extra)
        
#         downloaded_path = await storage_service.download_file_from_supabase(
#             bucket_name=storage_service.VIDEO_BUCKET_NAME, 
#             source_path=original_video_supabase_key,
#             local_temp_path=local_original_video_download_path
#         )
#         if not downloaded_path or not os.path.exists(local_original_video_download_path):
#             error_msg = "Failed to download original video from Supabase Storage for processing."
#             logger.error(error_msg, extra=log_extra)
#             async with get_db_session() as error_session:
#                 await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return
#         logger.info(f"Video successfully downloaded to local temp path: {local_original_video_download_path}", extra=log_extra)

#         async with get_db_session() as session:
#             try:
#                 logger.info(f"video_id: {video_id} - Updating status to MEDIA_EXTRACTING.", extra=log_extra)
#                 await database_service.update_video_status_and_error(
#                     session, video_id, VideoProcessingStatus.MEDIA_EXTRACTING
#                 )

#                 if not os.path.exists(original_video_path):
#                     error_msg = f"Original video file not found at {original_video_path}"
#                     logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                     await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                     return # Stop here

#                 ensure_dir(frames_output_dir) # ensure_dir might raise OSError
#                 ensure_dir(audio_output_dir)
#                 logger.info(f"video_id: {video_id} - Output directories ensured.", extra=log_extra)

#                 # Extract Frames
#                 frames_command = [
#                     "ffmpeg", "-i", original_video_path, "-vf", f"fps={FRAME_RATE_STR}",
#                     "-q:v", "2", os.path.join(frames_output_dir, "frame_%04d.jpg"),
#                     "-loglevel", "error", "-hide_banner"
#                 ]
#                 logger.info(f"video_id: {video_id} - Executing frames command: {' '.join(frames_command)}", extra=log_extra)
#                 result_frames = subprocess.run(frames_command, check=False, capture_output=True, text=True)
#                 if result_frames.returncode != 0:
#                     error_msg = f"Frame extraction failed. FFmpeg stderr: {result_frames.stderr.strip()}"
#                     logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                     await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                     return
#                 logger.info(f"video_id: {video_id} - Frame extraction successful.", extra=log_extra)

#                 # Extract Audio
#                 audio_command = [
#                     "ffmpeg", "-i", original_video_path, "-vn", "-acodec", "pcm_s16le",
#                     "-ar", AUDIO_SAMPLE_RATE, "-ac", AUDIO_CHANNELS, audio_file_path,
#                     "-y", "-loglevel", "error", "-hide_banner"
#                 ]
#                 logger.info(f"video_id: {video_id} - Executing audio command: {' '.join(audio_command)}", extra=log_extra)
#                 result_audio = subprocess.run(audio_command, check=False, capture_output=True, text=True)
#                 if result_audio.returncode != 0:
#                     error_msg = f"Audio extraction failed. FFmpeg stderr: {result_audio.stderr.strip()}"
#                     logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                     await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                     return
#                 logger.info(f"video_id: {video_id} - Audio extraction successful.", extra=log_extra)

#                 await database_service.update_video_asset_paths_record(
#                     session, video_id, audio_path=audio_file_path, frames_dir=frames_output_dir
#                 )
#                 await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.MEDIA_EXTRACTED)
#                 media_extraction_successful = True
#                 logger.info(f"video_id: {video_id} - Media extraction fully completed and DB updated.", extra=log_extra)

#             except OSError as e_os: # Specific exception for makedirs
#                 logger.exception(f"video_id: {video_id} - OSError during directory creation for media extraction.", extra=log_extra)
#                 await database_service.update_video_status_and_error(
#                     session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Media extraction directory error: {str(e_os)}"
#                 )
#                 return # Stop processing
#             except Exception as e_media:
#                 logger.exception(f"video_id: {video_id} - Unhandled exception during media extraction block.", extra=log_extra)
#                 await database_service.update_video_status_and_error(
#                     session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Media extraction critical error: {str(e_media)}"
#                 )
#                 return # Stop processing

#         # Subsequent processing happens only if media extraction was successful
#         # These calls are outside the above 'async with' block, so they will use their own sessions.
#         if not media_extraction_successful:
#             logger.error(f"video_id: {video_id} - Media extraction failed, skipping subsequent processing steps.", extra=log_extra)
#             return

#         # --- Transcription and Text Embedding ---
#         try:
#             logger.info(f"video_id: {video_id} - Triggering transcription (and chained text embedding) pipeline.", extra=log_extra)
#             # run_transcription_pipeline is async and internally updates DB status from TRANSCRIBING -> TRANSCRIBED -> TEXT_EMBEDDING -> TEXT_EMBEDDED or fails
#             # It should return True if TEXT_EMBEDDED status was successfully set, False otherwise.
#             transcript_pipeline_returned_success = await run_transcription_pipeline(
#                 video_id=video_id,
#                 audio_file_path=audio_file_path,
#                 processing_output_base_path=video_processing_base_path
#             )
#             if transcript_pipeline_returned_success:
#                 transcription_and_text_embedding_successful = True # Assume success if the function returns True
#                 logger.info(f"video_id: {video_id} - Transcription and text embedding pipeline reported success.", extra=log_extra)
#             else:
#                 logger.error(f"video_id: {video_id} - Transcription and text embedding pipeline reported failure. Check earlier logs for details.", extra=log_extra)
#                 # The sub-pipeline should have set the DB status to FAILED or PARTIAL_FAILURE
#         except Exception as e_transcribe_chain:
#             logger.exception(f"video_id: {video_id} - Top-level exception when calling transcription/text_embedding pipeline.", extra=log_extra)
#             async with get_db_session() as error_session: # New session for this isolated error update
#                 await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Transcription/Text Embedding Chain EXCEPTION: {str(e_transcribe_chain)}")

#     # --- Vision Embedding ---
#         try:
#             effective_fps = parse_frame_rate_to_fps(FRAME_RATE_STR, video_id_for_log=video_id) # Calculate FPS
#             logger.info(f"video_id: {video_id} - Triggering vision embedding pipeline with effective_fps: {effective_fps}.", extra=log_extra)
#             # run_vision_embedding_pipeline is async and internally updates DB status from VISION_EMBEDDING -> VISION_EMBEDDED or fails
#             # It should return True on success.
#             vision_pipeline_returned_success = await run_vision_embedding_pipeline(
#                 video_id=video_id,
#                 frames_directory_path=frames_output_dir,
#                 processing_output_base_path=video_processing_base_path,
#                 effective_fps=effective_fps # Pass calculated FPS
#             )
#             if vision_pipeline_returned_success:
#                 vision_embedding_successful = True # Assume success if the function returns True
#                 logger.info(f"video_id: {video_id} - Vision embedding pipeline reported success.", extra=log_extra)
#             else:
#                 logger.error(f"video_id: {video_id} - Vision embedding pipeline reported failure. Check earlier logs for details.", extra=log_extra)
#                 # The sub-pipeline should have set the DB status to FAILED or PARTIAL_FAILURE
#         except Exception as e_vision_chain:
#             logger.exception(f"video_id: {video_id} - Top-level exception when calling vision_embedding pipeline.", extra=log_extra)
#             async with get_db_session() as error_session: # New session for this isolated error update
#                 await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Vision Embedding Chain EXCEPTION: {str(e_vision_chain)}")

#         # --- Final Status Update Based on Outcomes ---
#         async with get_db_session() as final_session:
#             # Re-fetch the record to get the most current status set by sub-pipelines
#             current_video_record = await database_service.get_video_record_by_uuid(final_session, video_id)
            
#             if current_video_record and current_video_record.processing_status == VideoProcessingStatus.PROCESSING_FAILED:
#                 logger.error(f"video_id: {video_id} - Processing was already marked as FAILED. No further status update.", extra=log_extra)
#             elif transcription_and_text_embedding_successful and vision_embedding_successful:
#                 await database_service.update_video_status_and_error(final_session, video_id, VideoProcessingStatus.READY_FOR_SEARCH)
#                 logger.info(f"video_id: {video_id} - All processing and embedding successful. Status set to READY_FOR_SEARCH.", extra=log_extra)
#             else:
#                 # If not completely successful and not already marked as failed by a sub-process
#                 error_summary = "One or more core processing steps (transcription/text_embedding or vision_embedding) failed."
#                 if not transcription_and_text_embedding_successful:
#                     error_summary += " Text processing branch failed."
#                 if not vision_embedding_successful:
#                     error_summary += " Vision processing branch failed."
                
#                 logger.warning(f"video_id: {video_id} - Setting status to PARTIAL_FAILURE. Details: {error_summary}", extra=log_extra)
#                 await database_service.update_video_status_and_error(
#                     final_session, video_id, VideoProcessingStatus.PARTIAL_FAILURE, error_summary
#                 )
#     finally:
#         pass    
    
#     logger.info(f"video_id: {video_id} - process_video_for_extraction background task finished.", extra=log_extra)

async def process_video_for_extraction(
    video_id: str, 
    original_video_supabase_key: str, # This is now the Supabase key
    video_processing_base_path: str # Local temporary path for this task's artifacts
):
    log_extra = {'video_id': video_id}
    logger.info(f"Starting full processing chain. Supabase Key for original video: {original_video_supabase_key}", extra=log_extra)
    
    # Define local paths for intermediate files
    frames_output_dir = os.path.join(video_processing_base_path, "frames")
    audio_output_dir = os.path.join(video_processing_base_path, "audio")
    audio_filename = "audio.wav"
    local_extracted_audio_path = os.path.join(audio_output_dir, audio_filename) # Local path for extracted audio

    # Define local path for the downloaded original video
    original_filename_from_key = os.path.basename(original_video_supabase_key) # e.g., "myvideo.mp4"
    local_original_video_download_path = os.path.join(video_processing_base_path, f"downloaded_{original_filename_from_key}")

    media_extraction_successful = False
    transcription_chain_success = False
    vision_embedding_chain_success = False

    try:
        # --- Download original video from Supabase Storage ---
        ensure_dir(video_processing_base_path) # Ensure the main temp processing dir exists
        logger.info(f"Downloading original video from Supabase. Key: '{original_video_supabase_key}' to Local Temp: '{local_original_video_download_path}'", extra=log_extra)
        
        downloaded_path = await storage_service.download_file_from_supabase(
            bucket_name=storage_service.VIDEO_BUCKET_NAME, 
            source_path=original_video_supabase_key,
            local_temp_path=local_original_video_download_path
        )
        if not downloaded_path or not os.path.exists(local_original_video_download_path):
            error_msg = "Failed to download original video from Supabase Storage for processing."
            logger.error(error_msg, extra=log_extra)
            async with get_db_session() as error_session:
                await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            return
        logger.info(f"Video successfully downloaded to local temp path: {local_original_video_download_path}", extra=log_extra)
        
        # --- Media Extraction Part (uses local_original_video_download_path) ---
        async with get_db_session() as session:
            try:
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.MEDIA_EXTRACTING)
                ensure_dir(frames_output_dir); ensure_dir(audio_output_dir)

                # Extract Frames
                frames_command = ["ffmpeg", "-i", local_original_video_download_path, "-vf", f"fps={FRAME_RATE_STR}", "-q:v", "2", os.path.join(frames_output_dir, "frame_%04d.jpg"), "-loglevel", "error", "-hide_banner"]
                logger.info(f"Executing frames command: {' '.join(frames_command)}", extra=log_extra)
                result_frames = subprocess.run(frames_command, check=False, capture_output=True, text=True)
                if result_frames.returncode != 0:
                    error_msg = f"Frame extraction failed. FFmpeg stderr: {result_frames.stderr.strip()}"
                    logger.error(error_msg, extra=log_extra); await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg); return
                logger.info("Frame extraction successful.", extra=log_extra)

                # Extract Audio
                audio_command = ["ffmpeg", "-i", local_original_video_download_path, "-vn", "-acodec", "pcm_s16le", "-ar", AUDIO_SAMPLE_RATE, "-ac", AUDIO_CHANNELS, local_extracted_audio_path, "-y", "-loglevel", "error", "-hide_banner"]
                logger.info(f"Executing audio command: {' '.join(audio_command)}", extra=log_extra)
                result_audio = subprocess.run(audio_command, check=False, capture_output=True, text=True)
                if result_audio.returncode != 0:
                    error_msg = f"Audio extraction failed. FFmpeg stderr: {result_audio.stderr.strip()}"
                    logger.error(error_msg, extra=log_extra); await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg); return
                logger.info(f"Audio extraction successful. Saved to local: {local_extracted_audio_path}", extra=log_extra)

                # Store LOCAL paths for these intermediate files in DB.
                # These files are temporary to this processing run.
                await database_service.update_video_asset_paths_record(session, video_id, audio_path=local_extracted_audio_path, frames_dir=frames_output_dir)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.MEDIA_EXTRACTED)
                media_extraction_successful = True
            except OSError as e_os:
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Media extraction directory error: {str(e_os)}")
                return
            except Exception as e_media:
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Media extraction critical error: {str(e_media)}")
                return

        if not media_extraction_successful: return

        # --- Transcription and Text Embedding (uses local_extracted_audio_path) ---
        try:
            transcript_pipeline_output_path = await run_transcription_pipeline(
                video_id=video_id, audio_file_path=local_extracted_audio_path, 
                processing_output_base_path=video_processing_base_path
            )
            if transcript_pipeline_output_path: # This is the local path to transcript.json
                transcription_and_text_embedding_successful = True
                logger.info(f"Transcription & text embedding pipeline success. Transcript at: {transcript_pipeline_output_path}", extra=log_extra)
            else: logger.error("Transcription & text embedding pipeline reported failure.", extra=log_extra)
        except Exception as e_transcribe_chain:
            logger.exception("Top-level exception: transcription/text_embedding pipeline.", extra=log_extra)
            async with get_db_session() as error_session: await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Transcription Chain EXCEPTION: {str(e_transcribe_chain)}")

        # --- Vision Embedding (uses local frames_output_dir) ---
        try:
            effective_fps = parse_frame_rate_to_fps(FRAME_RATE_STR, video_id_for_log=video_id)
            vision_pipeline_returned_success = await run_vision_embedding_pipeline(
                video_id=video_id, frames_directory_path=frames_output_dir, 
                processing_output_base_path=video_processing_base_path,
                effective_fps=effective_fps
            )
            if vision_pipeline_returned_success: vision_embedding_successful = True
            else: logger.error("Vision embedding pipeline reported failure.", extra=log_extra)
        except Exception as e_vision_chain:
            logger.exception("Top-level exception: vision_embedding pipeline.", extra=log_extra)
            async with get_db_session() as error_session: await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Vision Embedding Chain EXCEPTION: {str(e_vision_chain)}")

        # --- Final Status Update ---
        async with get_db_session() as final_session:
            current_video_record = await database_service.get_video_record_by_uuid(final_session, video_id)
            if current_video_record and current_video_record.processing_status == VideoProcessingStatus.PROCESSING_FAILED:
                logger.error("Processing already marked FAILED by a sub-step.", extra=log_extra)
            elif transcription_and_text_embedding_successful and vision_embedding_successful:
                await database_service.update_video_status_and_error(final_session, video_id, VideoProcessingStatus.READY_FOR_SEARCH)
                logger.info("All processing successful. Status: READY_FOR_SEARCH.", extra=log_extra)
            else:
                # ... (existing partial failure logic) ...
                error_summary = "One or more core processing steps failed."
                if not transcription_and_text_embedding_successful: error_summary += " Text processing branch failed."
                if not vision_embedding_successful: error_summary += " Vision processing branch failed."
                logger.warning(f"Setting status to PARTIAL_FAILURE. Details: {error_summary}", extra=log_extra)
                await database_service.update_video_status_and_error(final_session, video_id, VideoProcessingStatus.PARTIAL_FAILURE, error_summary)
    
    finally:
        # Clean up the initially downloaded original video file from local temp storage
        if os.path.exists(local_original_video_download_path):
            try:
                os.remove(local_original_video_download_path)
                logger.info(f"Cleaned up local temp original video: {local_original_video_download_path}", extra=log_extra)
            except OSError as e_remove:
                logger.error(f"Failed to remove temp original video {local_original_video_download_path}: {e_remove}", extra=log_extra)
    logger.info(f"process_video_for_extraction background task finished.", extra=log_extra)

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=log_format) # DEBUG for direct tests

    logger.info("Running media_processor.py directly for testing.")
    
    test_video_id = "direct_media_processor_test_003"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    sample_video_path = os.path.join(project_root, "test_assets", "sample_short.mp4")
    direct_test_output_base = os.path.join(project_root, "direct_test_output", test_video_id, "media_processor_module")
    
    # Mock database_service for direct testing
    from contextlib import asynccontextmanager # For mock get_db_session
    from typing import AsyncGenerator # For mock get_db_session

    class MockDBSession:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        # No commit/rollback needed as we are not really changing state in this simple mock
    
    original_db_service_ref = database_service # Save original
    class MockDatabaseServiceModule:
        VideoProcessingStatus = original_db_service_ref.VideoProcessingStatus
        @asynccontextmanager
        async def get_db_session(self) -> AsyncGenerator[MockDBSession, None]:
            logger.debug("MOCK DB: get_db_session called")
            yield MockDBSession()
        async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
            logger.info(f"MOCK DB (media_proc_test): video_id={video_id}, status={status.value}, error='{error_msg}'")
            return True
        async def update_video_asset_paths_record(self, session, video_id, audio_path=None, frames_dir=None, transcript_path=None):
            logger.info(f"MOCK DB (media_proc_test): video_id={video_id}, audio={audio_path}, frames={frames_dir}, transcript={transcript_path}")
            return True
        async def get_video_record_by_uuid(self, session, video_id): # Needs to return a mock object
            logger.info(f"MOCK DB (media_proc_test): get_video_record_by_uuid for {video_id}")
            class MockVideoRecord:
                def __init__(self):
                    self.processing_status = VideoProcessingStatus.VISION_EMBEDDED # Simulate successful sub-steps for final check
            return MockVideoRecord()
    
    database_service = MockDatabaseServiceModule() # Monkey patch

    # Mock the chained service calls for isolated testing of media_processor logic
    original_run_transcription_pipeline = run_transcription_pipeline
    original_run_vision_embedding_pipeline = run_vision_embedding_pipeline

    async def mock_run_transcription_pipeline(video_id, audio_file_path, processing_output_base_path):
        logger.info(f"MOCK run_transcription_pipeline called for video_id: {video_id}")
        # Simulate that this function internally updated its status to TEXT_EMBEDDED
        async with database_service.get_db_session() as s: # Uses the mocked get_db_session
             await database_service.update_video_status_and_error(s, video_id, VideoProcessingStatus.TEXT_EMBEDDED)
        return f"{processing_output_base_path}/transcripts/mock_transcript.json" # Return a mock path

    async def mock_run_vision_embedding_pipeline(video_id, frames_directory_path, processing_output_base_path, effective_fps):
        logger.info(f"MOCK run_vision_embedding_pipeline called for video_id: {video_id} with fps: {effective_fps}")
        async with database_service.get_db_session() as s:
             await database_service.update_video_status_and_error(s, video_id, VideoProcessingStatus.VISION_EMBEDDED)
        return True # Indicate success

    run_transcription_pipeline = mock_run_transcription_pipeline
    run_vision_embedding_pipeline = mock_run_vision_embedding_pipeline


    if not os.getenv("DATABASE_URL"): # Check is still relevant if mock is not fully comprehensive
        logger.warning("DATABASE_URL not set for direct test. Using MOCK database_service.")

    if not os.path.exists(sample_video_path):
        # Attempt to create a dummy video if sample_short.mp4 is not present
        sample_video_dir = os.path.dirname(sample_video_path)
        os.makedirs(sample_video_dir, exist_ok=True)
        logger.warning(f"Sample video not found at '{sample_video_path}'. Attempting to create a dummy FFmpeg video.")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=120x90:rate=10", # Small and short
                "-f", "lavfi", "-i", "sine=frequency=300:duration=5:sample_rate=16000",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", sample_video_path
            ], check=True, capture_output=True, text=True)
            logger.info(f"Created dummy test video: {sample_video_path}")
        except Exception as e_ffmpeg_dummy:
            logger.error(f"Direct test failed: Could not create dummy test video with FFmpeg: {e_ffmpeg_dummy}. "
                         f"Please create '{sample_video_path}' or update the path.")
            exit() # Exit if no video to test with

    ensure_dir(direct_test_output_base)
    logger.info(f"Direct test: Processing video_id='{test_video_id}', input='{sample_video_path}', output_base='{direct_test_output_base}'")
    
    try:
        asyncio.run(process_video_for_extraction(
            video_id=test_video_id,
            original_video_path=sample_video_path,
            video_processing_base_path=direct_test_output_base
        ))
        logger.info(f"Direct test: Processing finished for video_id: {test_video_id}.")
    except Exception as e:
        logger.exception(f"Direct test failed for video_id: {test_video_id}")
    finally:
        # Restore original services
        database_service = original_db_service_ref
        run_transcription_pipeline = original_run_transcription_pipeline
        run_vision_embedding_pipeline = original_run_vision_embedding_pipeline
        logger.info("Restored original services after direct test.")
        # Optional: Clean up direct_test_output directory
        # import shutil
        # if os.path.exists(direct_test_output_base):
        #     shutil.rmtree(direct_test_output_base)
        #     logger.info(f"Cleaned up direct test output directory: {direct_test_output_base}")