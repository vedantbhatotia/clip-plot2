# # services/media_processor.py
# import os
# import subprocess
# import logging
# # Assuming transcription_service and embedding_service are in the same 'services' package
# from .transcription_service import run_transcription_pipeline
# from .embedding_service import run_vision_embedding_pipeline 
# # Remove this if you don't have database_service yet or if it's not ready
# # from . import database_service 
# # from .database_service import VideoProcessingStatus 

# # Make sure logging.basicConfig is NOT called here if main.py configures it globally.
# # This module will use a logger that inherits the root logger's configuration.
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - video_id:[%(video_id)s] - %(message)s') # DO NOT HAVE THIS HERE
# logger = logging.getLogger(__name__) # This will be 'services.media_processor'


# FRAME_RATE_STR = os.getenv("FRAME_RATE", "1/1")  # Default to 1 frame per second
# AUDIO_SAMPLE_RATE = os.getenv("AUDIO_SAMPLE_RATE", "16000") # Whisper
# AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1") 

# def ensure_dir(directory_path):
#     """Creates a directory if it doesn't exist."""
#     os.makedirs(directory_path, exist_ok=True)

# async def process_video_for_extraction(video_id: str, original_video_path: str, video_processing_base_path: str): # Made async
#     """
#     Extracts frames and audio, then triggers transcription and vision embedding.
#     """
#     log_extra = {'video_id': video_id} # For custom formatters/filters if used

#     logger.info(f"video_id: {video_id} - Starting media extraction for: {original_video_path}", extra=log_extra)

#     if not os.path.exists(original_video_path):
#         logger.error(f"video_id: {video_id} - Video file not found at {original_video_path}", extra=log_extra)
#         # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PROCESSING_FAILED, "Original video file not found")
#         return None, None # Indicate failure

#     # Standardized output subdirectories
#     frames_subdir_name = "frames"
#     audio_subdir_name = "audio"
    
#     frames_output_dir = os.path.join(video_processing_base_path, frames_subdir_name)
#     audio_output_dir = os.path.join(video_processing_base_path, audio_subdir_name)
    
#     try:
#         ensure_dir(frames_output_dir)
#         ensure_dir(audio_output_dir)
#     except OSError as e:
#         logger.exception(f"video_id: {video_id} - Failed to create output directories: {e}", extra=log_extra)
#         # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dirs: {e}")
#         return None, None


#     audio_filename = "audio.wav"
#     audio_file_path = os.path.join(audio_output_dir, audio_filename)

#     logger.info(f"video_id: {video_id} - Output frames to: {frames_output_dir}", extra=log_extra)
#     logger.info(f"video_id: {video_id} - Output audio to: {audio_file_path}", extra=log_extra)

#     # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.MEDIA_EXTRACTING)

#     try:
#         # Extract Frames
#         frames_command = [
#             "ffmpeg", "-i", original_video_path,
#             "-vf", f"fps={FRAME_RATE_STR}",
#             "-q:v", "2",
#             os.path.join(frames_output_dir, "frame_%04d.jpg"),
#             "-loglevel", "error",
#             "-hide_banner"
#         ]
#         logger.info(f"video_id: {video_id} - Executing frames command: {' '.join(frames_command)}", extra=log_extra)
#         # subprocess.run is blocking. For async, consider asyncio.create_subprocess_shell or similar
#         # For simplicity with BackgroundTasks, we keep it sync for now.
#         result_frames = subprocess.run(frames_command, check=False, capture_output=True, text=True) # check=False to inspect result
#         if result_frames.returncode != 0:
#             logger.error(f"video_id: {video_id} - Frame extraction failed. FFmpeg stderr: {result_frames.stderr}", extra=log_extra)
#             # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Frame extraction failed: {result_frames.stderr}")
#             return None, None # Critical failure
#         logger.info(f"video_id: {video_id} - Frame extraction successful.", extra=log_extra)

#         # Extract Audio
#         audio_command = [
#             "ffmpeg", "-i", original_video_path,
#             "-vn", 
#             "-acodec", "pcm_s16le",
#             "-ar", AUDIO_SAMPLE_RATE,
#             "-ac", AUDIO_CHANNELS,
#             audio_file_path,
#             "-y", 
#             "-loglevel", "error",
#             "-hide_banner"
#         ]
#         logger.info(f"video_id: {video_id} - Executing audio command: {' '.join(audio_command)}", extra=log_extra)
#         result_audio = subprocess.run(audio_command, check=False, capture_output=True, text=True)
#         if result_audio.returncode != 0:
#             logger.error(f"video_id: {video_id} - Audio extraction failed. FFmpeg stderr: {result_audio.stderr}", extra=log_extra)
#             # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Audio extraction failed: {result_audio.stderr}")
#             return None, None # Critical failure
#         logger.info(f"video_id: {video_id} - Audio extraction successful.", extra=log_extra)
        
#         logger.info(f"video_id: {video_id} - Media extraction fully completed. Frames: {frames_output_dir}, Audio: {audio_file_path}", extra=log_extra)
#         # await database_service.update_video_asset_paths(video_id, audio_path=audio_file_path, frames_dir=frames_output_dir)
#         # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.MEDIA_EXTRACTED)

#         # --- CHAIN TO TRANSCRIPTION & EMBEDDINGS ---
#         # These calls should ideally be made async if the functions they call are async
#         # and if you are using a more robust task queue or if process_video_for_extraction itself is truly async.
#         # For FastAPI's BackgroundTasks with sync functions, direct calls are okay.
        
#         transcription_ok = False
#         try:
#             logger.info(f"video_id: {video_id} - Triggering transcription pipeline.", extra=log_extra)
#             # If run_transcription_pipeline is async, you'd await it here.
#             # For now, assuming it (and its chained text_embedding) are effectively synchronous when called from BackgroundTasks
#             transcript_result_data = run_transcription_pipeline(
#                 video_id=video_id,
#                 audio_file_path=audio_file_path,
#                 processing_output_base_path=video_processing_base_path
#             ) # This function now also calls text embedding
#             if transcript_result_data: # Indicates transcription & text embedding started and reported
#                 transcription_ok = True # Assuming success if it returns data; a more robust check is needed based on actual outcome
#                 logger.info(f"video_id: {video_id} - Transcription (and text embedding) pipeline reported completion.", extra=log_extra)
#             else:
#                 logger.error(f"video_id: {video_id} - Transcription pipeline (or text embedding) failed or returned no data.", extra=log_extra)
#                 # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PARTIAL_FAILURE, "Transcription or text embedding failed")

#         except Exception as e_transcribe:
#             logger.exception(f"video_id: {video_id} - Exception during transcription/text_embedding chain.", extra=log_extra)
#             # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PARTIAL_FAILURE, f"Exception in transcription: {e_transcribe}")


#         vision_ok = False
#         try:
#             logger.info(f"video_id: {video_id} - Triggering vision embedding pipeline.", extra=log_extra)
#             # If run_vision_embedding_pipeline is async, await it
#             vision_embedding_success = run_vision_embedding_pipeline(
#                 video_id=video_id,
#                 frames_directory_path=frames_output_dir,
#                 processing_output_base_path=video_processing_base_path
#             )
#             if vision_embedding_success:
#                 vision_ok = True
#                 logger.info(f"video_id: {video_id} - Vision embedding pipeline reported completion.", extra=log_extra)
#             else:
#                 logger.error(f"video_id: {video_id} - Vision embedding pipeline failed.", extra=log_extra)
#                 # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PARTIAL_FAILURE, "Vision embedding failed")
#         except Exception as e_vision:
#             logger.exception(f"video_id: {video_id} - Exception during vision_embedding chain.", extra=log_extra)
#             # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PARTIAL_FAILURE, f"Exception in vision embedding: {e_vision}")


#         # Final status update
#         # if transcription_ok and vision_ok:
#         #     await database_service.update_video_processing_status(video_id, VideoProcessingStatus.READY_FOR_SEARCH)
#         # elif not transcription_ok and not vision_ok:
#         #     # Already marked as failed or partial failure from individual steps potentially
#         #     pass
#         # else: # One of them failed
#         #     await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PARTIAL_FAILURE, "One or more embedding steps failed")


#         return frames_output_dir, audio_file_path

#     except subprocess.CalledProcessError as e: # This might not be hit if check=False
#         logger.error(f"video_id: {video_id} - Error during FFmpeg execution for {original_video_path}: {e.stderr}", extra=log_extra)
#         # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PROCESSING_FAILED, f"FFmpeg error: {e.stderr}")
#         return None, None
#     except Exception as e:
#         logger.exception(f"video_id: {video_id} - An unexpected error occurred during media extraction for {original_video_path}", extra=log_extra)
#         # await database_service.update_video_processing_status(video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Unexpected error: {e}")
#         return None, None


# if __name__ == "__main__":
#     # This block only runs when you execute "python services/media_processor.py"
#     # It will NOT run when main.py imports this module.

#     # Basic logging setup for direct script execution, if no handlers are already configured
#     # (e.g., if main.py's basicConfig hasn't run because this is executed standalone)
#     if not logging.getLogger().hasHandlers():
#         # Define a format that can handle 'video_id' if provided by extra, or ignore if not.
#         # A more robust way is a custom formatter, but for direct testing, this is simple.
#         log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
#         if hasattr(logging.LogRecord('', 0, '', 0, '', (), None, None), 'video_id'): # Check if video_id is ever expected
#              log_format = '%(asctime)s - %(levelname)s - %(name)s - video_id:[%(video_id)s] - %(message)s'
#         logging.basicConfig(level=logging.INFO, format=log_format)

#     logger.info("Running media_processor.py directly for testing.")
    
#     # --- Configuration for direct test ---
#     test_video_id = "direct_media_processor_test_001"
#     # Create dummy file or use a real one. Path relative to this script's location if run directly
#     # Or use absolute paths.
#     # Assuming 'backend' is the CWD when running 'python services/media_processor.py' from 'backend'
#     # If rec.mp4 is in 'backend':
#     sample_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rec.mp4")) 
    
#     # Output directory for this direct test
#     # This will create it relative to where the script is run
#     direct_test_output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "direct_test_output", test_video_id))
    
#     if not os.path.exists(sample_video_path):
#         logger.error(f"Direct test failed: Sample video for direct test not found at '{sample_video_path}'. "
#                      f"Please create it or update the path.")
#     else:
#         ensure_dir(direct_test_output_base) # ensure_dir is defined in your file
#         logger.info(f"Direct test: Processing video_id='{test_video_id}', input='{sample_video_path}', output_base='{direct_test_output_base}'")
        
#         # Since process_video_for_extraction is now async, we need to run it in an event loop
#         # if called directly.
#         import asyncio
#         try:
#             asyncio.run(process_video_for_extraction(
#                 video_id=test_video_id,
#                 original_video_path=sample_video_path,
#                 video_processing_base_path=direct_test_output_base
#             ))
#             logger.info(f"Direct test: Processing finished for video_id: {test_video_id}.")
#         except Exception as e:
#             logger.exception(f"Direct test failed for video_id: {test_video_id}")














# backend/services/media_processor.py
import os
import subprocess
import logging
import asyncio # For asyncio.run in __main__ if testing directly

# --- Import Database Service ---
from . import database_service # Assuming services is a package, or use specific import
from .database_service import VideoProcessingStatus, get_db_session

# --- Import other service functions (these MUST also become async def) ---
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
        # Log or raise a more specific error if needed by the caller
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise

# --- Make this function ASYNC ---
async def process_video_for_extraction(video_id: str, original_video_path: str, video_processing_base_path: str):
    log_extra = {'video_id': video_id}
    logger.info(f"video_id: {video_id} - Starting full processing chain for: {original_video_path}", extra=log_extra)
    
    frames_output_dir = os.path.join(video_processing_base_path, "frames")
    audio_output_dir = os.path.join(video_processing_base_path, "audio")
    audio_filename = "audio.wav"
    audio_file_path = os.path.join(audio_output_dir, audio_filename)

    # Flags to track success of major pipeline branches
    media_extraction_successful = False
    transcription_and_text_embedding_successful = False
    vision_embedding_successful = False

    # --- Media Extraction Part ---
    async with get_db_session() as session:
        try:
            logger.info(f"video_id: {video_id} - Attempting to update status to MEDIA_EXTRACTING.", extra=log_extra)
            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.MEDIA_EXTRACTING
            )

            if not os.path.exists(original_video_path):
                error_msg = f"Original video file not found at {original_video_path}"
                logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return # Critical failure, stop here

            ensure_dir(frames_output_dir)
            ensure_dir(audio_output_dir)
            logger.info(f"video_id: {video_id} - Output directories ensured: Frames='{frames_output_dir}', Audio='{audio_output_dir}'.", extra=log_extra)

            # Extract Frames
            frames_command = [
                "ffmpeg", "-i", original_video_path, "-vf", f"fps={FRAME_RATE_STR}",
                "-q:v", "2", os.path.join(frames_output_dir, "frame_%04d.jpg"),
                "-loglevel", "error", "-hide_banner"
            ]
            logger.info(f"video_id: {video_id} - Executing frames command: {' '.join(frames_command)}", extra=log_extra)
            result_frames = subprocess.run(frames_command, check=False, capture_output=True, text=True)
            if result_frames.returncode != 0:
                error_msg = f"Frame extraction failed. FFmpeg stderr: {result_frames.stderr.strip()}"
                logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return
            logger.info(f"video_id: {video_id} - Frame extraction successful.", extra=log_extra)

            # Extract Audio
            audio_command = [
                "ffmpeg", "-i", original_video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", AUDIO_SAMPLE_RATE, "-ac", AUDIO_CHANNELS, audio_file_path,
                "-y", "-loglevel", "error", "-hide_banner"
            ]
            logger.info(f"video_id: {video_id} - Executing audio command: {' '.join(audio_command)}", extra=log_extra)
            result_audio = subprocess.run(audio_command, check=False, capture_output=True, text=True)
            if result_audio.returncode != 0:
                error_msg = f"Audio extraction failed. FFmpeg stderr: {result_audio.stderr.strip()}"
                logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return
            logger.info(f"video_id: {video_id} - Audio extraction successful.", extra=log_extra)

            # Update DB after successful media extraction
            await database_service.update_video_asset_paths_record(
                session, video_id, audio_path=audio_file_path, frames_dir=frames_output_dir
            )
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.MEDIA_EXTRACTED)
            media_extraction_successful = True
            logger.info(f"video_id: {video_id} - Media extraction fully completed and DB updated.", extra=log_extra)

        except Exception as e_media:
            logger.exception(f"video_id: {video_id} - Unhandled exception during media extraction block.", extra=log_extra)
            # Attempt to update status, rollback is handled by get_db_session context manager on exception
            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Media extraction critical error: {str(e_media)}"
            )
            return # Stop processing

    # --- Subsequent Processing (Transcription, Embeddings) ---
    # These are called outside the initial media extraction DB session.
    # They will manage their own DB sessions internally.
    if not media_extraction_successful:
        logger.error(f"video_id: {video_id} - Media extraction failed, skipping subsequent processing steps.", extra=log_extra)
        return

    # --- Transcription and Text Embedding ---
    try:
        logger.info(f"video_id: {video_id} - Triggering transcription (and chained text embedding) pipeline.", extra=log_extra)
        # run_transcription_pipeline is now async and should handle its own DB status updates
        # (e.g., TRANSCRIBING -> TRANSCRIBED -> TEXT_EMBEDDING -> TEXT_EMBEDDED or FAILED/PARTIAL)
        # It should return a clear success/failure indicator for text embedding part.
        text_pipeline_output_path = await run_transcription_pipeline( # This returns transcript path on success
            video_id=video_id,
            audio_file_path=audio_file_path,
            processing_output_base_path=video_processing_base_path
        )
        if text_pipeline_output_path: # If path is returned, transcription was ok, check embedding status from DB
            async with get_db_session() as check_session:
                video_status_after_text = await database_service.get_video_record_by_uuid(check_session, video_id)
                if video_status_after_text and video_status_after_text.processing_status == VideoProcessingStatus.TEXT_EMBEDDED:
                    transcription_and_text_embedding_successful = True
                    logger.info(f"video_id: {video_id} - Transcription and text embedding pipeline confirmed successful.", extra=log_extra)
                else:
                    logger.error(f"video_id: {video_id} - Transcription and text embedding pipeline did not complete successfully (status: {video_status_after_text.processing_status if video_status_after_text else 'Unknown'}).", extra=log_extra)
        else:
            logger.error(f"video_id: {video_id} - Transcription pipeline itself failed or returned no output path.", extra=log_extra)

    except Exception as e_transcribe_chain:
        logger.exception(f"video_id: {video_id} - Exception when calling transcription/text_embedding pipeline.", extra=log_extra)
        async with get_db_session() as error_session:
            await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Transcription/Text Embedding Chain EXCEPTION: {str(e_transcribe_chain)}")


    # --- Vision Embedding ---
    try:
        logger.info(f"video_id: {video_id} - Triggering vision embedding pipeline.", extra=log_extra)
        # run_vision_embedding_pipeline is now async and should handle its own DB status updates
        # (e.g., VISION_EMBEDDING -> VISION_EMBEDDED or FAILED/PARTIAL)
        # It should return a clear success/failure indicator.
        vision_pipeline_success = await run_vision_embedding_pipeline(
            video_id=video_id,
            frames_directory_path=frames_output_dir,
            processing_output_base_path=video_processing_base_path
        )
        if vision_pipeline_success:
            async with get_db_session() as check_session: # Re-fetch status after vision embedding
                video_status_after_vision = await database_service.get_video_record_by_uuid(check_session, video_id)
                if video_status_after_vision and video_status_after_vision.processing_status == VideoProcessingStatus.VISION_EMBEDDED:
                    vision_embedding_successful = True
                    logger.info(f"video_id: {video_id} - Vision embedding pipeline confirmed successful.", extra=log_extra)
                else: # Vision embedding function might have returned True but failed to update DB or other issue
                    logger.error(f"video_id: {video_id} - Vision embedding pipeline did not complete successfully (status: {video_status_after_vision.processing_status if video_status_after_vision else 'Unknown'}).", extra=log_extra)
        else:
            logger.error(f"video_id: {video_id} - Vision embedding pipeline itself reported failure.", extra=log_extra)

    except Exception as e_vision_chain:
        logger.exception(f"video_id: {video_id} - Exception when calling vision_embedding pipeline.", extra=log_extra)
        async with get_db_session() as error_session:
            await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Vision Embedding Chain EXCEPTION: {str(e_vision_chain)}")

    # --- Final Status Update Based on Outcomes ---
    async with get_db_session() as final_session:
        current_video_record = await database_service.get_video_record_by_uuid(final_session, video_id)
        # Only update if not already marked as a definitive PROCESSING_FAILED by one of the sub-pipelines
        if current_video_record and current_video_record.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
            if transcription_and_text_embedding_successful and vision_embedding_successful:
                await database_service.update_video_status_and_error(final_session, video_id, VideoProcessingStatus.READY_FOR_SEARCH)
                logger.info(f"video_id: {video_id} - All processing and embedding successful. Status set to READY_FOR_SEARCH.", extra=log_extra)
            else:
                await database_service.update_video_status_and_error(
                    final_session, video_id, VideoProcessingStatus.PARTIAL_FAILURE,
                    "One or more post-extraction processing steps (transcription/text_embedding or vision_embedding) failed."
                )
                logger.warning(f"video_id: {video_id} - Some processing steps failed. Status set to PARTIAL_FAILURE.", extra=log_extra)
    
    logger.info(f"video_id: {video_id} - process_video_for_extraction background task finished.", extra=log_extra)

# --- if __name__ == "__main__": block for direct testing ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    logger.info("Running media_processor.py directly for testing.")
    
    test_video_id = "direct_media_processor_test_002"
    # Adjust this path to an actual small video file for testing
    sample_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_assets", "sample_short.mp4"))
    direct_test_output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "direct_test_output", test_video_id))
    
    # Mock database_service for direct testing if DB is not setup or to isolate this module
    class MockDBSession:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def commit(self): logger.info("Mock DB: Commit called")
        async def rollback(self): logger.info("Mock DB: Rollback called")
        async def close(self): logger.info("Mock DB: Session closed")

    class MockDatabaseService:
        VideoProcessingStatus = database_service.VideoProcessingStatus # Use the real Enum
        async def get_db_session(self): return MockDBSession()
        async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
            logger.info(f"Mock DB: video_id={video_id}, status={status}, error='{error_msg}'")
        async def update_video_asset_paths_record(self, session, video_id, audio_path=None, frames_dir=None, transcript_path=None):
            logger.info(f"Mock DB: video_id={video_id}, audio={audio_path}, frames={frames_dir}, transcript={transcript_path}")
        async def get_video_record_by_uuid(self, session, video_id):
            logger.info(f"Mock DB: get_video_record_by_uuid for {video_id}")
            # Return a mock record that indicates success for sub-pipelines for testing final status
            class MockVideoRecord: pass
            record = MockVideoRecord()
            record.processing_status = VideoProcessingStatus.VISION_EMBEDDED # Simulate sub-steps are done
            return record


    # In a real direct test, you'd ensure DATABASE_URL is set or handle it.
    # For now, to test the flow if database_service is not fully mocked:
    if not os.getenv("DATABASE_URL"):
        logger.warning("DATABASE_URL not set for direct test. DB operations will likely fail or be skipped by database_service internal checks.")
        # To truly isolate, you'd inject a mock database_service. For now, we proceed.
        # For an isolated test, you could do:
        # database_service = MockDatabaseService() # Overwrite imported service with mock

    if not os.path.exists(sample_video_path):
        logger.error(f"Direct test failed: Sample video for direct test not found at '{sample_video_path}'. "
                     f"Please create 'backend/test_assets/sample_short.mp4' or update the path.")
    else:
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