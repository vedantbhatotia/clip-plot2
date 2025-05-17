# # backend/services/transcription_service.py
# import os
# import json
# import whisper # OpenAI's Whisper model
# import logging
# import time
# from typing import Optional, Dict, Any # For type hinting

# # --- Add these imports ---
# from . import database_service
# from .database_service import VideoProcessingStatus, get_db_session
# # --- End Add ---
# from .embedding_service import run_text_embedding_pipeline # This will also need to be async def

# logger = logging.getLogger(__name__) # Will be 'services.transcription_service'

# # --- Configuration ---
# DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
# TRANSCRIPTS_SUBDIR = "transcripts"
# TRANSCRIPT_FILENAME = "transcript_whisper.json"

# # Old placeholder functions - to be removed or commented out
# # def update_db_transcription_status(...):
# # def queue_text_embedding_task(...):

# def ensure_dir(directory_path):
#     """Creates a directory if it doesn't exist."""
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise


# # --- Make this function ASYNC ---
# async def run_transcription_pipeline(
#     video_id: str,
#     audio_file_path: str,
#     processing_output_base_path: str,
#     model_name: str = DEFAULT_WHISPER_MODEL
# ) -> Optional[str]: # Return transcript path on success, None on failure of this chain
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting transcription pipeline for audio: {audio_file_path} using model: {model_name}", extra=log_extra)
    
#     transcript_output_path = os.path.join(processing_output_base_path, TRANSCRIPTS_SUBDIR, TRANSCRIPT_FILENAME)
    
#     transcription_succeeded = False
#     whisper_result_data: Optional[Dict[str, Any]] = None

#     # --- Transcription Step ---
#     async with get_db_session() as session:
#         try:
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.TRANSCRIBING)

#             if not os.path.exists(audio_file_path):
#                 error_msg = f"Audio file not found at {audio_file_path}"
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None # Critical failure

#             ensure_dir(os.path.dirname(transcript_output_path))

#             logger.info(f"video_id: {video_id} - Loading Whisper model: {model_name}...", extra=log_extra)
#             start_time = time.time()
#             # whisper.load_model is synchronous / CPU-GPU bound
#             model = whisper.load_model(model_name)
#             load_time = time.time() - start_time
#             logger.info(f"video_id: {video_id} - Whisper model '{model_name}' loaded in {load_time:.2f} seconds.", extra=log_extra)

#             logger.info(f"video_id: {video_id} - Starting transcription for: {audio_file_path}", extra=log_extra)
#             start_time = time.time()
#             # model.transcribe is synchronous / CPU-GPU bound
#             whisper_result_data = model.transcribe(audio_file_path, fp16=False, verbose=False)
#             transcribe_time = time.time() - start_time
#             logger.info(f"video_id: {video_id} - Transcription completed in {transcribe_time:.2f} seconds.", extra=log_extra)

#             with open(transcript_output_path, "w", encoding="utf-8") as f:
#                 json.dump(whisper_result_data, f, indent=2, ensure_ascii=False)
#             logger.info(f"video_id: {video_id} - Transcript saved to: {transcript_output_path}", extra=log_extra)
            
#             await database_service.update_video_asset_paths_record(session, video_id, transcript_path=transcript_output_path)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.TRANSCRIBED)
#             transcription_succeeded = True
#             logger.info(f"video_id: {video_id} - Transcription step successful and DB updated.", extra=log_extra)

#         except Exception as e:
#             logger.exception(f"video_id: {video_id} - Error during Whisper model loading or transcription.", extra=log_extra)
#             # Rollback is handled by get_db_session context manager
#             # Attempt to update status in a new session if the current one is compromised,
#             # or ensure the update happens within the same session if it's still valid
#             try: # Nested try for status update robustness
#                 await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Transcription error: {str(e)}")
#             except Exception as db_err:
#                  logger.error(f"video_id: {video_id} - Additionally, failed to update DB status after transcription error: {db_err}", extra=log_extra)
#             return None # Stop if transcription fails

#     # --- Text Embedding Step (Chained after successful transcription) ---
#     # This step will manage its own DB session.
#     if not transcription_succeeded:
#         logger.error(f"video_id: {video_id} - Transcription failed, skipping text embedding.", extra=log_extra)
#         return None

#     text_embedding_succeeded = False
#     try:
#         logger.info(f"video_id: {video_id} - Triggering text embedding pipeline.", extra=log_extra)
#         # run_text_embedding_pipeline is now async and handles its own DB updates for TEXT_EMBEDDING -> TEXT_EMBEDDED or FAILED
#         embedding_success_flag = await run_text_embedding_pipeline(
#             video_id=video_id,
#             transcript_file_path=transcript_output_path, # Pass path to the saved transcript
#             processing_output_base_path=processing_output_base_path
#         )
#         if embedding_success_flag: # This function should return True on success
#             text_embedding_succeeded = True
#             logger.info(f"video_id: {video_id} - Text embedding pipeline reported success.", extra=log_extra)
#             # The status in DB should now be TEXT_EMBEDDED (set by run_text_embedding_pipeline)
#         else:
#             logger.error(f"video_id: {video_id} - Text embedding pipeline reported failure. Status should reflect this.", extra=log_extra)
#             # Status in DB should be PROCESSING_FAILED or PARTIAL_FAILURE (set by run_text_embedding_pipeline)

#     except Exception as e_embed:
#         logger.exception(f"video_id: {video_id} - Exception calling text embedding pipeline.", extra=log_extra)
#         # Attempt to update status to FAILED if embedding function didn't catch and update
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Text Embedding Chain EXCEPTION: {str(e_embed)}")

#     if transcription_succeeded and text_embedding_succeeded:
#         logger.info(f"video_id: {video_id} - Transcription and text embedding chain completed successfully.", extra=log_extra)
#         return transcript_output_path # Return path to transcript if all good
#     else:
#         logger.error(f"video_id: {video_id} - Transcription and text embedding chain DID NOT complete successfully.", extra=log_extra)
#         return None


# # --- if __name__ == "__main__": block for direct testing ---
# if __name__ == "__main__":
#     if not logging.getLogger().hasHandlers():
#         log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
#         logging.basicConfig(level=logging.INFO, format=log_format)

#     logger.info("Running transcription_service.py directly for testing.")
    
#     test_video_id = "direct_transcription_test_001"
#     # For direct testing, you need a dummy audio file and a base path.
#     # Example: Create a dummy processing directory structure
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root_dir = os.path.join(current_dir, "..") # Assuming services is one level down from backend root
    
#     test_base_path = os.path.join(project_root_dir, "direct_test_output", test_video_id)
#     test_audio_dir = os.path.join(test_base_path, "audio")
#     ensure_dir(test_audio_dir)
#     test_audio_file = os.path.join(test_audio_dir, "sample_audio.wav")

#     # Mock database_service for direct testing if DB is not setup or to isolate this module
#     class MockDBSession: # Simplified mock session
#         async def __aenter__(self): return self
#         async def __aexit__(self, exc_type, exc, tb): pass
#         async def commit(self): logger.info("Mock DB (transcription_test): Commit called")
#         async def rollback(self): logger.info("Mock DB (transcription_test): Rollback called")
#         async def close(self): logger.info("Mock DB (transcription_test): Session closed")

#     original_db_service = database_service # Keep a reference to the original
#     class MockDatabaseServiceModule:
#         VideoProcessingStatus = original_db_service.VideoProcessingStatus
#         async def get_db_session(self): return MockDBSession() # Use the simplified mock session
#         async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
#             logger.info(f"Mock DB (transcription_test): video_id={video_id}, status={status.value}, error='{error_msg}'")
#         async def update_video_asset_paths_record(self, session, video_id, transcript_path=None, **kwargs):
#             logger.info(f"Mock DB (transcription_test): video_id={video_id}, transcript_path={transcript_path}")
    
#     # Temporarily replace database_service with the mock for this direct test
#     database_service = MockDatabaseServiceModule()


#     # Create a short, silent dummy WAV file for testing if one doesn't exist
#     if not os.path.exists(test_audio_file):
#         logger.info(f"Creating dummy audio file for testing: {test_audio_file}")
#         try:
#             ffmpeg_command = [
#                 "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
#                 "-t", "2",  # Duration 2 seconds
#                 "-acodec", "pcm_s16le", test_audio_file, "-y", "-loglevel", "error"
#             ]
#             subprocess.run(ffmpeg_command, check=True)
#             logger.info(f"Dummy audio file created: {test_audio_file}")
#         except Exception as e_ffmpeg:
#             logger.error(f"Could not create dummy audio file using FFmpeg. Please provide a sample '{test_audio_file}'. Error: {e_ffmpeg}")
#             # exit(1) # Exit if dummy audio can't be created and is needed

#     if os.path.exists(test_audio_file):
#         logger.info(f"Direct test: Transcribing video_id='{test_video_id}', audio='{test_audio_file}'")
#         import asyncio
#         try:
#             # Mock run_text_embedding_pipeline as it's external to this module's direct test focus
#             original_run_text_embedding = run_text_embedding_pipeline
#             async def mock_run_text_embedding(video_id, transcript_file_path, processing_output_base_path):
#                 logger.info(f"Mock run_text_embedding_pipeline called for video_id: {video_id}")
#                 # Simulate successful embedding and DB update by that function
#                 async with get_db_session() as session: # Use our mocked get_db_session
#                     await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.TEXT_EMBEDDED)
#                 return True # Indicate success
            
#             # Temporarily patch run_text_embedding_pipeline
#             run_text_embedding_pipeline = mock_run_text_embedding

#             transcript_path = asyncio.run(run_transcription_pipeline(
#                 video_id=test_video_id,
#                 audio_file_path=test_audio_file,
#                 processing_output_base_path=test_base_path,
#                 model_name="tiny.en" # Use a small model for quick direct testing
#             ))
#             if transcript_path:
#                 logger.info(f"Direct test: Transcription pipeline successful. Transcript at: {transcript_path}")
#             else:
#                 logger.error("Direct test: Transcription pipeline failed.")
        
#         except Exception as e:
#             logger.exception(f"Direct test failed for video_id: {test_video_id}")
#         finally:
#             # Restore original run_text_embedding_pipeline if it was mocked
#             run_text_embedding_pipeline = original_run_text_embedding
#             database_service = original_db_service # Restore original db service
#             # Clean up dummy files if desired
#             # import shutil
#             # if os.path.exists(os.path.join(project_root_dir, "direct_test_output")):
#             #     shutil.rmtree(os.path.join(project_root_dir, "direct_test_output"))
#             # logger.info("Cleaned up direct test output directory.")
#     else:
#         logger.warning(f"Test audio file '{test_audio_file}' not found. Skipping direct transcription test.")








































# backend/services/transcription_service.py
import os
import json
import whisper # OpenAI's Whisper model
import logging
import time
import asyncio # For asyncio.run in __main__
from typing import Optional, Dict, Any, AsyncGenerator # Added AsyncGenerator
from contextlib import asynccontextmanager # For mocking in __main__
import subprocess # For creating dummy audio in __main__

# --- Import Database Service ---
from . import database_service # Assuming services is a package
from .database_service import VideoProcessingStatus, get_db_session

# --- Import Text Embedding function (will be async) ---
from .embedding_service import run_text_embedding_pipeline

logger = logging.getLogger(__name__) # Will be 'services.transcription_service'

# --- Configuration ---
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
TRANSCRIPTS_SUBDIR = "transcripts"
TRANSCRIPT_FILENAME = "transcript_whisper.json"

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise


# --- Make this function ASYNC ---
async def run_transcription_pipeline(
    video_id: str,
    audio_file_path: str,
    processing_output_base_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL
) -> Optional[str]: # Return transcript_output_path on success of this chain, else None
    log_extra = {'video_id': video_id}
    logger.info(f"video_id: {video_id} - Starting transcription pipeline for audio: {audio_file_path} using model: {model_name}", extra=log_extra)
    
    # Construct paths
    transcripts_dir = os.path.join(processing_output_base_path, TRANSCRIPTS_SUBDIR)
    transcript_output_path = os.path.join(transcripts_dir, TRANSCRIPT_FILENAME)
    
    transcription_succeeded_and_saved = False
    
    # --- Transcription Step (Whisper processing and initial DB updates) ---
    async with get_db_session() as session:
        try:
            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.TRANSCRIBING
            )

            if not os.path.exists(audio_file_path):
                error_msg = f"Audio file not found at {audio_file_path}"
                logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return None

            ensure_dir(transcripts_dir) # ensure_dir might raise OSError

            logger.info(f"video_id: {video_id} - Loading Whisper model: {model_name}...", extra=log_extra)
            start_time = time.time()
            # whisper.load_model is synchronous. For high concurrency, consider asyncio.to_thread
            model = whisper.load_model(model_name)
            load_time = time.time() - start_time
            logger.info(f"video_id: {video_id} - Whisper model '{model_name}' loaded in {load_time:.2f} seconds.", extra=log_extra)

            logger.info(f"video_id: {video_id} - Starting transcription for: {audio_file_path}", extra=log_extra)
            start_time = time.time()
            # model.transcribe is synchronous
            whisper_result_data = model.transcribe(audio_file_path, fp16=False, verbose=False)
            transcribe_time = time.time() - start_time
            logger.info(f"video_id: {video_id} - Transcription completed in {transcribe_time:.2f} seconds.", extra=log_extra)

            with open(transcript_output_path, "w", encoding="utf-8") as f:
                json.dump(whisper_result_data, f, indent=2, ensure_ascii=False)
            logger.info(f"video_id: {video_id} - Transcript saved to: {transcript_output_path}", extra=log_extra)
            
            await database_service.update_video_asset_paths_record(session, video_id, transcript_path=transcript_output_path)
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.TRANSCRIBED)
            transcription_succeeded_and_saved = True
            logger.info(f"video_id: {video_id} - Transcription step successful and DB updated to TRANSCRIBED.", extra=log_extra)

        except OSError as e_os: # Specific error for ensure_dir
            logger.exception(f"video_id: {video_id} - OSError during directory creation for transcription.", extra=log_extra)
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Transcription directory error: {str(e_os)}")
            return None
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            logger.exception(f"video_id: {video_id} - Error during Whisper model loading or transcription.", extra=log_extra)
            try:
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            except Exception as db_err_on_fail: # If session is bad, this might fail
                 logger.error(f"video_id: {video_id} - Additionally, failed to update DB status after transcription error: {db_err_on_fail}", extra=log_extra)
            return None # Stop if transcription itself fails critically

    # --- Text Embedding Step (Chained after successful transcription and its DB commit) ---
    # This step will manage its own DB session and update status from TEXT_EMBEDDING -> TEXT_EMBEDDED or fail
    if not transcription_succeeded_and_saved:
        logger.error(f"video_id: {video_id} - Transcription did not succeed, skipping text embedding.", extra=log_extra)
        return None # transcript_output_path would be invalid or non-existent

    text_embedding_chain_successful = False
    try:
        logger.info(f"video_id: {video_id} - Triggering text embedding pipeline with transcript: {transcript_output_path}", extra=log_extra)
        
        embedding_pipeline_returned_success = await run_text_embedding_pipeline(
            video_id=video_id,
            transcript_file_path=transcript_output_path,
            processing_output_base_path=processing_output_base_path
        )
        
        if embedding_pipeline_returned_success:
            # To be absolutely sure, we can re-check the DB status set by the embedding pipeline
            async with get_db_session() as check_session:
                video_record = await database_service.get_video_record_by_uuid(check_session, video_id)
                if video_record and video_record.processing_status == VideoProcessingStatus.TEXT_EMBEDDED:
                    text_embedding_chain_successful = True
                    logger.info(f"video_id: {video_id} - Text embedding pipeline confirmed successful (DB status is TEXT_EMBEDDED).", extra=log_extra)
                else:
                    logger.error(f"video_id: {video_id} - Text embedding pipeline returned success, but DB status is not TEXT_EMBEDDED (actual: {video_record.processing_status if video_record else 'Unknown'}). Assuming failure of chain.", extra=log_extra)
                    # It's possible embedding_success_flag was True but DB update failed in that func
                    # Or status was further changed by another concurrent process (less likely here)
        else: # embedding_pipeline_returned_success was False
            logger.error(f"video_id: {video_id} - Text embedding pipeline explicitly reported failure.", extra=log_extra)
            # The embedding pipeline should have set the status to FAILED or PARTIAL_FAILURE in the DB.

    except Exception as e_embed_chain:
        logger.exception(f"video_id: {video_id} - Top-level exception calling text embedding pipeline.", extra=log_extra)
        # Attempt a final status update to FAILED if the embedding function didn't
        async with get_db_session() as error_session:
            await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Text Embedding Chain EXCEPTION: {str(e_embed_chain)}")

    if transcription_succeeded_and_saved and text_embedding_chain_successful:
        logger.info(f"video_id: {video_id} - Transcription and text embedding chain FULLY completed successfully.", extra=log_extra)
        return transcript_output_path
    else:
        logger.error(f"video_id: {video_id} - Transcription and text embedding chain DID NOT fully complete successfully.", extra=log_extra)
        return None


# --- if __name__ == "__main__": block for direct testing ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=log_format) # Use DEBUG for direct tests

    logger.info("Running transcription_service.py directly for testing.")
    
    test_video_id_main = "direct_transcription_main_test_001" # Changed for clarity
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.join(current_dir, "..")
    test_base_path_main = os.path.join(project_root_dir, "direct_test_output", test_video_id_main, "transcription_module")
    test_audio_dir_main = os.path.join(test_base_path_main, "audio")
    ensure_dir(test_audio_dir_main) # ensure_dir is defined in this file
    test_audio_file_main = os.path.join(test_audio_dir_main, "sample_audio_for_transcription.wav")

    # --- Mock database_service and embedding_service for direct testing ---
    original_db_service_ref = database_service
    original_run_text_embedding_ref = run_text_embedding_pipeline

    class MockDBSessionMain:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def commit(self): logger.debug("MOCK DB (transcription_test): Commit called")
        async def rollback(self): logger.debug("MOCK DB (transcription_test): Rollback called")
        async def close(self): logger.debug("MOCK DB (transcription_test): Session closed")

    class MockDatabaseServiceModuleMain:
        VideoProcessingStatus = original_db_service_ref.VideoProcessingStatus
        @asynccontextmanager
        async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]:
            logger.debug("MOCK DB (transcription_test): get_db_session called")
            yield MockDBSessionMain()
        async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
            logger.info(f"MOCK DB (transcription_test): video_id={video_id}, status={status.value}, error='{error_msg}'")
            return True
        async def update_video_asset_paths_record(self, session, video_id, transcript_path=None, **kwargs):
            logger.info(f"MOCK DB (transcription_test): video_id={video_id}, transcript_path={transcript_path}")
            return True
        async def get_video_record_by_uuid(self, session, video_id): # For checking status after embedding
            logger.info(f"MOCK DB (transcription_test): get_video_record_by_uuid for {video_id}")
            class MockVideoRecord: pass
            record = MockVideoRecord()
            # Simulate that text embedding was successful by setting the status it would set
            record.processing_status = VideoProcessingStatus.TEXT_EMBEDDED 
            return record

    database_service = MockDatabaseServiceModuleMain() # Monkey patch

    async def mock_run_text_embedding_success(video_id, transcript_file_path, processing_output_base_path):
        logger.info(f"MOCK run_text_embedding_pipeline called for video_id: {video_id} (simulating success)")
        # Simulate that this function internally updated its status to TEXT_EMBEDDED
        async with database_service.get_db_session() as s: # Uses the mocked get_db_session
            await database_service.update_video_status_and_error(s, video_id, VideoProcessingStatus.TEXT_EMBEDDED)
        return True # Indicate success

    async def mock_run_text_embedding_failure(video_id, transcript_file_path, processing_output_base_path):
        logger.info(f"MOCK run_text_embedding_pipeline called for video_id: {video_id} (simulating failure)")
        async with database_service.get_db_session() as s:
            await database_service.update_video_status_and_error(s, video_id, VideoProcessingStatus.PROCESSING_FAILED, "Mock text embedding failure")
        return False # Indicate failure

    run_text_embedding_pipeline = mock_run_text_embedding_success # Default to success for testing

    # Create a dummy audio file if it doesn't exist
    if not os.path.exists(test_audio_file_main):
        logger.info(f"Creating dummy audio file for testing: {test_audio_file_main}")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                "-t", "1", "-acodec", "pcm_s16le", test_audio_file_main, "-loglevel", "error"
            ], check=True)
            logger.info(f"Dummy audio file created: {test_audio_file_main}")
        except Exception as e_ffmpeg:
            logger.error(f"Could not create dummy audio. Please provide '{test_audio_file_main}'. Error: {e_ffmpeg}")
            exit() # Exit if dummy audio can't be created for test

    async def run_all_tests():
        if os.path.exists(test_audio_file_main):
            logger.info(f"\n--- Test 1: Successful Transcription & Text Embedding ---")
            # Ensure run_text_embedding_pipeline is the success mock
            global run_text_embedding_pipeline # To reassign in this scope
            run_text_embedding_pipeline = mock_run_text_embedding_success
            database_service.update_video_status_and_error = MockDatabaseServiceModuleMain().update_video_status_and_error # Reset mock calls

            result_path = await run_transcription_pipeline(
                video_id=test_video_id_main + "_success",
                audio_file_path=test_audio_file_main,
                processing_output_base_path=test_base_path_main,
                model_name="tiny.en"
            )
            if result_path:
                logger.info(f"Test 1 SUCCESSFUL. Transcript at: {result_path}")
                assert os.path.exists(result_path)
            else:
                logger.error("Test 1 FAILED.")

            logger.info(f"\n--- Test 2: Text Embedding Failure ---")
            run_text_embedding_pipeline = mock_run_text_embedding_failure # Switch to failure mock
            database_service.update_video_status_and_error = MockDatabaseServiceModuleMain().update_video_status_and_error # Reset mock calls

            result_path_fail = await run_transcription_pipeline(
                video_id=test_video_id_main + "_embed_fail",
                audio_file_path=test_audio_file_main,
                processing_output_base_path=test_base_path_main,
                model_name="tiny.en"
            )
            if result_path_fail is None:
                logger.info(f"Test 2 SUCCESSFUL (pipeline correctly reported failure due to embedding).")
            else:
                logger.error(f"Test 2 FAILED (pipeline should have reported failure). Result: {result_path_fail}")
        else:
            logger.warning(f"Test audio file '{test_audio_file_main}' not found. Skipping direct transcription tests.")

    if not os.getenv("DATABASE_URL"):
        logger.warning("DATABASE_URL not set for direct test. Using MOCK database_service for tests.")
    
    asyncio.run(run_all_tests())
    
    # Restore original services if they were monkey-patched at module level for some reason
    # (though the test mocks are now scoped within run_all_tests better)
    database_service = original_db_service_ref
    run_text_embedding_pipeline = original_run_text_embedding_ref
    logger.info("Direct test for transcription_service.py finished.")