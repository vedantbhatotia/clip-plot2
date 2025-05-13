
# import os
# import json
# import logging
# import uuid
# from typing import List, Dict, Any, Optional

# from moviepy import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
#     AudioFileClip, # For MusicGen output
# )
# MUSIC_ENABLED = False # Explicitly disable music generation

# # --- Import Database Service ---
# from . import database_service
# from .database_service import VideoProcessingStatus, get_db_session # For updating status/paths

# logger = logging.getLogger(__name__)
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"
# # GENERATED_MUSIC_SUBDIR = "generated_music" # No longer needed

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

# # --- get_musicgen_model function is no longer needed ---
# # def get_musicgen_model():
# #     global musicgen_model
# #     # ... (implementation removed) ...
# #     return musicgen_model

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]:
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation (MUSIC DISABLED) with {len(segments_to_include)} segments.", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided for highlight generation.", extra=log_extra)
#         return None

#     original_video_file_path = None
#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             logger.error(f"video_id: {video_id} - Original video record or path not found in DB.", extra=log_extra)
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             logger.error(f"video_id: {video_id} - Original video file not found at path: {video_record.original_video_file_path}", extra=log_extra)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     ensure_dir(highlights_output_dir)
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:8]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)

#     sub_clips = []
#     total_clip_duration = 0
#     default_fontsize = 24
#     default_text_color = 'white'
    
#     # To store the final video object for closing in finally block
#     final_video_obj_to_close = None 

#     try:
#         with VideoFileClip(original_video_file_path) as main_video_clip:
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration}s", extra=log_extra)
#             for i, segment_info in enumerate(segments_to_include):
#                 start = segment_info.get("start_time")
#                 end = segment_info.get("end_time")
#                 text_content = segment_info.get("text_content")

#                 if start is None or end is None or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid segment times for segment {i}: start={start}, end={end}. Skipping.", extra=log_extra)
#                     continue
#                 if end > main_video_clip.duration:
#                     logger.warning(f"video_id: {video_id} - Segment {i} end time {end}s exceeds video duration {main_video_clip.duration}s. Capping.", extra=log_extra)
#                     end = main_video_clip.duration
#                 if start >= main_video_clip.duration: # Also check if start is beyond or at duration
#                     logger.warning(f"video_id: {video_id} - Segment {i} start time {start}s is at or beyond video duration {main_video_clip.duration}s. Skipping.", extra=log_extra)
#                     continue
#                 if start == end: # Skip zero-duration clips
#                     logger.warning(f"video_id: {video_id} - Segment {i} has zero duration (start == end). Skipping.", extra=log_extra)
#                     continue


#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start}s - {end}s", extra=log_extra)
#                 # Corrected method name
#                 sub_clip = main_video_clip.subclipped(start, end) 
#                 if sub_clip.audio is None:
#                     logger.error(f"video_id: {video_id} - Subclip {i+1} for segment {start}-{end} has NO audio! This will likely cause issues. Main clip audio was: {main_video_clip.audio is not None}", extra=log_extra)
#                     # Decide how to handle: skip this subclip? proceed without its audio? For now, it will just be added.
#                 else:
#                     logger.info(f"video_id: {video_id} - Subclip {i+1} for segment {start}-{end} has audio. Duration: {sub_clip.audio.duration}", extra=log_extra)

                
#                 if text_content and text_content.strip():
#                     try:
#                         txt_clip = TextClip(
#                             text_content,
#                             fontsize=default_fontsize,
#                             color=default_text_color,
#                             font="Arial", 
#                             method='caption',
#                             size=(sub_clip.w * 0.9, None),
#                             bg_color='transparent',
#                             stroke_color='black',
#                             stroke_width=1
#                         )
#                         txt_clip = txt_clip.set_pos(('center', 'bottom-10%')).set_duration(sub_clip.duration)
#                         sub_clip = CompositeVideoClip([sub_clip, txt_clip], use_bgclip=True if sub_clip.mask is None else False) # Added use_bgclip for safety with transparent TextClip
#                         logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
#                     except Exception as e_textclip:
#                         logger.error(f"video_id: {video_id} - Failed to create TextClip for segment {i+1}: {e_textclip}. Proceeding without subtitle for this segment.", extra=log_extra)

#                 sub_clips.append(sub_clip)
#                 total_clip_duration += sub_clip.duration
        
#         if not sub_clips:
#             logger.error(f"video_id: {video_id} - No valid sub-clips were generated.", extra=log_extra)
#             return None

#         final_video_obj_to_close = concatenate_videoclips(sub_clips, method="compose")
#         logger.info(f"video_id: {video_id} - Concatenated {len(sub_clips)} sub-clips. Total duration: {total_clip_duration:.2f}s", extra=log_extra)

#         # --- Music Generation Block is now effectively skipped due to MUSIC_ENABLED = False ---
#         # if MUSIC_ENABLED and total_clip_duration > 0:
#         #    ... (music logic would go here) ...
#         # The original audio from the concatenated clips will be preserved unless explicitly replaced.

#         logger.info(f"video_id: {video_id} - Writing final highlight clip (without new music) to: {final_clip_path}", extra=log_extra)
#         final_video_obj_to_close.write_videofile(
#             final_clip_path,
#             codec="libx264",
#             audio_codec="aac",
#             temp_audiofile=f'temp-audio-{str(uuid.uuid4())[:8]}.m4a', # Unique temp audio file
#             remove_temp=True,
#             threads=os.cpu_count() or 4, # Use available CPUs or default to 4
#             fps=24 
#         )
#         logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)

#         async with get_db_session() as session:
#             await database_service.update_video_asset_paths_record(session, video_id, highlight_clip_path=final_clip_path)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
#             logger.info(f"video_id: {video_id} - DB updated with highlight path and status.", extra=log_extra)

#         return final_clip_path

#     except Exception as e:
#         logger.exception(f"video_id: {video_id} - Error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as session: # Try to update status on failure
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Highlight generation error: {str(e)}")
#         return None
#     finally:
#         # Close all sub_clips and the final_video if they exist
#         for clip in sub_clips:
#             try:
#                 clip.close()
#             except Exception as e_close:
#                 logger.debug(f"video_id: {video_id} - Minor error closing a sub_clip: {e_close}", extra=log_extra)
#         if final_video_obj_to_close: # Renamed variable for clarity
#             try:
#                 final_video_obj_to_close.close()
#             except Exception as e_close_final:
#                 logger.debug(f"video_id: {video_id} - Minor error closing final_video: {e_close_final}", extra=log_extra)


# if __name__ == "__main__":
#     # ... (your existing test block, it should now run without attempting music generation) ...
#     # Ensure the mock for database_service.update_video_asset_paths_record also handles highlight_clip_path
#     # and that the mock for database_service.update_video_status_and_error handles HIGHLIGHT_GENERATED status.
#     if not logging.getLogger().hasHandlers():
#         log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
#         logging.basicConfig(level=logging.INFO, format=log_format)

#     async def test_clip_generation():
#         logger.info("Running clip_builder_service.py directly for testing (NO MUSIC)...")
        
#         class MockVideoRecord:
#             def __init__(self, uuid, path):
#                 self.video_uuid = uuid
#                 self.original_video_file_path = path
        
#         async def mock_get_video_record(session, uuid):
#             test_video_file = "sample_short_video.mp4" 
#             if not os.path.exists(test_video_file):
#                 logger.error(f"Test video '{test_video_file}' not found. Please create it or update path.")
#                 try:
#                     subprocess.run([
#                         "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=24", # Smaller size for faster test
#                         "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
#                         "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", test_video_file
#                     ], check=True, capture_output=True, text=True)
#                     logger.info(f"Created dummy test video: {test_video_file}")
#                 except Exception as e_ffmpeg_dummy:
#                     logger.error(f"Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}")
#                     return None
#             return MockVideoRecord(uuid, os.path.abspath(test_video_file))

#         class MockSessionObj: # Simple object to act as session for mock
#             pass

#         @asynccontextmanager
#         async def mock_get_db_session_context(): 
#             yield MockSessionObj() 

#         original_get_video_record = database_service.get_video_record_by_uuid
#         original_get_session = database_service.get_db_session
#         original_update_asset_paths = database_service.update_video_asset_paths_record
#         original_update_status = database_service.update_video_status_and_error

#         async def mock_update_asset_paths(session, video_uuid, highlight_clip_path=None, **kwargs):
#             logger.info(f"MOCK DB: video_id={video_uuid}, highlight_clip_path={highlight_clip_path}")
#             return True
#         async def mock_update_status(session, video_uuid, status, error_msg=None):
#             logger.info(f"MOCK DB: video_id={video_uuid}, status={status.value}, error='{error_msg}'")
#             return True

#         database_service.get_video_record_by_uuid = mock_get_video_record
#         database_service.get_db_session = mock_get_db_session_context
#         database_service.update_video_asset_paths_record = mock_update_asset_paths
#         database_service.update_video_status_and_error = mock_update_status

#         test_vid_id = "clip_builder_test_nomusic_001"
#         test_processing_path = f"/tmp/clippilot_test_clipbuilder/{test_vid_id}" 
#         ensure_dir(test_processing_path)

#         segments = [
#             {"start_time": 1.0, "end_time": 3.5, "text_content": "First amazing segment!"},
#             {"start_time": 5.2, "end_time": 8.0, "text_content": "Another key moment here."},
#         ]

#         logger.info(f"Attempting to generate highlight for video_id: {test_vid_id}")
#         highlight_path = await generate_highlight_clip(
#             video_id=test_vid_id,
#             segments_to_include=segments,
#             processing_base_path=test_processing_path
#         )

#         if highlight_path:
#             logger.info(f"Highlight clip generated successfully: {highlight_path}")
#         else:
#             logger.error("Highlight clip generation failed.")

#         database_service.get_video_record_by_uuid = original_get_video_record
#         database_service.get_db_session = original_get_session
#         database_service.update_video_asset_paths_record = original_update_asset_paths
#         database_service.update_video_status_and_error = original_update_status
        
#         # import shutil
#         # if os.path.exists(f"/tmp/clippilot_test_clipbuilder"):
#         #     shutil.rmtree(f"/tmp/clippilot_test_clipbuilder")
#         # if os.path.exists("sample_short_video.mp4"):
#         #    os.remove("sample_short_video.mp4")


#     import asyncio
#     import subprocess # Make sure subprocess is imported for the test
#     asyncio.run(test_clip_generation())




























# backend/services/clip_builder_service.py
import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager # For mocking in __main__

from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    # AudioFileClip, # Not explicitly used now that MusicGen is disabled
)
# from moviepy.config import change_settings # If you need to specify ImageMagick

MUSIC_ENABLED = False # Explicitly disable music generation

# --- Import Database Service ---
from . import database_service
from .database_service import VideoProcessingStatus, get_db_session 

logger = logging.getLogger(__name__)
HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

def ensure_dir(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise

async def generate_highlight_clip(
    video_id: str,
    segments_to_include: List[Dict[str, Any]],
    processing_base_path: str,
    output_filename: Optional[str] = None
) -> Optional[str]:
    log_extra = {'video_id': video_id}
    logger.info(f"video_id: {video_id} - Starting highlight clip generation (MUSIC DISABLED) with {len(segments_to_include)} segments.", extra=log_extra)

    if not segments_to_include:
        logger.warning(f"video_id: {video_id} - No segments provided for highlight generation.", extra=log_extra)
        return None

    original_video_file_path = None
    main_video_fps_for_output = 24 # Default FPS for output video
    source_audio_fps = 44100       # Default audio FPS if not detected from source

    async with get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, video_id)
        if not video_record or not video_record.original_video_file_path:
            logger.error(f"video_id: {video_id} - Original video record or path not found in DB.", extra=log_extra)
            return None
        if not os.path.exists(video_record.original_video_file_path):
            logger.error(f"video_id: {video_id} - Original video file not found at path: {video_record.original_video_file_path}", extra=log_extra)
            return None
        original_video_file_path = video_record.original_video_file_path
    
    logger.info(f"video_id: {video_id} - Original video path: {original_video_file_path}", extra=log_extra)

    highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
    ensure_dir(highlights_output_dir)
    
    if not output_filename:
        output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:8]}.mp4"
    final_clip_path = os.path.join(highlights_output_dir, output_filename)

    final_clip_path_to_return = None # Store the path if successful
    
    try:
        # Open the main video clip ONCE
        with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
            main_video_fps_for_output = main_video_clip.fps or 24 
            logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration}s, Video FPS: {main_video_fps_for_output}", extra=log_extra)
            
            if main_video_clip.audio is None:
                logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)
                source_audio_fps = None 
            else:
                logger.info(f"video_id: {video_id} - Original video audio: Duration={main_video_clip.audio.duration if main_video_clip.audio else 'N/A'}, FPS={main_video_clip.audio.fps if main_video_clip.audio else 'N/A'}", extra=log_extra)
                if main_video_clip.audio.fps:
                    source_audio_fps = main_video_clip.audio.fps

            sub_clips_for_concat = [] # Corrected variable name
            default_fontsize = 24
            default_text_color = 'white'

            for i, segment_info in enumerate(segments_to_include):
                start = segment_info.get("start_time")
                end = segment_info.get("end_time")
                text_content = segment_info.get("text_content")

                # --- Segment Validation ---
                if start is None or end is None or start >= end:
                    logger.warning(f"video_id: {video_id} - Invalid segment times for segment {i+1}: start={start}, end={end}. Skipping.", extra=log_extra)
                    continue
                current_main_duration = main_video_clip.duration
                if end > current_main_duration:
                    logger.warning(f"video_id: {video_id} - Segment {i+1} end time {end}s exceeds video duration {current_main_duration}s. Capping to {current_main_duration}.", extra=log_extra)
                    end = current_main_duration
                if start >= current_main_duration: 
                    logger.warning(f"video_id: {video_id} - Segment {i+1} start time {start}s is at or beyond video duration {current_main_duration}s. Skipping.", extra=log_extra)
                    continue
                if start == end: 
                    logger.warning(f"video_id: {video_id} - Segment {i+1} has zero duration after capping (start == end). Skipping.", extra=log_extra)
                    continue
                # --- End Segment Validation ---

                logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start}s - {end}s", extra=log_extra)
                sub_clip = main_video_clip.subclip(start, end) 
                 
                if sub_clip.audio is None:
                    logger.warning(f"video_id: {video_id} - Subclip {i+1} ({start}-{end}) has NO audio! Main clip audio was: {main_video_clip.audio is not None}", extra=log_extra)
                else:
                    logger.info(f"video_id: {video_id} - Subclip {i+1} ({start}-{end}) has audio. Duration: {sub_clip.audio.duration}, FPS: {sub_clip.audio.fps}", extra=log_extra)
                    if source_audio_fps is None and sub_clip.audio.fps:
                        source_audio_fps = sub_clip.audio.fps
                        logger.info(f"video_id: {video_id} - Setting source_audio_fps from subclip: {source_audio_fps}", extra=log_extra)
                
                if text_content and text_content.strip():
                    try:
                        txt_clip = TextClip(
                            text_content, fontsize=default_fontsize, color=default_text_color, font="Arial", 
                            method='caption', size=(sub_clip.w * 0.9, None), bg_color='transparent',
                            stroke_color='black', stroke_width=1
                        )
                        txt_clip = txt_clip.set_pos(('center', 'bottom-10%')).set_duration(sub_clip.duration)
                        sub_clip = CompositeVideoClip([sub_clip, txt_clip], use_bgclip=True if sub_clip.mask is None else False)
                        logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
                    except Exception as e_textclip:
                        logger.error(f"video_id: {video_id} - Failed to create TextClip: {e_textclip}. Segment {i+1} will not have this subtitle.", extra=log_extra)
                
                sub_clips_for_concat.append(sub_clip)
            # End of segment processing loop

            if not sub_clips_for_concat:
                logger.error(f"video_id: {video_id} - No valid sub-clips were generated to concatenate.", extra=log_extra)
                # No specific return here, will be handled by final_clip_path_to_return check
            else:
                logger.info(f"video_id: {video_id} - Attempting to concatenate {len(sub_clips_for_concat)} sub-clips.", extra=log_extra)
                
                concatenated_clip = concatenate_videoclips(sub_clips_for_concat, method="compose")
                logger.info(f"video_id: {video_id} - Concatenated. Final duration: {concatenated_clip.duration:.2f}s", extra=log_extra)

                if concatenated_clip and concatenated_clip.audio:
                    if not hasattr(concatenated_clip.audio, 'fps') or concatenated_clip.audio.fps is None:
                        if source_audio_fps:
                            logger.warning(f"video_id: {video_id} - CompositeAudioClip missing/None fps, setting to {source_audio_fps}", extra=log_extra)
                            concatenated_clip.audio.fps = source_audio_fps
                        else:
                            logger.warning(f"video_id: {video_id} - CompositeAudioClip missing fps and no source FPS. Defaulting to 44100.", extra=log_extra)
                            concatenated_clip.audio.fps = 44100
                elif concatenated_clip and concatenated_clip.audio is None:
                    logger.warning(f"video_id: {video_id} - Concatenated clip has NO audio component.", extra=log_extra)


                # ---- OPTIONAL: DETAILED INSPECTION OF CONCATENATED AUDIO ----
                logger.info(f"video_id: {video_id} - DEBUG: Inspecting concatenated_clip.audio after potential FPS set", extra=log_extra)
                if concatenated_clip and concatenated_clip.audio:
                    audio_obj_debug = concatenated_clip.audio # Use a different variable for debug scope
                    logger.info(f"video_id: {video_id} - DEBUG: concatenated_clip.audio type: {type(audio_obj_debug)}", extra=log_extra)
                    logger.info(f"video_id: {video_id} - DEBUG: audio_obj_debug.duration: {getattr(audio_obj_debug, 'duration', 'N/A')}", extra=log_extra)
                    logger.info(f"video_id: {video_id} - DEBUG: audio_obj_debug.fps: {getattr(audio_obj_debug, 'fps', 'N/A')}", extra=log_extra)
                    # ... (other inspection logs if needed) ...
                    try:
                        temp_debug_audio_path = os.path.join(highlights_output_dir, f"debug_concat_audio_{video_id}.aac")
                        debug_audio_fps = getattr(audio_obj_debug, 'fps', None) or source_audio_fps or 44100
                        logger.info(f"video_id: {video_id} - DEBUG: Attempting to write ONLY audio to {temp_debug_audio_path} with FPS {debug_audio_fps}", extra=log_extra)
                        audio_obj_debug.write_audiofile(temp_debug_audio_path, codec='aac', fps=debug_audio_fps, logger='bar')
                        logger.info(f"video_id: {video_id} - DEBUG: SUCCESSFULLY wrote concatenated audio to {temp_debug_audio_path}", extra=log_extra)
                    except Exception as e_audio_debug_write:
                        logger.error(f"video_id: {video_id} - DEBUG: FAILED to write concatenated audio directly: {e_audio_debug_write}", extra=log_extra)
                        logger.exception("Full traceback for direct audio write failure:")
                # ---- END OPTIONAL DEBUG INSPECTION ----


                has_audio_track = concatenated_clip and concatenated_clip.audio is not None
                logger.info(f"video_id: {video_id} - Writing final highlight clip to: {final_clip_path}. Has audio: {has_audio_track}", extra=log_extra)

                final_audio_fps_to_use = None
                if has_audio_track:
                    final_audio_fps_to_use = getattr(concatenated_clip.audio, 'fps', None) or source_audio_fps
                    if not final_audio_fps_to_use and source_audio_fps: 
                        final_audio_fps_to_use = source_audio_fps
                    elif not final_audio_fps_to_use: 
                         logger.warning(f"video_id: {video_id} - No definite audio FPS for final write, defaulting to 44100.", extra=log_extra)
                         final_audio_fps_to_use = 44100

                concatenated_clip.write_videofile(
                    final_clip_path,
                    codec="libx264",
                    audio_codec="aac" if has_audio_track else None,
                    audio=has_audio_track,
                    temp_audiofile=f'temp-audio-{str(uuid.uuid4())[:8]}.m4a',
                    remove_temp=True,
                    threads=os.cpu_count() or 4,
                    fps=main_video_fps_for_output,
                    audio_fps=final_audio_fps_to_use if has_audio_track else None,
                    logger='bar'
                )
                logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
                final_clip_path_to_return = final_clip_path 

                try: # Attempt to close the concatenated clip
                    concatenated_clip.close()
                except Exception as e_close_concat:
                     logger.debug(f"video_id: {video_id} - Minor error closing concatenated_clip: {e_close_concat}", extra=log_extra)
        # main_video_clip is now closed by the 'with' statement.

        if final_clip_path_to_return:
            async with get_db_session() as session_after_write:
                await database_service.update_video_asset_paths_record(session_after_write, video_id, highlight_clip_path=final_clip_path_to_return)
                await database_service.update_video_status_and_error(session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
                logger.info(f"video_id: {video_id} - DB updated with highlight path and status.", extra=log_extra)
            return final_clip_path_to_return
        else:
            logger.error(f"video_id: {video_id} - Highlight generation did not produce an output path (likely no valid segments or write failed before exception).", extra=log_extra)
            async with get_db_session() as error_session_no_output: 
                current_status_rec = await database_service.get_video_record_by_uuid(error_session_no_output, video_id)
                if current_status_rec and current_status_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
                    await database_service.update_video_status_and_error(error_session_no_output, video_id, VideoProcessingStatus.PROCESSING_FAILED, "Highlight generation failed (no output path).")
            return None

    except Exception as e:
        logger.exception(f"video_id: {video_id} - Error during highlight clip generation process.", extra=log_extra)
        async with get_db_session() as error_session:
            await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Highlight generation error: {str(e)}")
        return None
    finally:
        # Subclips from main_video_clip are views; their resources are tied to main_video_clip,
        # which is closed by its 'with' statement.
        # The concatenated_clip (if assigned to final_video_obj_for_closing or similar) might need closing
        # if write_videofile failed and it wasn't closed, but its primary resources are also from subclips.
        # The main explicit close needed was for main_video_clip (handled by 'with') and the result of concatenate_videoclips
        # if it's not used up by write_videofile or if write_videofile fails before it can manage resources.
        # The current structure with concatenated_clip.close() inside the 'with' block for main_video_clip is good.
        logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    async def test_clip_generation():
        logger.info("Running clip_builder_service.py directly for testing (NO MUSIC)...")
        
        class MockVideoRecord:
            def __init__(self, uuid, path, fps=24, audio_fps=44100):
                self.video_uuid = uuid
                self.original_video_file_path = path
                self.fps = fps # Mocking this for main_video_clip.fps
                self.duration = 10.0 
                self.audio = None
                if audio_fps:
                    class MockAudio:
                        def __init__(self, parent_duration, parent_fps):
                            self.duration = parent_duration
                            self.fps = parent_fps
                            self.nchannels = 2 
                        def subclip(self, start, end):
                            new_audio = MockAudio(end-start, self.fps)
                            return new_audio
                    self.audio = MockAudio(self.duration, audio_fps)

        async def mock_get_video_record(session, uuid):
            test_video_file = "sample_short_video.mp4" 
            if not os.path.exists(test_video_file):
                logger.error(f"Test video '{test_video_file}' not found. Please create it or update path.")
                try:
                    import subprocess 
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=24",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=10:sample_rate=44100", 
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", test_video_file
                    ], check=True, capture_output=True, text=True)
                    logger.info(f"Created dummy test video: {test_video_file}")
                except Exception as e_ffmpeg_dummy:
                    logger.error(f"Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}")
                    return None
            return MockVideoRecord(uuid, os.path.abspath(test_video_file)) # This mock is not fully used as VideoFileClip is called directly

        class MockSessionObj:
            pass

        @asynccontextmanager
        async def mock_get_db_session_context(): 
            yield MockSessionObj() 

        original_get_video_record = database_service.get_video_record_by_uuid
        original_get_session = database_service.get_db_session
        original_update_asset_paths = database_service.update_video_asset_paths_record
        original_update_status = database_service.update_video_status_and_error

        async def mock_update_asset_paths(session, video_uuid, highlight_clip_path=None, **kwargs):
            logger.info(f"MOCK DB: video_id={video_uuid}, highlight_clip_path={highlight_clip_path}")
            return True
        async def mock_update_status(session, video_uuid, status, error_msg=None):
            logger.info(f"MOCK DB: video_id={video_uuid}, status={status.value}, error='{error_msg}'")
            return True

        database_service.get_video_record_by_uuid = mock_get_video_record
        database_service.get_db_session = mock_get_db_session_context
        database_service.update_video_asset_paths_record = mock_update_asset_paths
        database_service.update_video_status_and_error = mock_update_status

        test_vid_id = "clip_builder_test_nomusic_001"
        # Ensure this path is writable and makes sense for your OS
        # On Windows, /tmp/ might go to C:\tmp or similar.
        test_processing_base_path_str = os.path.join(os.path.expanduser("~"), "clippilot_tests_output", test_vid_id)
        ensure_dir(test_processing_base_path_str)
        logger.info(f"Test processing base path: {test_processing_base_path_str}")


        segments = [
            {"start_time": 1.0, "end_time": 3.5, "text_content": "First amazing segment!"},
            {"start_time": 5.2, "end_time": 8.0, "text_content": "Another key moment here."},
        ]

        logger.info(f"Attempting to generate highlight for video_id: {test_vid_id}")
        highlight_path = await generate_highlight_clip(
            video_id=test_vid_id,
            segments_to_include=segments,
            processing_base_path=test_processing_base_path_str # Use the ensured path
        )

        if highlight_path:
            logger.info(f"Highlight clip generated successfully: {highlight_path}")
        else:
            logger.error("Highlight clip generation failed.")

        database_service.get_video_record_by_uuid = original_get_video_record
        database_service.get_db_session = original_get_session
        database_service.update_video_asset_paths_record = original_update_asset_paths
        database_service.update_video_status_and_error = original_update_status
        
    import asyncio
    asyncio.run(test_clip_generation())