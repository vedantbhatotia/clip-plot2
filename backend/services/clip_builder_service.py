
# import os
# import json
# import logging
# import uuid
# from typing import List, Dict, Any, Optional
# from contextlib import asynccontextmanager # For mocking in __main__

# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
#     # AudioFileClip, # Not explicitly used now that MusicGen is disabled
# )
# # from moviepy.config import change_settings # If you need to specify ImageMagick

# MUSIC_ENABLED = False # Explicitly disable music generation

# # --- Import Database Service ---
# from . import database_service
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__)
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

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
#     main_video_fps_for_output = 24 # Default FPS for output video
#     source_audio_fps = 44100       # Default audio FPS if not detected from source

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

#     final_clip_path_to_return = None # Store the path if successful
    
#     try:
#         # Open the main video clip ONCE
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24 
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration}s, Video FPS: {main_video_fps_for_output}", extra=log_extra)
            
#             if main_video_clip.audio is None:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)
#                 source_audio_fps = None 
#             else:
#                 logger.info(f"video_id: {video_id} - Original video audio: Duration={main_video_clip.audio.duration if main_video_clip.audio else 'N/A'}, FPS={main_video_clip.audio.fps if main_video_clip.audio else 'N/A'}", extra=log_extra)
#                 if main_video_clip.audio.fps:
#                     source_audio_fps = main_video_clip.audio.fps

#             sub_clips_for_concat = [] # Corrected variable name
#             default_fontsize = 24
#             default_text_color = 'white'

#             for i, segment_info in enumerate(segments_to_include):
#                 start = segment_info.get("start_time")
#                 end = segment_info.get("end_time")
#                 text_content = segment_info.get("text_content")

#                 # --- Segment Validation ---
#                 if start is None or end is None or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid segment times for segment {i+1}: start={start}, end={end}. Skipping.", extra=log_extra)
#                     continue
#                 current_main_duration = main_video_clip.duration
#                 if end > current_main_duration:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} end time {end}s exceeds video duration {current_main_duration}s. Capping to {current_main_duration}.", extra=log_extra)
#                     end = current_main_duration
#                 if start >= current_main_duration: 
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} start time {start}s is at or beyond video duration {current_main_duration}s. Skipping.", extra=log_extra)
#                     continue
#                 if start == end: 
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} has zero duration after capping (start == end). Skipping.", extra=log_extra)
#                     continue
#                 # --- End Segment Validation ---

#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start}s - {end}s", extra=log_extra)
#                 sub_clip = main_video_clip.subclip(start, end) 
                 
#                 if sub_clip.audio is None:
#                     logger.warning(f"video_id: {video_id} - Subclip {i+1} ({start}-{end}) has NO audio! Main clip audio was: {main_video_clip.audio is not None}", extra=log_extra)
#                 else:
#                     logger.info(f"video_id: {video_id} - Subclip {i+1} ({start}-{end}) has audio. Duration: {sub_clip.audio.duration}, FPS: {sub_clip.audio.fps}", extra=log_extra)
#                     if source_audio_fps is None and sub_clip.audio.fps:
#                         source_audio_fps = sub_clip.audio.fps
#                         logger.info(f"video_id: {video_id} - Setting source_audio_fps from subclip: {source_audio_fps}", extra=log_extra)
                
#                 if text_content and text_content.strip():
#                     try:
#                         txt_clip = TextClip(
#                             text_content, fontsize=default_fontsize, color=default_text_color, font="Arial", 
#                             method='caption', size=(sub_clip.w * 0.9, None), bg_color='transparent',
#                             stroke_color='black', stroke_width=1
#                         )
#                         y_position = sub_clip.h - (txt_clip.h if txt_clip.h else default_fontsize * 1.5) - (sub_clip.h * 0.10) 
#                         txt_clip = txt_clip.set_pos(('center', y_position)).set_duration(sub_clip.duration)
#                         sub_clip = CompositeVideoClip([sub_clip, txt_clip], use_bgclip=True if sub_clip.mask is None else False)
#                         logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
#                     except Exception as e_textclip:
#                         logger.error(f"video_id: {video_id} - Failed to create TextClip: {e_textclip}. Segment {i+1} will not have this subtitle.", extra=log_extra)
                
#                 sub_clips_for_concat.append(sub_clip)
#             # End of segment processing loop

#             if not sub_clips_for_concat:
#                 logger.error(f"video_id: {video_id} - No valid sub-clips were generated to concatenate.", extra=log_extra)
#                 # No specific return here, will be handled by final_clip_path_to_return check
#             else:
#                 logger.info(f"video_id: {video_id} - Attempting to concatenate {len(sub_clips_for_concat)} sub-clips.", extra=log_extra)
                
#                 concatenated_clip = concatenate_videoclips(sub_clips_for_concat, method="compose")
#                 logger.info(f"video_id: {video_id} - Concatenated. Final duration: {concatenated_clip.duration:.2f}s", extra=log_extra)

#                 if concatenated_clip and concatenated_clip.audio:
#                     if not hasattr(concatenated_clip.audio, 'fps') or concatenated_clip.audio.fps is None:
#                         if source_audio_fps:
#                             logger.warning(f"video_id: {video_id} - CompositeAudioClip missing/None fps, setting to {source_audio_fps}", extra=log_extra)
#                             concatenated_clip.audio.fps = source_audio_fps
#                         else:
#                             logger.warning(f"video_id: {video_id} - CompositeAudioClip missing fps and no source FPS. Defaulting to 44100.", extra=log_extra)
#                             concatenated_clip.audio.fps = 44100
#                 elif concatenated_clip and concatenated_clip.audio is None:
#                     logger.warning(f"video_id: {video_id} - Concatenated clip has NO audio component.", extra=log_extra)


#                 # ---- OPTIONAL: DETAILED INSPECTION OF CONCATENATED AUDIO ----
#                 logger.info(f"video_id: {video_id} - DEBUG: Inspecting concatenated_clip.audio after potential FPS set", extra=log_extra)
#                 if concatenated_clip and concatenated_clip.audio:
#                     audio_obj_debug = concatenated_clip.audio # Use a different variable for debug scope
#                     logger.info(f"video_id: {video_id} - DEBUG: concatenated_clip.audio type: {type(audio_obj_debug)}", extra=log_extra)
#                     logger.info(f"video_id: {video_id} - DEBUG: audio_obj_debug.duration: {getattr(audio_obj_debug, 'duration', 'N/A')}", extra=log_extra)
#                     logger.info(f"video_id: {video_id} - DEBUG: audio_obj_debug.fps: {getattr(audio_obj_debug, 'fps', 'N/A')}", extra=log_extra)
#                     # ... (other inspection logs if needed) ...
#                     try:
#                         temp_debug_audio_path = os.path.join(highlights_output_dir, f"debug_concat_audio_{video_id}.aac")
#                         debug_audio_fps = getattr(audio_obj_debug, 'fps', None) or source_audio_fps or 44100
#                         logger.info(f"video_id: {video_id} - DEBUG: Attempting to write ONLY audio to {temp_debug_audio_path} with FPS {debug_audio_fps}", extra=log_extra)
#                         audio_obj_debug.write_audiofile(temp_debug_audio_path, codec='aac', fps=debug_audio_fps, logger='bar')
#                         logger.info(f"video_id: {video_id} - DEBUG: SUCCESSFULLY wrote concatenated audio to {temp_debug_audio_path}", extra=log_extra)
#                     except Exception as e_audio_debug_write:
#                         logger.error(f"video_id: {video_id} - DEBUG: FAILED to write concatenated audio directly: {e_audio_debug_write}", extra=log_extra)
#                         logger.exception("Full traceback for direct audio write failure:")
#                 # ---- END OPTIONAL DEBUG INSPECTION ----


#                 has_audio_track = concatenated_clip and concatenated_clip.audio is not None
#                 logger.info(f"video_id: {video_id} - Writing final highlight clip to: {final_clip_path}. Has audio: {has_audio_track}", extra=log_extra)

#                 final_audio_fps_to_use = None
#                 if has_audio_track:
#                     final_audio_fps_to_use = getattr(concatenated_clip.audio, 'fps', None) or source_audio_fps
#                     if not final_audio_fps_to_use and source_audio_fps: 
#                         final_audio_fps_to_use = source_audio_fps
#                     elif not final_audio_fps_to_use: 
#                          logger.warning(f"video_id: {video_id} - No definite audio FPS for final write, defaulting to 44100.", extra=log_extra)
#                          final_audio_fps_to_use = 44100

#                 concatenated_clip.write_videofile(
#                     final_clip_path,
#                     codec="libx264",
#                     audio_codec="aac" if has_audio_track else None,
#                     audio=has_audio_track,
#                     temp_audiofile=f'temp-audio-{str(uuid.uuid4())[:8]}.m4a',
#                     remove_temp=True,
#                     threads=os.cpu_count() or 4,
#                     fps=main_video_fps_for_output,
#                     audio_fps=final_audio_fps_to_use if has_audio_track else None,
#                     logger='bar'
#                 )
#                 logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#                 final_clip_path_to_return = final_clip_path 

#                 try: # Attempt to close the concatenated clip
#                     concatenated_clip.close()
#                 except Exception as e_close_concat:
#                      logger.debug(f"video_id: {video_id} - Minor error closing concatenated_clip: {e_close_concat}", extra=log_extra)
#         # main_video_clip is now closed by the 'with' statement.

#         if final_clip_path_to_return:
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(session_after_write, video_id, highlight_clip_path=final_clip_path_to_return)
#                 await database_service.update_video_status_and_error(session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
#                 logger.info(f"video_id: {video_id} - DB updated with highlight path and status.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             logger.error(f"video_id: {video_id} - Highlight generation did not produce an output path (likely no valid segments or write failed before exception).", extra=log_extra)
#             async with get_db_session() as error_session_no_output: 
#                 current_status_rec = await database_service.get_video_record_by_uuid(error_session_no_output, video_id)
#                 if current_status_rec and current_status_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                     await database_service.update_video_status_and_error(error_session_no_output, video_id, VideoProcessingStatus.PROCESSING_FAILED, "Highlight generation failed (no output path).")
#             return None

#     except Exception as e:
#         logger.exception(f"video_id: {video_id} - Error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Highlight generation error: {str(e)}")
#         return None
#     finally:
#         # Subclips from main_video_clip are views; their resources are tied to main_video_clip,
#         # which is closed by its 'with' statement.
#         # The concatenated_clip (if assigned to final_video_obj_for_closing or similar) might need closing
#         # if write_videofile failed and it wasn't closed, but its primary resources are also from subclips.
#         # The main explicit close needed was for main_video_clip (handled by 'with') and the result of concatenate_videoclips
#         # if it's not used up by write_videofile or if write_videofile fails before it can manage resources.
#         # The current structure with concatenated_clip.close() inside the 'with' block for main_video_clip is good.
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)


# if __name__ == "__main__":
#     if not logging.getLogger().hasHandlers():
#         log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
#         logging.basicConfig(level=logging.INFO, format=log_format)

#     async def test_clip_generation():
#         logger.info("Running clip_builder_service.py directly for testing (NO MUSIC)...")
        
#         class MockVideoRecord:
#             def __init__(self, uuid, path, fps=24, audio_fps=44100):
#                 self.video_uuid = uuid
#                 self.original_video_file_path = path
#                 self.fps = fps # Mocking this for main_video_clip.fps
#                 self.duration = 10.0 
#                 self.audio = None
#                 if audio_fps:
#                     class MockAudio:
#                         def __init__(self, parent_duration, parent_fps):
#                             self.duration = parent_duration
#                             self.fps = parent_fps
#                             self.nchannels = 2 
#                         def subclip(self, start, end):
#                             new_audio = MockAudio(end-start, self.fps)
#                             return new_audio
#                     self.audio = MockAudio(self.duration, audio_fps)

#         async def mock_get_video_record(session, uuid):
#             test_video_file = "sample_short_video.mp4" 
#             if not os.path.exists(test_video_file):
#                 logger.error(f"Test video '{test_video_file}' not found. Please create it or update path.")
#                 try:
#                     import subprocess 
#                     subprocess.run([
#                         "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=24",
#                         "-f", "lavfi", "-i", "sine=frequency=440:duration=10:sample_rate=44100", 
#                         "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", test_video_file
#                     ], check=True, capture_output=True, text=True)
#                     logger.info(f"Created dummy test video: {test_video_file}")
#                 except Exception as e_ffmpeg_dummy:
#                     logger.error(f"Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}")
#                     return None
#             return MockVideoRecord(uuid, os.path.abspath(test_video_file)) # This mock is not fully used as VideoFileClip is called directly

#         class MockSessionObj:
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
#         # Ensure this path is writable and makes sense for your OS
#         # On Windows, /tmp/ might go to C:\tmp or similar.
#         test_processing_base_path_str = os.path.join(os.path.expanduser("~"), "clippilot_tests_output", test_vid_id)
#         ensure_dir(test_processing_base_path_str)
#         logger.info(f"Test processing base path: {test_processing_base_path_str}")


#         segments = [
#             {"start_time": 1.0, "end_time": 3.5, "text_content": "First amazing segment!"},
#             {"start_time": 5.2, "end_time": 8.0, "text_content": "Another key moment here."},
#         ]

#         logger.info(f"Attempting to generate highlight for video_id: {test_vid_id}")
#         highlight_path = await generate_highlight_clip(
#             video_id=test_vid_id,
#             segments_to_include=segments,
#             processing_base_path=test_processing_base_path_str # Use the ensured path
#         )

#         if highlight_path:
#             logger.info(f"Highlight clip generated successfully: {highlight_path}")
#         else:
#             logger.error("Highlight clip generation failed.")

#         database_service.get_video_record_by_uuid = original_get_video_record
#         database_service.get_db_session = original_get_session
#         database_service.update_video_asset_paths_record = original_update_asset_paths
#         database_service.update_video_status_and_error = original_update_status
        
#     import asyncio
#     asyncio.run(test_clip_generation())

















































































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager

# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
# )
# # from moviepy.config import change_settings

# MUSIC_ENABLED = False

# # --- Import Database Service ---
# # This 'database_service' will be the one potentially patched in __main__ for testing
# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__) # Will be 'services.clip_builder_service'
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]:
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided for highlight generation. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     source_audio_fps: Optional[int] = None

#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             logger.error(f"video_id: {video_id} - Original video record or path not found in DB. Cannot generate highlight.", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, "Original video record/path not found for clip gen")
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir:
#         logger.exception(f"video_id: {video_id} - Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {e_dir}")
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     final_clip_path_to_return = None
#     sub_clips_for_concat = []
#     concatenated_clip_obj = None 

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)
#                 source_audio_fps = None

#             default_fontsize = 24
#             default_text_color = 'white'

#             for i, segment_info in enumerate(segments_to_include):
#                 start = segment_info.get("start_time")
#                 end = segment_info.get("end_time")
#                 text_content = segment_info.get("text_content")

#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid segment times for segment {i+1}: start={start}, end={end}. Skipping.", extra=log_extra)
#                     continue
                
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration: 
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} start time {start:.2f}s at/beyond video duration {current_main_duration:.2f}s. Skipping.", extra=log_extra)
#                     continue
                
#                 end = min(end, current_main_duration)
#                 if start == end: 
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} has zero duration after capping (start {start:.2f}s == end {end:.2f}s). Skipping.", extra=log_extra)
#                     continue

#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
#                 sub_clip = main_video_clip.subclip(start, end)
                 
#                 if sub_clip.audio is None:
#                     logger.warning(f"video_id: {video_id} - Subclip {i+1} ({start:.2f}-{end:.2f}) has NO audio! Main clip audio was: {main_video_clip.audio is not None}", extra=log_extra)
#                 else:
#                     logger.info(f"video_id: {video_id} - Subclip {i+1} ({start:.2f}-{end:.2f}) has audio. Duration: {sub_clip.audio.duration:.2f}, FPS: {sub_clip.audio.fps}", extra=log_extra)
#                     if source_audio_fps is None and sub_clip.audio.fps:
#                         source_audio_fps = sub_clip.audio.fps
#                         logger.info(f"video_id: {video_id} - Inferred source_audio_fps from subclip {i+1}: {source_audio_fps}", extra=log_extra)
                
#                 if text_content and str(text_content).strip():
#                     try:
#                         text_to_render = str(text_content)
#                         txt_clip = TextClip(
#                             text_to_render, fontsize=default_fontsize, color=default_text_color, font="DejaVu-Sans",
#                             method='caption', size=(sub_clip.w * 0.9, None), bg_color='transparent',
#                             stroke_color='black', stroke_width=1
#                         )
#                         text_height_estimate = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
#                         y_position = sub_clip.h - text_height_estimate - (sub_clip.h * 0.05)
#                         txt_clip = txt_clip.set_pos(('center', y_position)).set_duration(sub_clip.duration)
                        
#                         if sub_clip.mask is None and 'transparent' in str(txt_clip.bg_color).lower():
#                              if sub_clip.ismask: pass
#                              else: sub_clip = sub_clip.add_mask()
#                         sub_clip = CompositeVideoClip([sub_clip, txt_clip], use_bgclip=True)
#                         if sub_clip.audio:
#                             temp_subclip_audio_path = os.path.join(highlights_output_dir, f"DEBUG_audio_of_subclip_{i+1}.aac")
#                             try:
#                                 logger.info(f"video_id: {video_id} - DEBUG: Writing audio of sub_clip {i+1} to {temp_subclip_audio_path}", extra=log_extra)
#                                 sub_clip_audio_fps = getattr(sub_clip.audio, 'fps', source_audio_fps or 44100)
#                                 sub_clip.audio.write_audiofile(temp_subclip_audio_path, fps=sub_clip_audio_fps, codec='aac', logger=None)
#                                 logger.info(f"video_id: {video_id} - DEBUG: Successfully wrote audio of sub_clip {i+1}.", extra=log_extra)
#                                 # NOW PLAY THIS FILE MANUALLY AND CHECK IF IT HAS THE ORIGINAL AUDIO OR A BEEP
#                             except Exception as e_sc_audio:
#                                 logger.error(f"video_id: {video_id} - DEBUG: Failed to write audio of sub_clip {i+1}: {e_sc_audio}", extra=log_extra)
#                         else:
#                             logger.warning(f"video_id: {video_id} - DEBUG: sub_clip {i+1} has NO audio after potential TextClip compositing.", extra=log_extra)

#                         sub_clips_for_concat.append(sub_clip)

#                         logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
#                     except Exception as e_textclip:
#                         logger.error(f"video_id: {video_id} - Failed to create or composite TextClip for segment {i+1}: {e_textclip}. No subtitle.", extra=log_extra)
                
#                 # sub_clips_for_concat.append(sub_clip)

#             if not sub_clips_for_concat:
#                 error_msg = "No valid sub-clips were generated to concatenate."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             logger.info(f"video_id: {video_id} - Attempting to concatenate {len(sub_clips_for_concat)} sub-clips.", extra=log_extra)
#             concatenated_clip_obj = concatenate_videoclips(sub_clips_for_concat, method="compose")
#             logger.info(f"video_id: {video_id} - Concatenated. Final video duration: {concatenated_clip_obj.duration:.2f}s", extra=log_extra)

#             has_audio_for_write = False
#             audio_fps_for_write = None
#             if concatenated_clip_obj.audio:
#                 logger.info(f"video_id: {video_id} - Concatenated clip HAS AUDIO. Type: {type(concatenated_clip_obj.audio)}, FPS: {getattr(concatenated_clip_obj.audio, 'fps', 'N/A')}", extra=log_extra)
#                 has_audio_for_write = True
#                 audio_fps_for_write = getattr(concatenated_clip_obj.audio, 'fps', None)
#                 if not audio_fps_for_write and source_audio_fps:
#                     logger.warning(f"video_id: {video_id} - Concatenated audio missing FPS, using source_audio_fps: {source_audio_fps}", extra=log_extra)
#                     concatenated_clip_obj.audio.fps = source_audio_fps
#                     audio_fps_for_write = source_audio_fps
#                 elif not audio_fps_for_write:
#                     audio_fps_for_write = 44100
#                     logger.warning(f"video_id: {video_id} - Audio FPS for write_videofile defaulted to 44100.", extra=log_extra)
#             else:
#                 logger.error(f"video_id: {video_id} - CRITICAL: Concatenated clip has NO AUDIO component.", extra=log_extra)

#             logger.info(f"video_id: {video_id} - Preparing to write final clip. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)

#             concatenated_clip_obj.write_videofile(
#                 final_clip_path, codec="libx264",
#                 audio_codec="aac" if has_audio_for_write else None,
#                 audio=has_audio_for_write,
#                 temp_audiofile=f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                 remove_temp=False,
#                 threads=os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
#                 fps=main_video_fps_for_output,
#                 audio_fps=audio_fps_for_write if has_audio_for_write else None,
#                 logger='bar'
#             )
#             logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#             final_clip_path_to_return = final_clip_path
        
#         if final_clip_path_to_return:
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(
#                     session_after_write, video_id, highlight_clip_path=final_clip_path_to_return
#                 )
#                 await database_service.update_video_status_and_error(
#                     session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED
#                 )
#                 logger.info(f"video_id: {video_id} - DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             error_msg_no_output = "Highlight generation completed but no output path was produced."
#             logger.error(f"video_id: {video_id} - {error_msg_no_output}", extra=log_extra)
#             async with get_db_session() as error_session_final:
#                 current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
#                 if current_rec and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                      await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
#             return None

#     except Exception as e_main:
#         error_msg_main = f"Highlight generation error: {str(e_main)}"
#         logger.exception(f"video_id: {video_id} - Overall error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session_main:
#             await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
#         return None
#     finally:
#         for clip in sub_clips_for_concat:
#             if clip:
#                 try: clip.close()
#                 except: pass
#         if concatenated_clip_obj:
#             try: concatenated_clip_obj.close()
#             except: pass
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)


# # --- if __name__ == "__main__": block for direct testing ---
# # This block allows you to run 'python -m services.clip_builder_service'
# # It uses mocks for database interactions.
# _original_db_service_module_for_test = database_service # Save before potential patching

# if __name__ == "__main__":
#     # Configure logging specifically for when this script is run directly
#     DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
#     if not logging.getLogger().hasHandlers(): # Avoid reconfiguring if already set (e.g., by importing main)
#         logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
#     else: # If handlers exist (e.g. from main.py if it imported this first somehow), try to set format for current logger
#         for handler in logging.getLogger(__name__).handlers or logging.getLogger().handlers: # Attempt to set format on existing handlers
#              handler.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT))
#         logging.getLogger(__name__).setLevel(logging.DEBUG)


#     async def test_clip_generation_main():
#         logger.info("Running clip_builder_service.py directly for testing (MUSIC DISABLED)...")
        
#         test_video_id_main = "clip_builder_direct_test_003"
        
#         # --- Mock Database Service for this direct test ---
#         class MockDBSessionMain:
#             async def __aenter__(self): return self
#             async def __aexit__(self, exc_type, exc, tb): pass
        
#         class MockDatabaseServiceModuleMain: # This will be our mock 'database_service' module
#             VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
            
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]:
#                 logger.debug("MOCK DB (clip_builder_test): get_db_session called")
#                 yield MockDBSessionMain()

#             async def get_video_record_by_uuid(self, session, video_id):
#                 logger.info(f"MOCK DB: get_video_record_by_uuid for video_id: {video_id}")
#                 dummy_video_filename = "sample_clip_builder_test_video.mp4" # Created in current dir for test
#                 if not os.path.exists(dummy_video_filename):
#                     logger.info(f"Creating dummy video '{dummy_video_filename}' for direct test...")
#                     try:
#                         import subprocess
#                         subprocess.run([
#                             "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=160x120:rate=15",
#                             "-f", "lavfi", "-i", "sine=frequency=440:duration=10:sample_rate=44100",
#                             "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename
#                         ], check=True, capture_output=True, text=True)
#                         logger.info(f"Created dummy test video: {dummy_video_filename}")
#                     except Exception as e_ffmpeg_dummy:
#                         logger.error(f"Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}.")
#                         return None 
                
#                 class MockVideoRecord: pass
#                 record = MockVideoRecord()
#                 record.original_video_file_path = os.path.abspath(dummy_video_filename)
#                 record.processing_status = self.VideoProcessingStatus.READY_FOR_SEARCH # Use a valid status from your enum
#                 return record

#             async def update_video_asset_paths_record(self, session, video_id, highlight_clip_path=None, **kwargs):
#                 logger.info(f"MOCK DB: video_id={video_id}, highlight_clip_path={highlight_clip_path}")
#                 return True

#             async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
#                 logger.info(f"MOCK DB: video_id={video_id}, status={status.value}, error='{error_msg}'")
#                 return True
        
#         # Monkey patch the 'database_service' that this module (clip_builder_service) sees
#         # This needs to be done carefully due to import mechanisms.
#         # The 'database_service' used by generate_highlight_clip is the one imported at the top.
#         global database_service # Declare intent to modify the module-level 'database_service'
#         _local_original_db_service = database_service # Save the one currently in this module's scope
#         database_service = MockDatabaseServiceModuleMain() # Patch it
#         # --- End Mock Database Service ---

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.join(current_dir, "..")
#         # Create test output dir inside the 'backend' directory if it doesn't exist
#         test_output_parent_dir = os.path.join(project_root, "direct_test_output_clip_builder")
#         ensure_dir(test_output_parent_dir)
#         test_processing_base = os.path.join(test_output_parent_dir, test_video_id_main)
#         ensure_dir(test_processing_base)
#         logger.info(f"Direct test: Output base path: {test_processing_base}")

#         segments_to_test = [
#             {"start_time": 0.5, "end_time": 2.5, "text_content": "Builder Test Seg 1"},
#             {"start_time": 4.0, "end_time": 6.0, "text_content": "Interesting Moment Two"},
#         ]

#         logger.info(f"Attempting to generate highlight for test video_id: {test_video_id_main} with {len(segments_to_test)} segments.")
        
#         highlight_file_path = None
#         try:
#             highlight_file_path = await generate_highlight_clip(
#                 video_id=test_video_id_main,
#                 segments_to_include=segments_to_test,
#                 processing_base_path=test_processing_base,
#                 output_filename=f"direct_test_highlight_{test_video_id_main}.mp4"
#             )

#             if highlight_file_path and os.path.exists(highlight_file_path):
#                 logger.info(f"Direct test: Highlight clip GENERATED SUCCESSFULLY: {highlight_file_path}")
#                 logger.info("Please manually verify the video content, audio, and subtitles.")
#             else:
#                 logger.error(f"Direct test: Highlight clip generation FAILED or path not returned: {highlight_file_path}")
#         except Exception as e_test_main:
#             logger.exception(f"Direct test: Exception during clip generation test for video_id: {test_video_id_main}")
#         finally:
#             database_service = _local_original_db_service # Restore original module reference
#             logger.info("Restored original database_service module after direct test.")
            
#             dummy_video_to_clean = "sample_clip_builder_test_video.mp4"
#             if os.path.exists(dummy_video_to_clean): 
#                 try: os.remove(dummy_video_to_clean)
#                 except Exception as e_clean: logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")
#             # Optionally clean up test_processing_base directory
#             # import shutil
#             # if os.path.exists(test_processing_base): shutil.rmtree(test_processing_base)

#     import asyncio
#     import subprocess # Ensure subprocess is imported for the dummy video creation
    
#     # Save the original database_service module reference from the global scope of this file
#     # This is done *outside* the test function to capture the true module-level import.
#     _original_db_service_module_for_test = database_service 

#     asyncio.run(test_clip_generation_main())














































































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid # For unique temp audio file and default output filename
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager

# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
# )
# # from moviepy.config import change_settings # Uncomment if ImageMagick path needs explicit setting

# MUSIC_ENABLED = False # Explicitly disable music generation for now

# # --- Import Database Service ---
# # This 'database_service' will be the one potentially patched in __main__ for testing
# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__) # Will be 'services.clip_builder_service'
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise # Re-raise to be handled by the caller

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]: # Returns path to the generated clip or None on failure
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided for highlight generation. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     source_audio_fps: Optional[int] = None # Initialize as None, try to detect

#     # --- Step 1: Fetch original video path from DB ---
#     async with get_db_session() as session: # This session is for this DB interaction only
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             error_msg = "Original video record or path not found in DB for clip generation."
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             # Attempt to update status if HIGHLIGHT_GENERATING was set by caller
#             # This assumes the caller (main.py) sets HIGHLIGHT_GENERATING before queueing this task.
#             # If not, this update_video_status_and_error might be the first one.
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     # --- Step 2: Prepare output paths ---
#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir:
#         logger.exception(f"video_id: {video_id} - Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
#         async with get_db_session() as error_session: # New session for this isolated error update
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {str(e_dir)}")
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     final_clip_path_to_return = None
#     sub_clips_for_concat = []
#     concatenated_clip_obj = None # To ensure it can be closed in finally

#     try:
#         # --- Step 3: Process video and generate clip using MoviePy ---
#         # MoviePy operations are CPU/GPU bound and blocking.
#         # They will run in FastAPI's BackgroundTasks' threadpool.
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s, Channels: {getattr(main_video_clip.audio, 'nchannels', 'N/A')}", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)
#                 source_audio_fps = None

#             default_fontsize = 24
#             default_text_color = 'white'

#             for i, segment_info in enumerate(segments_to_include):
#                 start = segment_info.get("start_time")
#                 end = segment_info.get("end_time")
#                 text_content = segment_info.get("text_content")

#                 # --- Segment Validation ---
#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid/non-numeric segment times for segment {i+1}: start={start}, end={end}. Skipping.", extra=log_extra)
#                     continue
                
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration: 
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} start time {start:.2f}s at/beyond video duration {current_main_duration:.2f}s. Skipping.", extra=log_extra)
#                     continue
                
#                 end = min(end, current_main_duration) # Cap end time first
#                 if start == end: # Check for zero duration after capping
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} has zero duration after capping (start {start:.2f}s == end {end:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 # --- End Segment Validation ---

#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
#                 # Create the subclip
#                 sub_clip = main_video_clip.subclip(start, end)
                 
#                 # Log audio status of this specific subclip
#                 if sub_clip.audio:
#                     logger.info(f"video_id: {video_id} - Subclip {i+1} ({start:.2f}-{end:.2f}) has audio. Duration: {sub_clip.audio.duration:.2f}, FPS: {sub_clip.audio.fps}", extra=log_extra)
#                     if source_audio_fps is None and sub_clip.audio.fps: # Try to get FPS from first valid subclip
#                         source_audio_fps = sub_clip.audio.fps
#                         logger.info(f"video_id: {video_id} - Inferred source_audio_fps from subclip {i+1}: {source_audio_fps}", extra=log_extra)
#                 else:
#                     logger.warning(f"video_id: {video_id} - Subclip {i+1} ({start:.2f}-{end:.2f}) has NO audio! Main clip audio was: {main_video_clip.audio is not None}", extra=log_extra)
                
#                 # Attempt to add subtitles
#                 final_sub_clip_for_this_segment = sub_clip # Start with the original subclip
#                 if text_content and str(text_content).strip():
#                     try:
#                         text_to_render = str(text_content)
#                         txt_clip = TextClip(
#                             text_to_render, fontsize=default_fontsize, color=default_text_color, font="DejaVu-Sans", # Common font
#                             method='caption', size=(sub_clip.w * 0.9, None), bg_color='transparent',
#                             stroke_color='black', stroke_width=1
#                         )
#                         text_height_estimate = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
#                         y_position = sub_clip.h - text_height_estimate - (sub_clip.h * 0.05) # 5% from bottom
#                         txt_clip = txt_clip.set_pos(('center', y_position)).set_duration(sub_clip.duration)
                        
#                         current_sub_clip_video = sub_clip.without_audio() if sub_clip.audio else sub_clip
#                         if current_sub_clip_video.mask is None and 'transparent' in str(txt_clip.bg_color).lower():
#                              if not current_sub_clip_video.ismask : current_sub_clip_video = current_sub_clip_video.add_mask()

#                         composited_video = CompositeVideoClip([current_sub_clip_video, txt_clip], use_bgclip=True)
                        
#                         if sub_clip.audio: # Re-attach original audio
#                             final_sub_clip_for_this_segment = composited_video.set_audio(sub_clip.audio)
#                         else:
#                             final_sub_clip_for_this_segment = composited_video
                            
#                         logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
#                     except Exception as e_textclip:
#                         logger.error(f"video_id: {video_id} - Failed to create or composite TextClip for segment {i+1}: {e_textclip}. Segment will use original subclip without subtitle.", extra=log_extra)
#                         # final_sub_clip_for_this_segment remains the original sub_clip
                
#                 sub_clips_for_concat.append(final_sub_clip_for_this_segment)
#             # End of segment processing loop

#             if not sub_clips_for_concat:
#                 error_msg = "No valid sub-clips were generated after processing segments."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             logger.info(f"video_id: {video_id} - Attempting to concatenate {len(sub_clips_for_concat)} sub-clips.", extra=log_extra)
#             concatenated_clip_obj = concatenate_videoclips(sub_clips_for_concat, method="compose")
#             logger.info(f"video_id: {video_id} - Concatenated. Final video duration: {concatenated_clip_obj.duration:.2f}s", extra=log_extra)

#             # --- Crucial Audio Check and Handling for Concatenated Clip ---
#             has_audio_for_write = False
#             audio_fps_for_write = None
#             if concatenated_clip_obj.audio:
#                 logger.info(f"video_id: {video_id} - Concatenated clip HAS AUDIO. Type: {type(concatenated_clip_obj.audio)}, FPS: {getattr(concatenated_clip_obj.audio, 'fps', 'N/A')}, Duration: {getattr(concatenated_clip_obj.audio, 'duration', 'N/A')}", extra=log_extra)
#                 has_audio_for_write = True
#                 audio_fps_for_write = getattr(concatenated_clip_obj.audio, 'fps', None)
#                 if not audio_fps_for_write and source_audio_fps: # If source had audio FPS
#                     logger.warning(f"video_id: {video_id} - Concatenated audio missing FPS, attempting to set from source_audio_fps: {source_audio_fps}", extra=log_extra)
#                     try:
#                         concatenated_clip_obj.audio.fps = source_audio_fps
#                         audio_fps_for_write = source_audio_fps
#                     except Exception as e_set_fps: # Some audio clip types might not allow direct fps setting
#                         logger.error(f"video_id: {video_id} - Failed to set FPS on concatenated audio: {e_set_fps}. Will use fallback if needed.", extra=log_extra)
                
#                 if not audio_fps_for_write: # If still no FPS after attempting to set from source
#                     audio_fps_for_write = 44100 # Absolute fallback
#                     logger.warning(f"video_id: {video_id} - Audio FPS for write_videofile defaulted to 44100 after all checks.", extra=log_extra)
#                     # Attempt to set it on the object if it's missing for the write
#                     if hasattr(concatenated_clip_obj.audio, 'fps'):
#                         concatenated_clip_obj.audio.fps = audio_fps_for_write

#             else: # concatenated_clip_obj.audio is None
#                 logger.error(f"video_id: {video_id} - CRITICAL: Concatenated clip has NO AUDIO component after concatenate_videoclips.", extra=log_extra)
            
#             logger.info(f"video_id: {video_id} - Preparing to write final clip. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)

#             # Write the final video
#             concatenated_clip_obj.write_videofile(
#                 final_clip_path,
#                 codec="libx264",
#                 audio_codec="aac" if has_audio_for_write else None,
#                 audio=has_audio_for_write,
#                 temp_audiofile=f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                 remove_temp=True, # Set to False to inspect the temp audio file
#                 threads=os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
#                 fps=main_video_fps_for_output,
#                 audio_fps=audio_fps_for_write if has_audio_for_write else None,
#                 logger='bar' # Shows FFmpeg command and output
#             )
#             logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#             final_clip_path_to_return = final_clip_path
#         # End of 'with VideoFileClip(...)' -> main_video_clip is closed here

#         # --- Update Database after successful write ---
#         if final_clip_path_to_return:
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(
#                     session_after_write, video_id, highlight_clip_path=final_clip_path_to_return
#                 )
#                 await database_service.update_video_status_and_error(
#                     session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED
#                 )
#                 logger.info(f"video_id: {video_id} - DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             # This case should ideally be caught earlier if sub_clips_for_concat was empty,
#             # or if write_videofile failed (which would raise an exception caught below).
#             error_msg_no_output = "Highlight generation process completed but no output path was set (e.g. no valid segments or write failed silently)."
#             logger.error(f"video_id: {video_id} - {error_msg_no_output}", extra=log_extra)
#             async with get_db_session() as error_session_final:
#                 current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
#                 if current_rec and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                      await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
#             return None

#     except Exception as e_main:
#         error_msg_main = f"Highlight generation error: {str(e_main)}"
#         logger.exception(f"video_id: {video_id} - Overall error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session_main: # New session for this specific error update
#             await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
#         return None
#     finally:
#         # Explicitly close MoviePy objects that were created and might hold resources
#         for clip in sub_clips_for_concat: # Close individual sub_clips that were appended
#             if clip:
#                 try: clip.close()
#                 except Exception: pass 
#         if concatenated_clip_obj: # Close the concatenated clip if it was created
#             try: concatenated_clip_obj.close()
#             except Exception: pass
#         # main_video_clip is closed by its 'with' statement.
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)


# # --- if __name__ == "__main__": block for direct testing ---
# # This global variable will store the original database_service module
# _original_db_service_module_for_test = database_service

# if __name__ == "__main__":
#     import asyncio
#     import subprocess # Ensure subprocess is imported for the dummy video creation
    
#     # Configure logging specifically for when this script is run directly
#     DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
#     # Check if handlers are already configured (e.g. if this module was imported then run)
#     if not logging.getLogger().hasHandlers(): 
#         logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
#     else: # If handlers exist, try to set format for current module's logger
#         current_module_logger = logging.getLogger(__name__)
#         current_module_logger.setLevel(logging.DEBUG)
#         # This part is tricky; ideally, for direct run, you'd want full control
#         # or no prior config. If run as 'python -m ...' basicConfig here is authoritative.
#         # If prior config exists, we might not be able to easily override root handlers' formatters here.
#         # The 'if not logging.getLogger().hasHandlers()' is the cleaner path for direct script runs.

#     async def test_clip_generation_main():
#         logger.info("Running clip_builder_service.py directly for testing (MUSIC DISABLED)...")
        
#         test_video_id_main = "clip_builder_direct_test_004" # New ID for cleaner testing
        
#         # --- Mock Database Service for this direct test ---
#         class MockDBSessionMain:
#             async def __aenter__(self): return self
#             async def __aexit__(self, exc_type, exc, tb): pass
        
#         class MockDatabaseServiceModuleMain:
#             VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
            
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]:
#                 logger.debug("MOCK DB (clip_builder_test): get_db_session called")
#                 yield MockDBSessionMain()

#             async def get_video_record_by_uuid(self, session, video_id):
#                 logger.info(f"MOCK DB (get_video_record_by_uuid): video_id={video_id}")
#                 dummy_video_filename = "sample_clip_builder_test_video.mp4"
#                 if not os.path.exists(dummy_video_filename):
#                     logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename}' for direct test...")
#                     try:
#                         subprocess.run([
#                             "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=24", # Increased rate
#                             "-f", "lavfi", "-i", "sine=frequency=440:duration=10:sample_rate=44100", # Added audio stream
#                             "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename
#                         ], check=True, capture_output=True, text=True)
#                         logger.info(f"MOCK DB: Created dummy test video: {dummy_video_filename}")
#                     except Exception as e_ffmpeg_dummy:
#                         logger.error(f"MOCK DB: Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}.")
#                         return None 
                
#                 class MockVideoRecord: pass
#                 record = MockVideoRecord()
#                 record.original_video_file_path = os.path.abspath(dummy_video_filename)
#                 record.processing_status = self.VideoProcessingStatus.READY_FOR_SEARCH
#                 return record

#             async def update_video_asset_paths_record(self, session, video_id, highlight_clip_path=None, **kwargs):
#                 logger.info(f"MOCK DB (update_asset_paths): video_id={video_id}, highlight_clip_path={highlight_clip_path}")
#                 return True

#             async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
#                 logger.info(f"MOCK DB (update_status): video_id={video_id}, status={status.value}, error='{error_msg}'")
#                 return True
        
#         global database_service # Declare intent to modify the module-level 'database_service'
#         # _local_original_db_service = database_service # This was causing UnboundLocalError
#         database_service = MockDatabaseServiceModuleMain() # Monkey patch
#         # --- End Mock Database Service ---

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.join(current_dir, "..")
#         # Create test output dir inside the 'backend' directory if it doesn't exist
#         test_output_parent_dir = os.path.join(project_root, "direct_test_output_clip_builder")
#         ensure_dir(test_output_parent_dir)
#         test_processing_base = os.path.join(test_output_parent_dir, test_video_id_main)
#         ensure_dir(test_processing_base)
#         logger.info(f"Direct test: Output base path: {test_processing_base}")

#         segments_to_test = [
#             {"start_time": 0.5, "end_time": 2.5, "text_content": "Test Segment Alpha"},
#             {"start_time": 4.0, "end_time": 6.0, "text_content": "Test Segment Beta with more text"},
#             {"start_time": 7.0, "end_time": 7.0, "text_content": "Zero Duration Invalid Segment"}, # Will be skipped
#             {"start_time": 11.0, "end_time": 12.0, "text_content": "Exceeds Duration (10s video)"} # Will be skipped/capped
#         ]

#         logger.info(f"Attempting to generate highlight for test video_id: {test_video_id_main} with {len(segments_to_test)} segments.")
        
#         highlight_file_path = None
#         try:
#             highlight_file_path = await generate_highlight_clip(
#                 video_id=test_video_id_main,
#                 segments_to_include=segments_to_test,
#                 processing_base_path=test_processing_base,
#                 output_filename=f"direct_test_highlight_{test_video_id_main}.mp4"
#             )

#             if highlight_file_path and os.path.exists(highlight_file_path):
#                 logger.info(f"Direct test: Highlight clip GENERATED SUCCESSFULLY: {highlight_file_path}")
#                 logger.info("Please manually verify the video content, audio, and subtitles.")
#             else:
#                 logger.error(f"Direct test: Highlight clip generation FAILED or path not returned: {highlight_file_path}")
#         except Exception as e_test_main:
#             logger.exception(f"Direct test: Exception during clip generation test for video_id: {test_video_id_main}")
#         finally:
#             # Restore original 'database_service' module that was imported at the top of the file
#             database_service = _original_db_service_module_for_test 
#             logger.info("Restored original database_service module after direct test.")
            
#             dummy_video_to_clean = "sample_clip_builder_test_video.mp4"
#             if os.path.exists(dummy_video_to_clean): 
#                 try: os.remove(dummy_video_to_clean)
#                 except Exception as e_clean: logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")

#     asyncio.run(test_clip_generation_main())














































































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid # For potential unique filenames if needed later, not primary here
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager

# from moviepy.editor import VideoFileClip
# # TextClip, CompositeVideoClip, concatenate_videoclips are not needed for this specific test

# MUSIC_ENABLED = False # Not relevant for this test

# # --- Import Database Service ---
# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__)
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights" # Still define for path consistency

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

# async def generate_highlight_clip( # Renaming to reflect test focus
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None # Not used in this test
# ) -> Optional[str]: # Returns path to the DEBUG audio file, or None
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - AUDIO TEST 1: Testing RAW subclip audio. Segments received: {len(segments_to_include)}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided. Aborting test.", extra=log_extra)
#         return None

#     # --- For audio debug, process only the FIRST segment ---
#     segment_info = segments_to_include[0]
#     logger.info(f"video_id: {video_id} - Using first segment for test: {segment_info}", extra=log_extra)

#     original_video_file_path: Optional[str] = None
#     source_audio_fps: Optional[int] = None # Try to detect from main clip

#     # --- Fetch original video path from DB ---
#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             logger.error(f"video_id: {video_id} - Original video record or path not found in DB.", extra=log_extra)
#             # No DB status update here as this is a focused debug function now
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             logger.error(f"video_id: {video_id} - Original video file not found at path: {video_record.original_video_file_path}", extra=log_extra)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError: # Error already logged by ensure_dir
#         logger.error(f"video_id: {video_id} - Cannot proceed without output directory.", extra=log_extra)
#         return None
    
#     debug_single_subclip_audio_path: Optional[str] = None # Initialize

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Test will likely produce no audio.", extra=log_extra)
#                 source_audio_fps = None


#             start = segment_info.get("start_time")
#             end = segment_info.get("end_time")

#             # --- Segment Validation (simplified for this single segment test) ---
#             if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                 logger.warning(f"video_id: {video_id} - Invalid segment times for test: start={start}, end={end}. Skipping.", extra=log_extra)
#                 return None
            
#             current_main_duration = main_video_clip.duration
#             if start >= current_main_duration: 
#                 logger.warning(f"video_id: {video_id} - Test segment start time {start:.2f}s at/beyond video duration {current_main_duration:.2f}s. Skipping.", extra=log_extra)
#                 return None
            
#             end = min(end, current_main_duration) # Cap end time
#             if start == end: 
#                 logger.warning(f"video_id: {video_id} - Test segment has zero duration after capping (start {start:.2f}s == end {end:.2f}s). Skipping.", extra=log_extra)
#                 return None
#             # --- End Segment Validation ---

#             logger.info(f"video_id: {video_id} - AUDIO TEST 1: Processing segment: {start:.2f}s - {end:.2f}s", extra=log_extra)
#             sub_clip_direct = main_video_clip.subclip(start, end)
            
#             if sub_clip_direct.audio:
#                 logger.info(f"video_id: {video_id} - AUDIO TEST 1: RAW sub_clip has audio. Duration: {sub_clip_direct.audio.duration:.2f}, FPS: {sub_clip_direct.audio.fps}", extra=log_extra)
                
#                 debug_single_subclip_audio_path = os.path.join(highlights_output_dir, f"DEBUG_RAW_SUBCLIP_AUDIO_{video_id}_seg1.aac")
#                 try:
#                     logger.info(f"video_id: {video_id} - AUDIO TEST 1: Writing audio of RAW sub_clip to {debug_single_subclip_audio_path}", extra=log_extra)
                    
#                     # Attempt to use detected FPS, fallback if necessary
#                     sub_clip_audio_fps_to_use = getattr(sub_clip_direct.audio, 'fps', None) or source_audio_fps
#                     if not sub_clip_audio_fps_to_use:
#                         sub_clip_audio_fps_to_use = 44100 # Absolute fallback
#                         logger.warning(f"video_id: {video_id} - AUDIO TEST 1: Subclip audio FPS not detected, defaulting to {sub_clip_audio_fps_to_use} for write.", extra=log_extra)
                    
#                     sub_clip_direct.audio.write_audiofile(
#                         debug_single_subclip_audio_path, 
#                         fps=sub_clip_audio_fps_to_use, 
#                         codec='aac', 
#                         logger=None # Set to 'bar' for FFmpeg output, None for less verbose
#                     )
#                     logger.info(f"video_id: {video_id} - AUDIO TEST 1: Successfully wrote audio of RAW sub_clip.", extra=log_extra)
#                     logger.info(f"AUDIO TEST 1: PLEASE MANUALLY PLAY AND VERIFY: {debug_single_subclip_audio_path}")
#                     return debug_single_subclip_audio_path # Return path for checking

#                 except Exception as e_sc_audio_direct:
#                     logger.exception(f"video_id: {video_id} - AUDIO TEST 1: Failed to write audio of RAW sub_clip: {e_sc_audio_direct}", extra=log_extra)
#                     return None
#             else:
#                 logger.warning(f"video_id: {video_id} - AUDIO TEST 1: RAW sub_clip created with NO audio component!", extra=log_extra)
#                 return None
#         # VideoFileClip is closed by 'with'

#     except Exception as e_main_test1:
#         logger.exception(f"video_id: {video_id} - Overall error in AUDIO TEST 1 for generate_highlight_clip: {e_main_test1}", extra=log_extra)
#         return None
#     finally:
#         # No complex objects like concatenated_clip_obj to close in this focused test
#         logger.debug(f"video_id: {video_id} - Exiting AUDIO TEST 1 function (finally block).", extra=log_extra)


# # --- if __name__ == "__main__": block for direct testing ---
# _original_db_service_module_for_test = database_service

# if __name__ == "__main__":
#     import asyncio
#     import subprocess
    
#     DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
#     if not logging.getLogger().hasHandlers():
#         logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
#     else:
#         for handler in logging.getLogger(__name__).handlers or logging.getLogger().handlers:
#              handler.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT))
#         logging.getLogger(__name__).setLevel(logging.DEBUG)

#     async def test_clip_generation_main():
#         logger.info("Running clip_builder_service.py (AUDIO TEST 1 - RAW SUBCLIP AUDIO)...")
        
#         test_video_id_main = "clip_builder_audiotest1_001"
        
#         class MockDBSessionMain:
#             async def __aenter__(self): return self
#             async def __aexit__(self, exc_type, exc, tb): pass
        
#         class MockDatabaseServiceModuleMain:
#             VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]:
#                 logger.debug("MOCK DB (clip_builder_test): get_db_session called")
#                 yield MockDBSessionMain()
#             async def get_video_record_by_uuid(self, session, video_id):
#                 logger.info(f"MOCK DB: get_video_record_by_uuid for video_id: {video_id}")
#                 dummy_video_filename = "sample_clip_builder_test_video_audio_test.mp4"
#                 if not os.path.exists(dummy_video_filename):
#                     logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename}' for audio test...")
#                     try:
#                         subprocess.run([
#                             "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=160x120:rate=15", # Shorter 5s video
#                             "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100", # 5s sine wave
#                             "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename
#                         ], check=True, capture_output=True, text=True)
#                         logger.info(f"MOCK DB: Created dummy test video: {dummy_video_filename}")
#                     except Exception as e_ffmpeg_dummy:
#                         logger.error(f"MOCK DB: Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}.")
#                         return None 
                
#                 class MockVideoRecord: pass
#                 record = MockVideoRecord()
#                 record.original_video_file_path = os.path.abspath(dummy_video_filename)
#                 record.processing_status = self.VideoProcessingStatus.READY_FOR_SEARCH
#                 return record
#             async def update_video_asset_paths_record(self, session, video_id, **kwargs): # Simplified
#                 logger.info(f"MOCK DB (update_asset_paths): video_id={video_id}, assets_updated_with_kwargs: {kwargs}")
#                 return True
#             async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
#                 logger.info(f"MOCK DB (update_status): video_id={video_id}, status={status.value}, error='{error_msg}'")
#                 return True
        
#         global database_service
#         database_service = MockDatabaseServiceModuleMain()

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.join(current_dir, "..")
#         test_processing_base = os.path.join(project_root, "direct_test_output_clip_builder_audio_test", test_video_id_main)
#         ensure_dir(test_processing_base)
#         logger.info(f"Direct test: Output base path: {test_processing_base}")

#         # Test with only one segment to isolate its audio
#         segments_to_test = [
#             {"start_time": 0.5, "end_time": 2.5, "text_content": "This is a test segment for audio."},
#             # {"start_time": 3.0, "end_time": 4.5, "text_content": "Another segment."}, # Comment out for single segment test
#         ]

#         logger.info(f"AUDIO TEST 1: Attempting to generate audio for first segment of video_id: {test_video_id_main}")
        
#         debug_audio_file_path = None
#         try:
#             # The function now directly tests and writes the first segment's audio
#             debug_audio_file_path = await generate_highlight_clip(
#                 video_id=test_video_id_main,
#                 segments_to_include=segments_to_test, # Will only use the first one internally for this test version
#                 processing_base_path=test_processing_base
#             )

#             if debug_audio_file_path and os.path.exists(debug_audio_file_path):
#                 logger.info(f"AUDIO TEST 1: Debug audio file for raw subclip generated: {debug_audio_file_path}")
#                 logger.info(">>>> PLEASE MANUALLY PLAY THIS FILE TO CHECK FOR BEEPS VS ORIGINAL AUDIO <<<<")
#             elif debug_audio_file_path is None and os.path.exists(f"DEBUG_RAW_SUBCLIP_AUDIO_{test_video_id_main}_seg1.aac"): # Check fallback name if needed
#                 logger.info(f"AUDIO TEST 1: Debug audio file for raw subclip generated at implicit path. Please check output directory.")
#             else:
#                 logger.error("AUDIO TEST 1: Debug audio file generation FAILED or path not returned.")
#         except Exception as e_test_main:
#             logger.exception(f"AUDIO TEST 1: Exception during test for video_id: {test_video_id_main}")
#         finally:
#             database_service = _original_db_service_module_for_test
#             logger.info("Restored original database_service module after direct test.")
            
#             dummy_video_to_clean = "sample_clip_builder_test_video_audio_test.mp4"
#             if os.path.exists(dummy_video_to_clean): 
#                 try: os.remove(dummy_video_to_clean)
#                 except Exception as e_clean: logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")

#     asyncio.run(test_clip_generation_main())




























































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager
# import subprocess # For dummy video creation in __main__

# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
# )
# # from moviepy.config import change_settings # Uncomment if ImageMagick path needs explicit setting

# MUSIC_ENABLED = False

# # --- Import Database Service ---
# # This 'database_service' will be the one potentially patched in __main__ for testing
# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__) # Will be 'services.clip_builder_service'
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]: # Returns path to the generated clip or None on failure
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided for highlight generation. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     source_audio_fps: Optional[int] = None

#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             error_msg = "Original video record or path not found in DB for clip generation."
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir:
#         logger.exception(f"video_id: {video_id} - Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {str(e_dir)}")
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     final_clip_path_to_return = None
#     sub_clips_for_concat = []
#     concatenated_clip_obj = None 

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s, Channels: {getattr(main_video_clip.audio, 'nchannels', 'N/A')}", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)

#             default_fontsize = 24
#             default_text_color = 'white'

#             # --- AUDIO TEST 2: Integrated into the main loop for the first text-bearing segment ---
#             audio_test_2_done_for_segment_idx = -1 

#             for i, segment_info_loop in enumerate(segments_to_include):
#                 start = segment_info_loop.get("start_time")
#                 end = segment_info_loop.get("end_time")
#                 text_content_loop = segment_info_loop.get("text_content")

#                 # --- Segment Validation ---
#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"video_id: {video_id} - MainLoop: Invalid segment {i+1} times. Skipping.", extra=log_extra)
#                     continue
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration:
#                     logger.warning(f"video_id: {video_id} - MainLoop: Segment {i+1} start ({start:.2f}s) out of bounds ({current_main_duration:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 end = min(end, current_main_duration) # Cap end time
#                 if start == end:
#                     logger.warning(f"video_id: {video_id} - MainLoop: Segment {i+1} zero duration ({start:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 # --- End Segment Validation ---

#                 logger.info(f"video_id: {video_id} - MainLoop: Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
#                 sub_clip = main_video_clip.subclip(start, end)
                
#                 final_sub_clip_for_concat = sub_clip # Default if no text or text op fails
                
#                 if text_content_loop and str(text_content_loop).strip():
#                     try:
#                         text_to_render = str(text_content_loop)
#                         txt_clip = TextClip(
#                             text_to_render, fontsize=default_fontsize, color=default_text_color, font="DejaVu-Sans",
#                             method='caption', size=(sub_clip.w * 0.9, None), bg_color='transparent',
#                             stroke_color='black', stroke_width=1
#                         )
#                         text_h_est = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
#                         y_pos = sub_clip.h - text_h_est - (sub_clip.h * 0.05)
#                         txt_clip = txt_clip.set_pos(('center', y_pos)).set_duration(sub_clip.duration)
                        
#                         video_part = sub_clip.without_audio() if sub_clip.audio else sub_clip
#                         if video_part.mask is None and 'transparent' in str(txt_clip.bg_color).lower():
#                              if not video_part.ismask : video_part = video_part.add_mask()
                        
#                         composited_video_with_text = CompositeVideoClip([video_part, txt_clip], use_bgclip=True)
                        
#                         if sub_clip.audio: # Re-attach original audio from the subclip
#                             final_sub_clip_for_concat = composited_video_with_text.set_audio(sub_clip.audio)
#                         else:
#                             final_sub_clip_for_concat = composited_video_with_text
#                         logger.info(f"video_id: {video_id} - MainLoop: Added subtitle to segment {i+1}.", extra=log_extra)
#                     except Exception as e_text_main:
#                         logger.error(f"video_id: {video_id} - MainLoop: Failed TextClip for seg {i+1}: {e_text_main}. Using original subclip.", extra=log_extra)
#                         # final_sub_clip_for_concat remains original sub_clip
                
#                 # --- AUDIO TEST 2: Perform for the first segment that had text_content and TextClip was attempted ---
#                 if audio_test_2_done_for_segment_idx == -1 and text_content_loop and str(text_content_loop).strip():
#                     audio_test_2_done_for_segment_idx = i # Mark that we've done this test
#                     logger.info(f"video_id: {video_id} - AUDIO TEST 2: Testing audio of segment {i+1} (after text op).", extra=log_extra)
#                     if final_sub_clip_for_concat.audio:
#                         debug_audio_after_text_path = os.path.join(highlights_output_dir, f"DEBUG_AUDIO_AFTER_TEXT_{video_id}_seg{i+1}.aac")
#                         try:
#                             fps_to_use = getattr(final_sub_clip_for_concat.audio, 'fps', source_audio_fps or 44100)
#                             final_sub_clip_for_concat.audio.write_audiofile(debug_audio_after_text_path, fps=fps_to_use, codec='aac', logger=None)
#                             logger.info(f"video_id: {video_id} - AUDIO TEST 2: Wrote audio. PLEASE PLAY: {debug_audio_after_text_path}", extra=log_extra)
#                         except Exception as e_sc_audio_proc:
#                             logger.error(f"video_id: {video_id} - AUDIO TEST 2: Failed to write audio of processed subclip: {e_sc_audio_proc}", extra=log_extra)
#                     else:
#                         logger.warning(f"video_id: {video_id} - AUDIO TEST 2: Processed subclip for text test (seg {i+1}) has NO audio!", extra=log_extra)
#                 # --- END AUDIO TEST 2 for this segment ---

#                 sub_clips_for_concat.append(final_sub_clip_for_concat)
#             # --- End regular segment processing loop ---

#             if not sub_clips_for_concat:
#                 error_msg = "No valid sub-clips were generated after processing all segments."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             logger.info(f"video_id: {video_id} - Attempting to concatenate {len(sub_clips_for_concat)} sub-clips.", extra=log_extra)
#             concatenated_clip_obj = concatenate_videoclips(sub_clips_for_concat, method="compose")
#             logger.info(f"video_id: {video_id} - Concatenated. Final video duration: {concatenated_clip_obj.duration:.2f}s", extra=log_extra)

#             has_audio_for_write = False
#             audio_fps_for_write = None
#             if concatenated_clip_obj.audio:
#                 logger.info(f"video_id: {video_id} - Concatenated clip HAS AUDIO. Type: {type(concatenated_clip_obj.audio)}, FPS: {getattr(concatenated_clip_obj.audio, 'fps', 'N/A')}, Duration: {getattr(concatenated_clip_obj.audio, 'duration', 'N/A')}", extra=log_extra)
#                 has_audio_for_write = True
#                 audio_fps_for_write = getattr(concatenated_clip_obj.audio, 'fps', None)
#                 if not audio_fps_for_write and source_audio_fps:
#                     logger.warning(f"video_id: {video_id} - Concatenated audio missing FPS, setting from source_audio_fps: {source_audio_fps}", extra=log_extra)
#                     try:
#                         concatenated_clip_obj.audio.fps = source_audio_fps # Attempt to set
#                         audio_fps_for_write = source_audio_fps
#                     except Exception as e_set_fps:
#                         logger.error(f"video_id: {video_id} - Failed to set FPS on concatenated audio: {e_set_fps}. Will use fallback.", extra=log_extra)
                
#                 if not audio_fps_for_write: # Absolute fallback if still None
#                     audio_fps_for_write = 44100
#                     logger.warning(f"video_id: {video_id} - Audio FPS for write_videofile defaulted to 44100.", extra=log_extra)
#                     if hasattr(concatenated_clip_obj.audio, 'fps'): # Try to set it on the object too
#                         concatenated_clip_obj.audio.fps = audio_fps_for_write
#             else:
#                 logger.error(f"video_id: {video_id} - CRITICAL: Concatenated clip has NO AUDIO component after concatenate_videoclips.", extra=log_extra)
            
#             logger.info(f"video_id: {video_id} - Preparing to write final clip. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)

#             concatenated_clip_obj.write_videofile(
#                 final_clip_path, codec="libx264",
#                 audio_codec="aac" if has_audio_for_write else None,
#                 audio=has_audio_for_write,
#                 temp_audiofile=f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                 remove_temp=True, # Set to False to inspect temp audio file if audio bug persists
#                 threads=os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
#                 fps=main_video_fps_for_output,
#                 audio_fps=audio_fps_for_write if has_audio_for_write else None,
#                 logger='bar'
#             )
#             logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#             final_clip_path_to_return = final_clip_path
        
#         if final_clip_path_to_return:
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(
#                     session_after_write, video_id, highlight_clip_path=final_clip_path_to_return
#                 )
#                 await database_service.update_video_status_and_error(
#                     session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED
#                 )
#                 logger.info(f"video_id: {video_id} - DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             error_msg_no_output = "Highlight generation completed but no output path was produced."
#             logger.error(f"video_id: {video_id} - {error_msg_no_output}", extra=log_extra)
#             async with get_db_session() as error_session_final:
#                 current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
#                 if current_rec and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                      await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
#             return None

#     except Exception as e_main:
#         error_msg_main = f"Highlight generation error: {str(e_main)}"
#         logger.exception(f"video_id: {video_id} - Overall error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session_main:
#             await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
#         return None
#     finally:
#         for clip_obj in sub_clips_for_concat: # Changed variable name
#             if clip_obj:
#                 try: clip_obj.close()
#                 except: pass
#         if concatenated_clip_obj:
#             try: concatenated_clip_obj.close()
#             except: pass
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)

# # --- if __name__ == "__main__": block for direct testing ---
# _original_db_service_module_for_test = database_service # Save before potential patching

# if __name__ == "__main__":
#     import asyncio
#     import subprocess
    
#     DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
#     if not logging.getLogger().hasHandlers():
#         logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
#     else:
#         for handler in logging.getLogger(__name__).handlers or logging.getLogger().handlers:
#              handler.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT))
#         logging.getLogger(__name__).setLevel(logging.DEBUG)

#     async def test_clip_generation_main():
#         logger.info("Running clip_builder_service.py (AUDIO TEST 2 - AFTER TEXT COMPOSITE)...")
        
#         test_video_id_main = "clip_builder_audiotest2_001" # Updated test ID
        
#         # --- Mock Database Service ---
#         class MockDBSessionMain:
#             async def __aenter__(self): return self
#             async def __aexit__(self, exc_type, exc, tb): pass
        
#         class MockDatabaseServiceModuleMain:
#             VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]:
#                 logger.debug("MOCK DB (clip_builder_test): get_db_session called")
#                 yield MockDBSessionMain()
#             async def get_video_record_by_uuid(self, session, video_id):
#                 logger.info(f"MOCK DB: get_video_record_by_uuid for video_id: {video_id}")
#                 dummy_video_filename = "sample_clip_builder_test_video_audio_test.mp4"
#                 if not os.path.exists(dummy_video_filename):
#                     logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename}' for audio test...")
#                     try:
#                         subprocess.run([
#                             "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=160x120:rate=15",
#                             "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100",
#                             "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename
#                         ], check=True, capture_output=True, text=True)
#                         logger.info(f"MOCK DB: Created dummy test video: {dummy_video_filename}")
#                     except Exception as e_ffmpeg_dummy:
#                         logger.error(f"MOCK DB: Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}.")
#                         return None 
#                 class MockVideoRecord: pass
#                 record = MockVideoRecord()
#                 record.original_video_file_path = os.path.abspath(dummy_video_filename)
#                 record.processing_status = self.VideoProcessingStatus.READY_FOR_SEARCH # Corrected status
#                 return record
#             async def update_video_asset_paths_record(self, session, video_id, highlight_clip_path=None, **kwargs):
#                 logger.info(f"MOCK DB (update_asset_paths): video_id={video_id}, highlight_clip_path={highlight_clip_path}")
#                 return True
#             async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
#                 logger.info(f"MOCK DB (update_status): video_id={video_id}, status={status.value}, error='{error_msg}'")
#                 return True
        
#         global database_service
#         database_service = MockDatabaseServiceModuleMain()
#         # --- End Mock ---

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.join(current_dir, "..")
#         test_output_parent_dir = os.path.join(project_root, "direct_test_output_clip_builder") # Main dir for all tests
#         ensure_dir(test_output_parent_dir)
#         test_processing_base = os.path.join(test_output_parent_dir, test_video_id_main) # Subdir for this test
#         ensure_dir(test_processing_base)
#         logger.info(f"Direct test: Output base path: {test_processing_base}")

#         segments_to_test = [
#             {"start_time": 0.5, "end_time": 2.5, "text_content": "Audio Test 2 - Segment 1 with Text"},
#             {"start_time": 3.0, "end_time": 4.5, "text_content": "Audio Test 2 - Segment 2 with More Text"}
#         ]

#         logger.info(f"AUDIO TEST 2: Calling generate_highlight_clip for video_id: {test_video_id_main}")
        
#         final_clip_output_path = None
#         try:
#             # This call will now execute the "AUDIO TEST 2" logic internally for the first text segment,
#             # then proceed to process all segments for a (potentially silent if audio bug persists) final clip.
#             final_clip_output_path = await generate_highlight_clip(
#                 video_id=test_video_id_main,
#                 segments_to_include=segments_to_test,
#                 processing_base_path=test_processing_base
#                 # output_filename is optional, will be auto-generated
#             )

#             if final_clip_output_path and os.path.exists(final_clip_output_path):
#                 logger.info(f"AUDIO TEST 2 (and full run): Final clip path: {final_clip_output_path}")
#                 logger.info(">>>> Check logs for 'AUDIO TEST 2: Successfully wrote audio. PLEASE PLAY: ...'")
#                 logger.info(">>>> Also check the final clip for audio and subtitles (if ImageMagick is working).")
#             else:
#                 logger.error("AUDIO TEST 2 (and full run): Highlight clip generation failed or path not returned.")
        
#         except Exception as e_test_main:
#             logger.exception(f"AUDIO TEST 2: Exception during main test call for video_id: {test_video_id_main}")
#         finally:
#             database_service = _original_db_service_module_for_test # Restore
#             logger.info("Restored original database_service module after direct test.")
            
#             dummy_video_to_clean = "sample_clip_builder_test_video_audio_test.mp4"
#             if os.path.exists(dummy_video_to_clean): 
#                 try: os.remove(dummy_video_to_clean)
#                 except Exception as e_clean: logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")

#     asyncio.run(test_clip_generation_main())


















































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager
# import subprocess

# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
#     AudioFileClip, # For loading audio if needed, not directly for concatenating
#     concatenate_audioclips # Explicitly imported
# )
# # from moviepy.config import change_settings

# MUSIC_ENABLED = False

# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__)
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]:
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     source_audio_fps: Optional[int] = None

#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             error_msg = "Original video record/path not found in DB for clip generation."
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir:
#         logger.exception(f"video_id: {video_id} - Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {str(e_dir)}")
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     final_clip_path_to_return = None
#     processed_sub_clips_for_concat = [] # Will hold VideoClip objects (some with audio, some without text if failed)
#     concatenated_clip_obj = None

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)

#             default_fontsize = 24
#             default_text_color = 'white'

#             for i, segment_info_loop in enumerate(segments_to_include):
#                 start = segment_info_loop.get("start_time")
#                 end = segment_info_loop.get("end_time")
#                 text_content_loop = segment_info_loop.get("text_content")

#                 # --- Segment Validation ---
#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid segment {i+1} times. Skipping.", extra=log_extra)
#                     continue
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} start ({start:.2f}s) out of bounds ({current_main_duration:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 end = min(end, current_main_duration)
#                 if start == end:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} zero duration ({start:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 # --- End Segment Validation ---

#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
#                 sub_clip = main_video_clip.subclip(start, end)
                
#                 if sub_clip.audio and source_audio_fps is None and sub_clip.audio.fps:
#                     source_audio_fps = sub_clip.audio.fps # Try to capture from first valid subclip
#                     logger.info(f"video_id: {video_id} - Inferred source_audio_fps from subclip {i+1}: {source_audio_fps}", extra=log_extra)

#                 final_sub_clip_for_this_segment = sub_clip
#                 if text_content_loop and str(text_content_loop).strip():
#                     try:
#                         text_to_render = str(text_content_loop)
#                         txt_clip = TextClip(
#                             text_to_render, fontsize=default_fontsize, color=default_text_color, font="DejaVu-Sans",
#                             method='caption', size=(sub_clip.w * 0.9, None), bg_color='transparent',
#                             stroke_color='black', stroke_width=1
#                         )
#                         text_h_est = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
#                         y_pos = sub_clip.h - text_h_est - (sub_clip.h * 0.05)
#                         txt_clip = txt_clip.set_pos(('center', y_pos)).set_duration(sub_clip.duration)
                        
#                         video_part = sub_clip.without_audio() if sub_clip.audio else sub_clip
#                         if video_part.mask is None and 'transparent' in str(txt_clip.bg_color).lower():
#                              if not video_part.ismask : video_part = video_part.add_mask()
                        
#                         composited_video_with_text = CompositeVideoClip([video_part, txt_clip], use_bgclip=True)
                        
#                         if sub_clip.audio: # Re-attach original audio from the subclip
#                             final_sub_clip_for_this_segment = composited_video_with_text.set_audio(sub_clip.audio)
#                         else:
#                             final_sub_clip_for_this_segment = composited_video_with_text
#                         logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
#                     except Exception as e_text_main:
#                         logger.error(f"video_id: {video_id} - Failed TextClip for seg {i+1}: {e_text_main}. Using original subclip (no subtitle for this segment).", extra=log_extra)
#                         # final_sub_clip_for_this_segment remains the original sub_clip
                
#                 processed_sub_clips_for_concat.append(final_sub_clip_for_this_segment)
#             # --- End regular segment processing loop ---

#             if not processed_sub_clips_for_concat:
#                 error_msg = "No valid sub-clips were generated after processing all segments."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             # --- EXPLICIT AUDIO CONCATENATION (Test 3 Logic integrated as primary) ---
#             logger.info(f"video_id: {video_id} - Attempting explicit audio and video concatenation.", extra=log_extra)
            
#             audio_tracks_to_concat = [sc.audio for sc in processed_sub_clips_for_concat if sc.audio is not None]
#             final_combined_audio_track = None

#             if audio_tracks_to_concat:
#                 logger.info(f"video_id: {video_id} - Found {len(audio_tracks_to_concat)} audio tracks for explicit concatenation.", extra=log_extra)
#                 try:
#                     final_combined_audio_track = concatenate_audioclips(audio_tracks_to_concat)
#                     if final_combined_audio_track:
#                         # Attempt to set FPS if missing, using detected source_audio_fps or a robust default
#                         current_concat_audio_fps = getattr(final_combined_audio_track, 'fps', None)
#                         if not current_concat_audio_fps and source_audio_fps:
#                             logger.info(f"video_id: {video_id} - Setting explicit concat audio FPS to source FPS: {source_audio_fps}", extra=log_extra)
#                             final_combined_audio_track.fps = source_audio_fps
#                         elif not current_concat_audio_fps:
#                             final_combined_audio_track.fps = 44100 # Absolute fallback
#                             logger.warning(f"video_id: {video_id} - Explicit concat audio FPS defaulted to 44100", extra=log_extra)
                        
#                         logger.info(f"video_id: {video_id} - Explicitly concatenated audio. Duration: {final_combined_audio_track.duration:.2f}s, FPS: {final_combined_audio_track.fps}", extra=log_extra)
                        
#                         # Optional: Write this explicit audio track to a debug file
#                         debug_explicit_concat_audio_path = os.path.join(highlights_output_dir, f"DEBUG_EXPLICIT_CONCAT_AUDIO_{video_id}.aac")
#                         try:
#                             final_combined_audio_track.write_audiofile(debug_explicit_concat_audio_path, fps=final_combined_audio_track.fps, codec='aac', logger=None)
#                             logger.info(f"video_id: {video_id} - DEBUG: Successfully wrote EXPLICITLY concatenated audio. PLEASE PLAY: {debug_explicit_concat_audio_path}", extra=log_extra)
#                         except Exception as e_audio_write_explicit:
#                              logger.error(f"video_id: {video_id} - DEBUG: FAILED to write EXPLICITLY concatenated audio: {e_audio_write_explicit}", extra=log_extra)
#                     else: # concatenate_audioclips returned None
#                         logger.error(f"video_id: {video_id} - concatenate_audioclips returned None! No combined audio track.", extra=log_extra)
#                 except Exception as e_concat_audio_explicit:
#                     logger.exception(f"video_id: {video_id} - Error during explicit audio concatenation: {e_concat_audio_explicit}", extra=log_extra)
#             else: # No audio tracks to concat
#                 logger.warning(f"video_id: {video_id} - No audio tracks found in subclips for explicit concatenation. Final clip will be silent.", extra=log_extra)

#             # Concatenate video parts
#             video_parts_only = [sc.without_audio() for sc in processed_sub_clips_for_concat]
#             if not video_parts_only:
#                  error_msg = "No video parts to concatenate after attempting to remove audio."
#                  logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                  async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                  return None
            
#             concatenated_video_track_only = concatenate_videoclips(video_parts_only, method="compose")
#             logger.info(f"video_id: {video_id} - Concatenated video-only track. Duration: {concatenated_video_track_only.duration:.2f}s", extra=log_extra)

#             # Set the (explicitly concatenated) audio to the video-only track
#             if final_combined_audio_track and concatenated_video_track_only:
#                 concatenated_clip_obj = concatenated_video_track_only.set_audio(final_combined_audio_track)
#                 logger.info(f"video_id: {video_id} - Successfully set explicitly concatenated audio to video-only track.", extra=log_extra)
#             elif concatenated_video_track_only:
#                 concatenated_clip_obj = concatenated_video_track_only # No audio to set
#                 logger.warning(f"video_id: {video_id} - Proceeding with video-only concatenated clip as no final_combined_audio_track was available.", extra=log_extra)
#             else: # Should not happen if video_parts_only was not empty
#                 error_msg = "Failed to create final concatenated video track."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                      await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None
#             # --- END EXPLICIT AUDIO CONCATENATION ---

#             # --- Final Write ---
#             has_audio_for_write = concatenated_clip_obj.audio is not None
#             audio_fps_for_write = None
#             if has_audio_for_write:
#                 audio_fps_for_write = getattr(concatenated_clip_obj.audio, 'fps', None)
#                 if not audio_fps_for_write: # If still None after explicit concatenation FPS setting attempts
#                     audio_fps_for_write = source_audio_fps or 44100 # Fallback
#                     logger.warning(f"video_id: {video_id} - Final audio FPS for write_videofile defaulted to {audio_fps_for_write}", extra=log_extra)
#                     if hasattr(concatenated_clip_obj.audio, 'fps'): # Try to set it again
#                         concatenated_clip_obj.audio.fps = audio_fps_for_write
            
#             logger.info(f"video_id: {video_id} - Preparing to write final clip. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)
#             concatenated_clip_obj.write_videofile(
#                 final_clip_path, codec="libx264",
#                 audio_codec="aac" if has_audio_for_write else None,
#                 audio=has_audio_for_write,
#                 temp_audiofile=f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                 remove_temp=False, # Set to False to inspect final temp audio file
#                 threads=os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
#                 fps=main_video_fps_for_output,
#                 audio_fps=audio_fps_for_write if has_audio_for_write else None,
#                 logger='bar',
#                 verbose=True,
#             )
#             logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#             final_clip_path_to_return = final_clip_path
#         # End of 'with VideoFileClip(...)'

#         if final_clip_path_to_return:
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(session_after_write, video_id, highlight_clip_path=final_clip_path_to_return)
#                 await database_service.update_video_status_and_error(session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
#             logger.info(f"video_id: {video_id} - DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             error_msg_no_output = "Highlight generation completed but no output path was produced."
#             logger.error(f"video_id: {video_id} - {error_msg_no_output}", extra=log_extra)
#             async with get_db_session() as error_session_final:
#                 current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
#                 if current_rec and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                      await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
#             return None

#     except Exception as e_main:
#         error_msg_main = f"Highlight generation error: {str(e_main)}"
#         logger.exception(f"video_id: {video_id} - Overall error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session_main:
#             await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
#         return None
#     finally:
#         for clip in processed_sub_clips_for_concat: # Use the list that was actually populated
#             if clip: 
#                 try: clip.close()
#                 except: pass
#         if concatenated_clip_obj:
#             try: concatenated_clip_obj.close()
#             except: pass
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)

# # --- if __name__ == "__main__": block for direct testing ---
# _original_db_service_module_for_test = database_service

# if __name__ == "__main__":
#     import asyncio
#     import subprocess
    
#     DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
#     if not logging.getLogger().hasHandlers():
#         logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
#     else:
#         for handler in logging.getLogger(__name__).handlers or logging.getLogger().handlers:
#              handler.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT))
#         logging.getLogger(__name__).setLevel(logging.DEBUG)

#     async def test_clip_generation_main():
#         logger.info("Running clip_builder_service.py (AUDIO TEST 3 - EXPLICIT AUDIO CONCAT)...")
        
#         test_video_id_main = "clip_builder_audiotest3_001"
        
#         # (Mock Database Service as in the previous complete example, ensuring get_video_record_by_uuid creates the dummy video)
#         # class MockDBSessionMain: async def __aenter__(self): return self; async def __aexit__(self, exc_type, exc, tb): pass
#         class MockDBSessionMain:
#             async def __aenter__(self):
#                 return self
#             async def __aexit__(self, exc_type, exc, tb):
#                 pass
#         class MockDatabaseServiceModuleMain:
#             VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]: yield MockDBSessionMain()
#             async def get_video_record_by_uuid(self, session, video_id):
#                 logger.info(f"MOCK DB: get_video_record_by_uuid for {video_id}")
#                 dummy_video_filename = "sample_clip_builder_test_video_audio_test.mp4"
#                 if not os.path.exists(dummy_video_filename):
#                     logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename}' for test...")
#                     try:
#                         subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=160x120:rate=15", "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100", "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename], check=True)
#                         logger.info(f"MOCK DB: Created dummy: {dummy_video_filename}")
#                     except Exception as e: logger.error(f"MOCK DB: Failed to create dummy: {e}"); return None
#                 class MockRecord: original_video_file_path = os.path.abspath(dummy_video_filename)
#                 return MockRecord()
#             async def update_video_status_and_error(self, s, vid, st, e=None): logger.info(f"MOCK DB: status: {vid}, {st.value}, {e}")
#             async def update_video_asset_paths_record(self, s, vid, **p): logger.info(f"MOCK DB: paths: {vid}, {p}")
#         global database_service
#         database_service = MockDatabaseServiceModuleMain()

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.join(current_dir, "..")
#         test_output_parent_dir = os.path.join(project_root, "direct_test_output_clip_builder")
#         ensure_dir(test_output_parent_dir)
#         test_processing_base = os.path.join(test_output_parent_dir, test_video_id_main)
#         ensure_dir(test_processing_base)
#         logger.info(f"Direct test: Output base path: {test_processing_base}")

#         # Use segments that will actually be processed (not zero duration or out of bounds for a 5s video)
#         segments_to_test = [
#             {"start_time": 0.5, "end_time": 1.5, "text_content": "Audio Test 3 - Seg One"},
#             {"start_time": 2.0, "end_time": 3.0, "text_content": "Audio Test 3 - Seg Two"}
#         ]
#         logger.info(f"AUDIO TEST 3: Calling generate_highlight_clip for {test_video_id_main}")
        
#         final_clip_output_path = None
#         try:
#             final_clip_output_path = await generate_highlight_clip(
#                 video_id=test_video_id_main,
#                 segments_to_include=segments_to_test,
#                 processing_base_path=test_processing_base
#             )
#             if final_clip_output_path and os.path.exists(final_clip_output_path):
#                 logger.info(f"AUDIO TEST 3 (and full run): Final clip path: {final_clip_output_path}")
#                 logger.info(">>>> Check logs for 'AUDIO TEST 3: Successfully wrote EXPLICITLY concatenated audio. PLEASE PLAY: ...'")
#                 logger.info(">>>> Also check the final clip for audio and subtitles (if ImageMagick is working).")
#             else:
#                 logger.error("AUDIO TEST 3 (and full run): Highlight clip generation failed or path not returned.")
#         except Exception as e_test_main:
#             logger.exception(f"AUDIO TEST 3: Exception during main test call for video_id: {test_video_id_main}")
#         finally:
#             database_service = _original_db_service_module_for_test
#             logger.info("Restored original database_service module after direct test.")
#             dummy_video_to_clean = "sample_clip_builder_test_video_audio_test.mp4"
#             if os.path.exists(dummy_video_to_clean): 
#                 try: os.remove(dummy_video_to_clean)
#                 except Exception as e_clean: logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")

#     asyncio.run(test_clip_generation_main())

































































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager
# import subprocess

# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
#     AudioFileClip,
#     concatenate_audioclips
# )

# # Fix for ImageMagick - Add this configuration
# # from moviepy.config import change_settings
# # # Try to configure ImageMagick path - adjust path as needed for your system
# # try:
# #     # Common Windows ImageMagick paths
# #     possible_paths = [
# #         r"C:\Program Files\ImageMagick-7.1.0-Q16\magick.exe",
# #         r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
# #         r"C:\ImageMagick\magick.exe",
# #         r"C:\Program Files (x86)\ImageMagick-7.1.0-Q16\magick.exe"
# #     ]
    
# #     imagemagick_path = None
# #     for path in possible_paths:
# #         if os.path.exists(path):
# #             imagemagick_path = path
# #             break
    
# #     if imagemagick_path:
# #         change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
# #         print(f"ImageMagick configured at: {imagemagick_path}")
# #     else:
# #         print("Warning: ImageMagick not found. Text overlays will be skipped.")
# # except Exception as e:
# #     print(f"Could not configure ImageMagick: {e}")

# MUSIC_ENABLED = False

# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__)
# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}")
#         raise

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]:
#     log_extra = {'video_id': video_id}
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     source_audio_fps: Optional[int] = None

#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             error_msg = "Original video record/path not found in DB for clip generation."
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
#             logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir:
#         logger.exception(f"video_id: {video_id} - Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {str(e_dir)}")
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     final_clip_path_to_return = None
#     processed_sub_clips_for_concat = []
#     concatenated_clip_obj = None

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)

#             default_fontsize = 24
#             default_text_color = 'white'

#             for i, segment_info_loop in enumerate(segments_to_include):
#                 start = segment_info_loop.get("start_time")
#                 end = segment_info_loop.get("end_time")
#                 text_content_loop = segment_info_loop.get("text_content")

#                 # --- Segment Validation ---
#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid segment {i+1} times. Skipping.", extra=log_extra)
#                     continue
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} start ({start:.2f}s) out of bounds ({current_main_duration:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 end = min(end, current_main_duration)
#                 if start == end:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} zero duration ({start:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 # --- End Segment Validation ---

#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
#                 sub_clip = main_video_clip.subclip(start, end)
                
#                 if sub_clip.audio and source_audio_fps is None and sub_clip.audio.fps:
#                     source_audio_fps = sub_clip.audio.fps
#                     logger.info(f"video_id: {video_id} - Inferred source_audio_fps from subclip {i+1}: {source_audio_fps}", extra=log_extra)

#                 final_sub_clip_for_this_segment = sub_clip
                
#                 # Enhanced TextClip creation with better error handling
#                 if text_content_loop and str(text_content_loop).strip():
#                     try:
#                         text_to_render = str(text_content_loop)
                        
#                         # Try different font options if DejaVu-Sans fails
#                         fonts_to_try = ['DejaVu-Sans', 'Arial', 'Helvetica', 'sans-serif', None]
#                         txt_clip = None
                        
#                         for font in fonts_to_try:
#                             try:
#                                 txt_clip = TextClip(
#                                     text_to_render, 
#                                     fontsize=default_fontsize, 
#                                     color=default_text_color, 
#                                     font=font,
#                                     method='caption', 
#                                     size=(sub_clip.w * 0.9, None), 
#                                     bg_color='transparent',
#                                     stroke_color='black', 
#                                     stroke_width=1
#                                 )
#                                 logger.info(f"video_id: {video_id} - TextClip created successfully with font: {font}", extra=log_extra)
#                                 break
#                             except Exception as font_error:
#                                 logger.warning(f"video_id: {video_id} - Font {font} failed: {font_error}", extra=log_extra)
#                                 continue
                        
#                         if txt_clip is not None:
#                             text_h_est = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
#                             y_pos = sub_clip.h - text_h_est - (sub_clip.h * 0.05)
#                             txt_clip = txt_clip.set_pos(('center', y_pos)).set_duration(sub_clip.duration)
                            
#                             video_part = sub_clip.without_audio() if sub_clip.audio else sub_clip
                            
#                             # Fix the mask issue
#                             try:
#                                 composited_video_with_text = CompositeVideoClip([video_part, txt_clip])
#                             except Exception as composite_error:
#                                 logger.warning(f"video_id: {video_id} - CompositeVideoClip failed: {composite_error}. Using simple overlay.", extra=log_extra)
#                                 # Fallback to simpler composition
#                                 composited_video_with_text = CompositeVideoClip([video_part, txt_clip], use_bgclip=False)
                            
#                             if sub_clip.audio:
#                                 final_sub_clip_for_this_segment = composited_video_with_text.set_audio(sub_clip.audio)
#                             else:
#                                 final_sub_clip_for_this_segment = composited_video_with_text
#                             logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
#                         else:
#                             logger.warning(f"video_id: {video_id} - All fonts failed for TextClip. Using segment without text overlay.", extra=log_extra)
                            
#                     except Exception as e_text_main:
#                         logger.error(f"video_id: {video_id} - Failed TextClip for seg {i+1}: {e_text_main}. Using original subclip (no subtitle for this segment).", extra=log_extra)
                
#                 processed_sub_clips_for_concat.append(final_sub_clip_for_this_segment)

#             if not processed_sub_clips_for_concat:
#                 error_msg = "No valid sub-clips were generated after processing all segments."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             # --- EXPLICIT AUDIO CONCATENATION ---
#             logger.info(f"video_id: {video_id} - Attempting explicit audio and video concatenation.", extra=log_extra)
            
#             audio_tracks_to_concat = [sc.audio for sc in processed_sub_clips_for_concat if sc.audio is not None]
#             final_combined_audio_track = None

#             if audio_tracks_to_concat:
#                 logger.info(f"video_id: {video_id} - Found {len(audio_tracks_to_concat)} audio tracks for explicit concatenation.", extra=log_extra)
#                 try:
#                     final_combined_audio_track = concatenate_audioclips(audio_tracks_to_concat)
#                     if final_combined_audio_track:
#                         # Fix FPS setting
#                         if not hasattr(final_combined_audio_track, 'fps') or not final_combined_audio_track.fps:
#                             target_fps = source_audio_fps or 44100
#                             final_combined_audio_track = final_combined_audio_track.set_fps(target_fps)
#                             logger.info(f"video_id: {video_id} - Set explicit concat audio FPS to: {target_fps}", extra=log_extra)
                        
#                         logger.info(f"video_id: {video_id} - Explicitly concatenated audio. Duration: {final_combined_audio_track.duration:.2f}s, FPS: {final_combined_audio_track.fps}", extra=log_extra)
                        
#                         # Write debug audio file
#                         debug_explicit_concat_audio_path = os.path.join(highlights_output_dir, f"DEBUG_EXPLICIT_CONCAT_AUDIO_{video_id}.aac")
#                         try:
#                             final_combined_audio_track.write_audiofile(
#                                 debug_explicit_concat_audio_path, 
#                                 fps=final_combined_audio_track.fps, 
#                                 codec='aac', 
#                                 logger=None,
#                                 verbose=False
#                             )
#                             logger.info(f"video_id: {video_id} - DEBUG: Successfully wrote EXPLICITLY concatenated audio. PLEASE PLAY: {debug_explicit_concat_audio_path}", extra=log_extra)
#                         except Exception as e_audio_write_explicit:
#                             logger.error(f"video_id: {video_id} - DEBUG: FAILED to write EXPLICITLY concatenated audio: {e_audio_write_explicit}", extra=log_extra)
#                     else:
#                         logger.error(f"video_id: {video_id} - concatenate_audioclips returned None! No combined audio track.", extra=log_extra)
#                 except Exception as e_concat_audio_explicit:
#                     logger.exception(f"video_id: {video_id} - Error during explicit audio concatenation: {e_concat_audio_explicit}", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - No audio tracks found in subclips for explicit concatenation. Final clip will be silent.", extra=log_extra)

#             # Concatenate video parts
#             video_parts_only = [sc.without_audio() for sc in processed_sub_clips_for_concat]
#             if not video_parts_only:
#                 error_msg = "No video parts to concatenate after attempting to remove audio."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None
            
#             concatenated_video_track_only = concatenate_videoclips(video_parts_only, method="compose")
#             logger.info(f"video_id: {video_id} - Concatenated video-only track. Duration: {concatenated_video_track_only.duration:.2f}s", extra=log_extra)

#             # Set the audio to the video track
#             if final_combined_audio_track and concatenated_video_track_only:
#                 concatenated_clip_obj = concatenated_video_track_only.set_audio(final_combined_audio_track)
#                 logger.info(f"video_id: {video_id} - Successfully set explicitly concatenated audio to video-only track.", extra=log_extra)
#             elif concatenated_video_track_only:
#                 concatenated_clip_obj = concatenated_video_track_only
#                 logger.warning(f"video_id: {video_id} - Proceeding with video-only concatenated clip as no final_combined_audio_track was available.", extra=log_extra)
#             else:
#                 error_msg = "Failed to create final concatenated video track."
#                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             # --- Final Write with Enhanced Audio Handling ---
#             has_audio_for_write = concatenated_clip_obj.audio is not None
#             audio_fps_for_write = None
            
#             if has_audio_for_write:
#                 audio_fps_for_write = getattr(concatenated_clip_obj.audio, 'fps', None)
#                 if not audio_fps_for_write:
#                     audio_fps_for_write = source_audio_fps or 44100
#                     logger.warning(f"video_id: {video_id} - Final audio FPS for write_videofile defaulted to {audio_fps_for_write}", extra=log_extra)
#                     # Try to set FPS one more time
#                     concatenated_clip_obj.audio = concatenated_clip_obj.audio.set_fps(audio_fps_for_write)
            
#             logger.info(f"video_id: {video_id} - Preparing to write final clip. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)
            
#             # Enhanced write_videofile call
#             write_params = {
#                 "codec": "libx264",
#                 "temp_audiofile": f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                 "remove_temp": False,
#                 "threads": os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
#                 "fps": main_video_fps_for_output,
#                 "logger": 'bar',
#                 "verbose": True,
#                 "audio_codec":'aac',
#             }
            
#             if has_audio_for_write:
#                 write_params.update({
#                     "audio_codec": "aac",
#                     "audio": True,
#                     "audio_fps": audio_fps_for_write,
#                     "audio_bitrate": "128k"  # Explicit audio bitrate
#                 })
#             else:
#                 write_params.update({
#                     "audio": False,
#                     "audio_codec": None
#                 })
            
#             concatenated_clip_obj.write_videofile(final_clip_path, **write_params)
#             logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#             final_clip_path_to_return = final_clip_path

#         if final_clip_path_to_return:
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(session_after_write, video_id, highlight_clip_path=final_clip_path_to_return)
#                 await database_service.update_video_status_and_error(session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
#             logger.info(f"video_id: {video_id} - DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             error_msg_no_output = "Highlight generation completed but no output path was produced."
#             logger.error(f"video_id: {video_id} - {error_msg_no_output}", extra=log_extra)
#             async with get_db_session() as error_session_final:
#                 current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
#                 if current_rec and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                     await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
#             return None

#     except Exception as e_main:
#         error_msg_main = f"Highlight generation error: {str(e_main)}"
#         logger.exception(f"video_id: {video_id} - Overall error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session_main:
#             await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
#         return None
#     finally:
#         for clip in processed_sub_clips_for_concat:
#             if clip: 
#                 try: clip.close()
#                 except: pass
#         if concatenated_clip_obj:
#             try: concatenated_clip_obj.close()
#             except: pass
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)

# # --- if __name__ == "__main__": block for direct testing ---
# _original_db_service_module_for_test = database_service

# if __name__ == "__main__":
#     import asyncio
#     import subprocess
    
#     DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
#     if not logging.getLogger().hasHandlers():
#         logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
#     else:
#         for handler in logging.getLogger(__name__).handlers or logging.getLogger().handlers:
#             handler.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT))
#         logging.getLogger(__name__).setLevel(logging.DEBUG)

#     async def test_clip_generation_main():
#         logger.info("Running clip_builder_service.py (AUDIO TEST 3 - EXPLICIT AUDIO CONCAT - FIXED VERSION)...")
        
#         test_video_id_main = "clip_builder_audiotest3_001"
        
#         class MockDBSessionMain:
#             async def __aenter__(self):
#                 return self
#             async def __aexit__(self, exc_type, exc, tb):
#                 pass
                
#         class MockDatabaseServiceModuleMain:
#             VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]: 
#                 yield MockDBSessionMain()
                
#             async def get_video_record_by_uuid(self, session, video_id):
#                 logger.info(f"MOCK DB: get_video_record_by_uuid for {video_id}")
#                 dummy_video_filename = "sample_clip_builder_test_video_audio_test.mp4"
#                 if not os.path.exists(dummy_video_filename):
#                     logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename}' for test...")
#                     try:
#                         subprocess.run([
#                             "ffmpeg", "-y", 
#                             "-f", "lavfi", "-i", "testsrc=duration=5:size=160x120:rate=15", 
#                             "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100", 
#                             "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", 
#                             dummy_video_filename
#                         ], check=True)
#                         logger.info(f"MOCK DB: Created dummy: {dummy_video_filename}")
#                     except Exception as e: 
#                         logger.error(f"MOCK DB: Failed to create dummy: {e}")
#                         return None
                        
#                 class MockRecord: 
#                     original_video_file_path = os.path.abspath(dummy_video_filename)
#                 return MockRecord()
                
#             async def update_video_status_and_error(self, s, vid, st, e=None): 
#                 logger.info(f"MOCK DB: status: {vid}, {st.value}, {e}")
                
#             async def update_video_asset_paths_record(self, s, vid, **p): 
#                 logger.info(f"MOCK DB: paths: {vid}, {p}")

#         global database_service
#         database_service = MockDatabaseServiceModuleMain()

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.join(current_dir, "..")
#         test_output_parent_dir = os.path.join(project_root, "direct_test_output_clip_builder")
#         ensure_dir(test_output_parent_dir)
#         test_processing_base = os.path.join(test_output_parent_dir, test_video_id_main)
#         ensure_dir(test_processing_base)
#         logger.info(f"Direct test: Output base path: {test_processing_base}")

#         segments_to_test = [
#             {"start_time": 0.5, "end_time": 1.5, "text_content": "Audio Test 3 - Seg One"},
#             {"start_time": 2.0, "end_time": 3.0, "text_content": "Audio Test 3 - Seg Two"}
#         ]
#         logger.info(f"AUDIO TEST 3 FIXED: Calling generate_highlight_clip for {test_video_id_main}")
        
#         final_clip_output_path = None
#         try:
#             final_clip_output_path = await generate_highlight_clip(
#                 video_id=test_video_id_main,
#                 segments_to_include=segments_to_test,
#                 processing_base_path=test_processing_base
#             )
#             if final_clip_output_path and os.path.exists(final_clip_output_path):
#                 logger.info(f"AUDIO TEST 3 FIXED: Final clip path: {final_clip_output_path}")
#                 logger.info(">>>> Check logs for 'Successfully wrote EXPLICITLY concatenated audio. PLEASE PLAY: ...'")
#                 logger.info(">>>> Also check the final clip for audio and subtitles.")
                
#                 # Check file size as indicator of success
#                 file_size = os.path.getsize(final_clip_output_path)
#                 logger.info(f">>>> Final clip file size: {file_size} bytes")
                
#             else:
#                 logger.error("AUDIO TEST 3 FIXED: Highlight clip generation failed or path not returned.")
#         except Exception as e_test_main:
#             logger.exception(f"AUDIO TEST 3 FIXED: Exception during main test call for video_id: {test_video_id_main}")
#         finally:
#             database_service = _original_db_service_module_for_test
#             logger.info("Restored original database_service module after direct test.")
#             dummy_video_to_clean = "sample_clip_builder_test_video_audio_test.mp4"
#             if os.path.exists(dummy_video_to_clean): 
#                 try: 
#                     os.remove(dummy_video_to_clean)
#                 except Exception as e_clean: 
#                     logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")

#     asyncio.run(test_clip_generation_main())

































































# # backend/services/clip_builder_service.py
# import os
# import logging
# import uuid
# from typing import List, Dict, Any, Optional, AsyncGenerator
# from contextlib import asynccontextmanager
# import subprocess
# # import numpy as np # Uncomment if fix_audio_duration_mismatch and AudioArrayClip are used
# from moviepy.editor import (
#     VideoFileClip,
#     TextClip,
#     CompositeVideoClip,
#     concatenate_videoclips,
#     # AudioFileClip, # Uncomment if used directly
#     concatenate_audioclips,
#     # AudioArrayClip # Uncomment if used
# )
# from moviepy.config import change_settings

# # --- ImageMagick Configuration (Corrected) ---
# IMAGEMAGICK_WINDOWS_CANDIDATE_PATHS = [
#     r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe"
#     # Add other common paths if necessary
# ]
# _imagemagick_path_found = None
# for _im_path_candidate in IMAGEMAGICK_WINDOWS_CANDIDATE_PATHS: # Corrected variable name
#     if os.path.exists(_im_path_candidate):
#         _imagemagick_path_found = _im_path_candidate
#         break

# # If you still have issues, you can hardcode the direct path from 'Get-Command magick.exe' here:
# # _imagemagick_path_found = r"YOUR_EXACT_PATH_TO_MAGICK.EXE"

# if os.name == 'nt' and _imagemagick_path_found:
#     try:
#         change_settings({"IMAGEMAGICK_BINARY": _imagemagick_path_found})
#         # Use __name__ for logger
#         logging.getLogger(__name__).info(f"MoviePy IMAGEMAGICK_BINARY explicitly set to: {_imagemagick_path_found}")
#     except Exception as e_im_config_local: # Use unique exception variable name
#         logging.getLogger(__name__).warning(f"Could not programmatically set IMAGEMAGICK_BINARY to {_imagemagick_path_found}: {e_im_config_local}. TextClip might still rely on auto-detect or MoviePy config_defaults.py.")
# elif os.name == 'nt':
#     logging.getLogger(__name__).warning(f"ImageMagick path not found in candidate paths. MoviePy will attempt auto-detect. Ensure ImageMagick 'magick.exe' is in your SYSTEM PATH.")
# else:
#     logging.getLogger(__name__).info("Not Windows, MoviePy will attempt auto-detect for ImageMagick.")
# # --- End ImageMagick Configuration ---


# MUSIC_ENABLED = False

# from . import database_service 
# from .database_service import VideoProcessingStatus, get_db_session 

# logger = logging.getLogger(__name__) # Corrected: Use __name__
# # Add filter to logger instance for this module
# class ServiceVideoIDLogFilterClipBuilder(logging.Filter): # Unique filter name
#     def filter(self, record):
#         if not hasattr(record, 'video_id'):
#             record.video_id = 'CLIP_BUILDER_SVC_CTX' 
#         return True
# logger.addFilter(ServiceVideoIDLogFilterClipBuilder())


# HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"

# def ensure_dir(directory_path):
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#     except OSError as e:
#         logger.error(f"Failed to create directory {directory_path}: {e}", extra={'video_id': 'DIR_ENSURE_ERR'}) # Added extra
#         raise

# async def generate_highlight_clip(
#     video_id: str,
#     segments_to_include: List[Dict[str, Any]],
#     processing_base_path: str,
#     output_filename: Optional[str] = None
# ) -> Optional[str]:
#     log_extra = {'video_id': video_id}
#     logger.info(f"Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

#     if not segments_to_include:
#         logger.warning("No segments provided. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     source_audio_fps: Optional[int] = None

#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             error_msg = "Original video record or path not found in DB for clip generation."
#             logger.error(error_msg, extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
#             logger.error(error_msg, extra=log_extra)
#             await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir:
#         logger.exception(f"Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
#         async with get_db_session() as error_session:
#             await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {str(e_dir)}")
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     processed_sub_clips_for_concat: List[VideoFileClip] = []
#     concatenated_clip_obj: Optional[VideoFileClip] = None

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             if main_video_clip.audio:
#                 source_audio_fps = main_video_clip.audio.fps
#                 logger.info(f"Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s", extra=log_extra)
#             else:
#                 logger.warning("Original video clip has NO audio. Highlights will be silent.", extra=log_extra)

#             default_fontsize = 24
#             default_text_color = 'white'
#             fonts_to_try_main = ['DejaVu-Sans', 'Arial', 'Verdana', 'sans-serif', None] # For main loop

#             for i, segment_info_loop in enumerate(segments_to_include):
#                 start = segment_info_loop.get("start_time")
#                 end = segment_info_loop.get("end_time")
#                 text_content_loop = segment_info_loop.get("text_content")

#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"Invalid segment {i+1} times ({start=}, {end=}). Skipping.", extra=log_extra)
#                     continue
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration:
#                     logger.warning(f"Segment {i+1} start ({start:.2f}s) at/beyond video duration ({current_main_duration:.2f}s). Skipping.", extra=log_extra)
#                     continue
#                 end = min(end, current_main_duration)
#                 if start >= end: # Check again after capping
#                     logger.warning(f"Segment {i+1} zero or negative duration ({start:.2f}s) after capping. Skipping.", extra=log_extra)
#                     continue

#                 logger.info(f"Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
#                 sub_clip: VideoFileClip = main_video_clip.subclip(start, end)
                
#                 if sub_clip.audio and source_audio_fps is None and sub_clip.audio.fps:
#                     source_audio_fps = sub_clip.audio.fps
#                     logger.info(f"Inferred source_audio_fps from subclip {i+1}: {source_audio_fps}", extra=log_extra)

#                 # --- INTEGRATED TEXTCLIP LOGIC FROM YOUR SNIPPET ---
#                 final_sub_clip_for_this_segment = sub_clip 
#                 if text_content_loop and str(text_content_loop).strip():
#                     try:
#                         text_to_render = str(text_content_loop)
#                         current_bg_color_for_text = 'transparent' # The color you are setting
                        
#                         txt_clip_created_main_loop = None # Use unique name
#                         # Using font list from the "original code" which was more robust
#                         for font_choice_main_loop in fonts_to_try_main: 
#                             try:
#                                 txt_clip_created_main_loop = TextClip(
#                                     text_to_render, fontsize=default_fontsize, color=default_text_color, font=font_choice_main_loop, # Use font_choice
#                                     method='caption', size=(sub_clip.w * 0.9, None), bg_color=current_bg_color_for_text,
#                                     stroke_color='black', stroke_width=1
#                                 )
#                                 logger.info(f"TextClip created successfully for segment {i+1} with font: {font_choice_main_loop}", extra=log_extra)
#                                 break 
#                             except Exception as font_error_main_loop:
#                                 logger.warning(f"Font '{font_choice_main_loop}' failed for TextClip on segment {i+1}: {font_error_main_loop}", extra=log_extra)
#                                 txt_clip_created_main_loop = None
                        
#                         if txt_clip_created_main_loop: # Check if any font succeeded
#                             text_h_est_main_loop = txt_clip_created_main_loop.h if hasattr(txt_clip_created_main_loop, 'h') and txt_clip_created_main_loop.h is not None else default_fontsize * 1.5
#                             y_pos_main_loop = sub_clip.h - text_h_est_main_loop - (sub_clip.h * 0.05)
#                             txt_clip_created_main_loop = txt_clip_created_main_loop.set_pos(('center', y_pos_main_loop)).set_duration(sub_clip.duration)
                            
#                             video_part_main_loop = sub_clip.without_audio() if sub_clip.audio else sub_clip
                            
#                             if video_part_main_loop.mask is None and 'transparent' in current_bg_color_for_text.lower():
#                                 # if not video_part_main_loop.ismask : # This was the potentially problematic line from snippet
#                                 video_part_main_loop = video_part_main_loop.add_mask() # Add mask if none and transparent bg
                            
#                             composited_video_with_text_main_loop = CompositeVideoClip([video_part_main_loop, txt_clip_created_main_loop], use_bgclip=True)
                            
#                             if sub_clip.audio:
#                                 final_sub_clip_for_this_segment = composited_video_with_text_main_loop.set_audio(sub_clip.audio)
#                             else:
#                                 final_sub_clip_for_this_segment = composited_video_with_text_main_loop
#                             logger.info(f"Added subtitle to segment {i+1}.", extra=log_extra)
#                         else:
#                             logger.warning(f"All fonts failed for TextClip on segment {i+1}. Using segment without text overlay.", extra=log_extra)
#                             # final_sub_clip_for_this_segment remains original sub_clip
                            
#                     except Exception as e_text_main_loop_outer: # Renamed exception var
#                         logger.error(f"Failed TextClip/Compositing for seg {i+1}: {e_text_main_loop_outer}. Using original subclip.", extra=log_extra)
#                         # final_sub_clip_for_this_segment remains original sub_clip
#                 # --- END INTEGRATED TEXTCLIP LOGIC ---
                
#                 processed_sub_clips_for_concat.append(final_sub_clip_for_this_segment)
#             # --- End regular segment processing loop ---

#             if not processed_sub_clips_for_concat:
#                 # ... (Error handling as before) ...
#                 error_msg = "No valid sub-clips were generated after processing all segments."
#                 logger.error(error_msg, extra=log_extra)
#                 async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None

#             # --- EXPLICIT AUDIO CONCATENATION ---
#             # (This section remains largely the same as your last complete version)
#             logger.info("Starting explicit audio and video concatenation.", extra=log_extra)
#             audio_tracks_to_concat = [sc.audio for sc in processed_sub_clips_for_concat if sc.audio is not None]
#             final_combined_audio_track: Optional[AudioFileClip] = None # Type hint

#             if audio_tracks_to_concat:
#                 logger.info(f"Found {len(audio_tracks_to_concat)} audio tracks for explicit concatenation.", extra=log_extra)
#                 # Log properties of each audio track before concatenation
#                 for idx_debug_audio, audio_track_debug_item in enumerate(audio_tracks_to_concat): 
#                     logger.debug(f"Audio track {idx_debug_audio+1} before concat: Duration={getattr(audio_track_debug_item,'duration', 'N/A'):.2f}, FPS={getattr(audio_track_debug_item, 'fps', 'N/A')}", extra=log_extra)
#                 try:
#                     final_combined_audio_track = concatenate_audioclips(audio_tracks_to_concat, method="compose") # Added method="compose"
#                     if final_combined_audio_track:
#                         current_concat_audio_fps_final = getattr(final_combined_audio_track, 'fps', None) # Renamed var
#                         if not current_concat_audio_fps_final and source_audio_fps:
#                             logger.info(f"Setting explicit concat audio FPS to source FPS: {source_audio_fps}", extra=log_extra)
#                             final_combined_audio_track.fps = source_audio_fps
#                         elif not current_concat_audio_fps_final:
#                             final_combined_audio_track.fps = 44100 
#                             logger.warning("Explicit concat audio FPS defaulted to 44100", extra=log_extra)
                        
#                         logger.info(f"Explicitly concatenated audio. Duration: {final_combined_audio_track.duration:.2f}s, FPS: {final_combined_audio_track.fps}", extra=log_extra)
                        
#                         debug_explicit_concat_audio_path = os.path.join(highlights_output_dir, f"DEBUG_EXPLICIT_CONCAT_AUDIO_{video_id}.aac")
#                         try:
#                             final_combined_audio_track.write_audiofile(
#                                 debug_explicit_concat_audio_path, fps=final_combined_audio_track.fps, 
#                                 codec='aac', logger=None, verbose=False # Added verbose=False
#                             )
#                             logger.info(f"DEBUG: Successfully wrote EXPLICITLY concatenated audio. PLEASE PLAY: {debug_explicit_concat_audio_path}", extra=log_extra)
#                         except Exception as e_audio_write_explicit_final: # Renamed var
#                              logger.error(f"DEBUG: FAILED to write EXPLICITLY concatenated audio: {e_audio_write_explicit_final}", extra=log_extra)
#                     else: 
#                         logger.error("concatenate_audioclips returned None! No combined audio track.", extra=log_extra)
#                 except Exception as e_concat_audio_explicit_final: # Renamed var
#                     logger.exception(f"Error during explicit audio concatenation: {e_concat_audio_explicit_final}", extra=log_extra)
#                     final_combined_audio_track = None # Ensure it's None on failure
#             else: 
#                 logger.warning("No audio tracks found in subclips for explicit concatenation. Final clip will be silent.", extra=log_extra)

#             logger.info("Concatenating video parts (without their original audio).", extra=log_extra) # Moved log up
#             video_parts_only_final = [sc.without_audio() for sc in processed_sub_clips_for_concat] # Renamed var
#             if not video_parts_only_final:
#                  error_msg = "No video parts to concatenate after removing audio."
#                  logger.error(error_msg, extra=log_extra)
#                  async with get_db_session() as error_session:
#                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                  return None
            
#             concatenated_video_track_only_final = concatenate_videoclips(video_parts_only_final, method="compose") # Renamed var
#             logger.info(f"Concatenated video-only track. Duration: {concatenated_video_track_only_final.duration:.2f}s", extra=log_extra)

#             if final_combined_audio_track and concatenated_video_track_only_final:
#                 # Apply fix_audio_duration_mismatch if you have it defined and imported
#                 # For now, directly setting, but fix_audio_duration_mismatch is better
#                 if abs(final_combined_audio_track.duration - concatenated_video_track_only_final.duration) > 0.02:
#                      logger.warning(f"Audio ({final_combined_audio_track.duration:.3f}s) and Video ({concatenated_video_track_only_final.duration:.3f}s) durations differ by >20ms. MoviePy might truncate.", extra=log_extra)
#                 concatenated_clip_obj = concatenated_video_track_only_final.set_audio(final_combined_audio_track)
#                 logger.info("Successfully set explicitly concatenated audio to video-only track.", extra=log_extra)
#             elif concatenated_video_track_only_final:
#                 concatenated_clip_obj = concatenated_video_track_only_final
#                 logger.warning("Proceeding with video-only concatenated clip (no audio or explicit audio concat failed).", extra=log_extra)
#             else:
#                 error_msg = "Failed to create final concatenated video track."
#                 logger.error(error_msg, extra=log_extra)
#                 async with get_db_session() as error_session:
#                      await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
#                 return None
#             # --- END EXPLICIT AUDIO CONCATENATION ---

#             # --- Final Write ---
#             # (This section remains largely the same, ensure all logger calls use `extra=log_extra`)
#             # (And ensure write_params are correctly set up as in your most complete version)
#             if concatenated_clip_obj: # Ensure we have a clip to write
#                 has_audio_for_write_final_check = concatenated_clip_obj.audio is not None # Renamed var
#                 audio_fps_for_write_final_check = None # Renamed var
#                 if has_audio_for_write_final_check:
#                     logger.info(f"Final concatenated clip reports audio. FPS: {getattr(concatenated_clip_obj.audio, 'fps', 'N/A')}", extra=log_extra)
#                     audio_fps_for_write_final_check = getattr(concatenated_clip_obj.audio, 'fps', None) 
#                     if not audio_fps_for_write_final_check:
#                         audio_fps_for_write_final_check = source_audio_fps or 44100
#                         logger.warning(f"Final audio FPS for write_videofile defaulted to {audio_fps_for_write_final_check}", extra=log_extra)
#                         if hasattr(concatenated_clip_obj.audio, 'fps'): 
#                             concatenated_clip_obj.audio.fps = audio_fps_for_write_final_check
#                 else:
#                      logger.error("CRITICAL: Final clip object (concatenated_clip_obj) has NO AUDIO before write_videofile.", extra=log_extra)

#                 logger.info(f"Preparing to write final clip. Has Audio: {has_audio_for_write_final_check}, Audio FPS: {audio_fps_for_write_final_check}, Video FPS: {main_video_fps_for_output}", extra=log_extra)
                
#                 write_params_final = { # Renamed
#                     "codec": "libx264", 
#                     "temp_audiofile": f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                     "remove_temp": False, # Keep for inspection
#                     "threads": os.cpu_count() or 2,
#                     "fps": main_video_fps_for_output,
#                     "logger": 'bar', 
#                     "preset": "medium", 
#                     "ffmpeg_params": ["-crf", "23", "-pix_fmt", "yuv420p"] # Added pix_fmt for compatibility
#                 }
                
#                 if has_audio_for_write_final_check:
#                     write_params_final.update({
#                         "audio": True, "audio_codec": "aac", 
#                         "audio_fps": audio_fps_for_write_final_check, "audio_bitrate": "128k"
#                     })
#                 else:
#                     write_params_final["audio"] = False
                
#                 concatenated_clip_obj.write_videofile(final_clip_path, **write_params_final)
#                 logger.info(f"Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#                 final_clip_path_to_return = final_clip_path
#             else: # concatenated_clip_obj is None
#                 logger.error("Cannot write final clip: concatenated_clip_obj is None (likely error in concat).", extra=log_extra)
#                 # This path should ideally be caught by earlier checks returning None

#         # End of 'with VideoFileClip(...)'

#         if final_clip_path_to_return:
#             # (DB updates for HIGHLIGHT_GENERATED as before)
#             async with get_db_session() as session_after_write:
#                 await database_service.update_video_asset_paths_record(session_after_write, video_id, highlight_clip_path=final_clip_path_to_return)
#                 await database_service.update_video_status_and_error(session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
#             logger.info("DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
#             return final_clip_path_to_return
#         else:
#             # (Error handling and DB update for final failure as before)
#             error_msg_no_output = "Highlight generation process completed but no output path was produced."
#             logger.error(error_msg_no_output, extra=log_extra)
#             async with get_db_session() as error_session_final:
#                 current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
#                 if current_rec and hasattr(current_rec, 'processing_status') and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
#                      await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
#             return None

#     except Exception as e_main_final: # Renamed
#         error_msg_main_final = f"Highlight generation error: {str(e_main_final)}" # Renamed
#         logger.exception("Overall error during highlight clip generation process.", extra=log_extra)
#         async with get_db_session() as error_session_main_final: # Renamed
#             await database_service.update_video_status_and_error(error_session_main_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main_final)
#         return None
#     finally:
#         # (Cleanup as before)
#         for clip_final_cleanup in processed_sub_clips_for_concat: # Renamed
#             if clip_final_cleanup: 
#                 try: clip_final_cleanup.close()
#                 except: pass
#         if concatenated_clip_obj:
#             try: concatenated_clip_obj.close()
#             except: pass
#         logger.debug("Exiting generate_highlight_clip function (finally block).", extra=log_extra)

# # --- if __name__ == "__main__": block for direct testing ---
# _original_db_service_module_for_test_final = database_service # Renamed

# if __name__ == "__main__": # Corrected from "if name == "main":"
#     import asyncio
#     # subprocess is already imported at the top
    
#     # --- Root Logger Configuration for Direct Run (More Robust) ---
#     ROOT_LOGGER_MAIN_RUN_FINAL = logging.getLogger() # Renamed
#     for handler_main_run_final in ROOT_LOGGER_MAIN_RUN_FINAL.handlers[:]: # Renamed
#         ROOT_LOGGER_MAIN_RUN_FINAL.removeHandler(handler_main_run_final)
    
#     DIRECT_RUN_LOG_FORMAT_MAIN_RUN_FINAL = '%(asctime)s - %(levelname)s - [%(video_id)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s' # Renamed
    
#     console_handler_main_run_final = logging.StreamHandler() # Renamed
#     console_handler_main_run_final.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT_MAIN_RUN_FINAL))
#     ROOT_LOGGER_MAIN_RUN_FINAL.addHandler(console_handler_main_run_final)
#     ROOT_LOGGER_MAIN_RUN_FINAL.setLevel(logging.DEBUG)

#     class VideoIDLogFilterForDirectRunFinal(logging.Filter): # Renamed
#         def filter(self, record):
#             if not hasattr(record, 'video_id'): 
#                 record.video_id = 'DIRECT_MAIN_RUN_CTX_FINAL' # Renamed 
#             return True
#     console_handler_main_run_final.addFilter(VideoIDLogFilterForDirectRunFinal())
#     logger.setLevel(logging.DEBUG) # Ensure this module's logger is also DEBUG
#     # --- End Root Logger Configuration ---


#     async def test_clip_generation_main_final(): # Renamed
#         global database_service 
#         test_setup_log_extra_final = {'video_id': 'MAIN_TEST_RUN_SETUP_FINAL_V2'} # Renamed
#         logger.info("Running clip_builder_service.py (Main Logic with Snippet Fix)...", extra=test_setup_log_extra_final)
        
#         test_video_id_for_run_final_test = "clip_builder_main_logic_test_001" # Renamed
#         current_test_log_extra_final_test = {'video_id': test_video_id_for_run_final_test} # Renamed

#         # (Mock Database Service as before, ensure variable names are unique if needed)
#         class MockDBSessionMainFinal: # Renamed
#             async def __aenter__(self): return self
#             async def __aexit__(self, exc_type, exc, tb): pass
        
#         class MockDatabaseServiceModuleMainFinal: # Renamed
#             VideoProcessingStatus = _original_db_service_module_for_test_final.VideoProcessingStatus
#             @asynccontextmanager
#             async def get_db_session(self) -> AsyncGenerator[MockDBSessionMainFinal, None]: 
#                 logger.debug("MOCK DB: get_db_session called", extra={'video_id': 'MOCK_DB_ASYNC_CTX_FINAL'})
#                 yield MockDBSessionMainFinal()

#             async def get_video_record_by_uuid(self, session, video_id_param_for_mock_final): # Renamed
#                 mock_call_log_extra_local_final = {'video_id': video_id_param_for_mock_final} # Renamed
#                 logger.info(f"MOCK DB: get_video_record_by_uuid for {video_id_param_for_mock_final}", extra=mock_call_log_extra_local_final)
#                 dummy_video_filename_for_mock_final = "sample_clip_builder_test_video.mp4" 
#                 if not os.path.exists(dummy_video_filename_for_mock_final):
#                     # ... (Dummy video creation logic as before, ensure variables are uniquely named if needed) ...
#                     logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename_for_mock_final}' for test...", extra=mock_call_log_extra_local_final)
#                     try:
#                         subprocess.run([
#                             "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=24",
#                             "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100",
#                             "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename_for_mock_final
#                         ], check=True, capture_output=True, text=True)
#                         logger.info(f"MOCK DB: Created dummy: {dummy_video_filename_for_mock_final}", extra=mock_call_log_extra_local_final)
#                     except Exception as e_dummy_vid_mock_final: 
#                         logger.error(f"MOCK DB: Error creating dummy video: {e_dummy_vid_mock_final}", extra=mock_call_log_extra_local_final)
#                         return None
                
#                 class MockVideoRecordForTestFinal: pass # Renamed
#                 record_mock_final = MockVideoRecordForTestFinal() # Renamed
#                 record_mock_final.original_video_file_path = os.path.abspath(dummy_video_filename_for_mock_final)
#                 record_mock_final.processing_status = VideoProcessingStatus.EMBEDDINGS_GENERATED 
#                 return record_mock_final

#             async def update_video_status_and_error(self, s, vid_param_final, status_param_final, err_param_final=None): # Renamed
#                 logger.info(f"MOCK DB: status update: vid={vid_param_final}, status={status_param_final.value if hasattr(status_param_final,'value') else status_param_final}, error='{err_param_final}'", extra={'video_id': vid_param_final})
#             async def update_video_asset_paths_record(self, s, vid_param_final, **paths_param_final): # Renamed
#                 logger.info(f"MOCK DB: asset paths update: vid={vid_param_final}, paths={paths_param_final}", extra={'video_id': vid_param_final})
        
#         database_service = MockDatabaseServiceModuleMainFinal() # Apply monkey-patch

#         # (Test directory setup as before, ensure variables are uniquely named if needed)
#         current_file_dir_for_test_final = os.path.dirname(os.path.abspath(__file__)) 
#         project_root_approx_for_test_final = os.path.join(current_file_dir_for_test_final, "..", "..")  
#         test_output_parent_dir_for_test_final = os.path.join(project_root_approx_for_test_final, "direct_test_output_clip_builder_v5_snippet_integrated") 
#         ensure_dir(test_output_parent_dir_for_test_final)
        
#         test_processing_base_path_for_run_final_test = os.path.join(test_output_parent_dir_for_test_final, test_video_id_for_run_final_test) 
#         ensure_dir(test_processing_base_path_for_run_final_test)
#         logger.info(f"Direct test: Output base path: {test_processing_base_path_for_run_final_test}", extra=current_test_log_extra_final_test)

#         segments_to_test_run_final_test = [ 
#             {"start_time": 0.5, "end_time": 2.0, "text_content": "Final Test - Segment Alpha"},
#             {"start_time": 2.5, "end_time": 4.0, "text_content": "Final Test - Segment Beta"}
#         ]
#         logger.info(f"Test Run: Calling generate_highlight_clip for {test_video_id_for_run_final_test}", extra=current_test_log_extra_final_test)
        
#         final_clip_output_path_for_run_result_final = None # Renamed
#         try:
#             final_clip_output_path_for_run_result_final = await generate_highlight_clip(
#                 video_id=test_video_id_for_run_final_test,
#                 segments_to_include=segments_to_test_run_final_test,
#                 processing_base_path=test_processing_base_path_for_run_final_test
#             )
#             if final_clip_output_path_for_run_result_final and os.path.exists(final_clip_output_path_for_run_result_final):
#                 logger.info(f"Test Run: Final clip path: {final_clip_output_path_for_run_result_final}", extra=current_test_log_extra_final_test)
#                 logger.info(">>>> Check logs for 'DEBUG: Successfully wrote EXPLICITLY concatenated audio...' (if audio was present)", extra=current_test_log_extra_final_test)
#                 logger.info(">>>> Also check the final clip for audio and subtitles.", extra=current_test_log_extra_final_test)
#                 file_size = os.path.getsize(final_clip_output_path_for_run_result_final)
#                 logger.info(f">>>> Final clip file size: {file_size} bytes (non-zero size is a good sign).")

#             else:
#                 logger.error(f"Test Run: Highlight clip generation failed or path not returned. Result: {final_clip_output_path_for_run_result_final}", extra=current_test_log_extra_final_test)
#         except Exception as e_test_main_run_final_test: # Renamed
#             logger.exception(f"Test Run: Exception during main test call for video_id: {test_video_id_for_run_final_test}", extra=current_test_log_extra_final_test)
#         finally:
#             database_service = _original_db_service_module_for_test_final 
#             logger.info("Restored original database_service module after direct test.", extra={'video_id': 'SYSTEM_CLEANUP_MAIN_RUN_FINAL'})
            
#             dummy_video_to_clean_for_run_final_test = "sample_clip_builder_test_video.mp4" 
#             if os.path.exists(dummy_video_to_clean_for_run_final_test): 
#                 try: os.remove(dummy_video_to_clean_for_run_final_test)
#                 except Exception as e_clean_run_final_test: logger.warning(f"Could not clean up {dummy_video_to_clean_for_run_final_test}: {e_clean_run_final_test}", extra={'video_id': 'SYSTEM_CLEANUP_MAIN_RUN_FINAL'})

#     asyncio.run(test_clip_generation_main_final())



















































# backend/services/clip_builder_service.py
import os
import logging
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import subprocess

from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    concatenate_audioclips, # For explicit audio concatenation
    AudioFileClip # Potentially for loading generated music if MUSIC_ENABLED=True
)
from moviepy.config import change_settings # For ImageMagick path if needed
IMAGEMAGICK_WINDOWS_EXE_PATH = r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe" # !!! UPDATE THIS PATH !!!

_logger_for_im_config = logging.getLogger(f"{__name__}.config") # Separate logger for this config step
try:
    if os.name == 'nt': # Only attempt for Windows
        if os.path.exists(IMAGEMAGICK_WINDOWS_EXE_PATH):
            change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_WINDOWS_EXE_PATH})
            _logger_for_im_config.info(f"MoviePy IMAGEMAGICK_BINARY explicitly set to: {IMAGEMAGICK_WINDOWS_EXE_PATH}")
        else:
            _logger_for_im_config.warning(
                f"Specified ImageMagick path not found: {IMAGEMAGICK_WINDOWS_EXE_PATH}. "
                "TextClip will rely on auto-detect or conf.py. "
                "Ensure ImageMagick 'magick.exe' is in your SYSTEM PATH if this explicit path is wrong."
            )
    # For non-Windows, MoviePy's auto-detect from PATH is usually sufficient if ImageMagick is installed.
except Exception as e_im_config:
    _logger_for_im_config.warning(f"Could not programmatically set IMAGEMAGICK_BINARY: {e_im_config}.")
# --- End ImageMagick Configuration ---


MUSIC_ENABLED = False

# --- Import Database Service ---
# This 'database_service' will be the one potentially patched in __main__ for testing
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
    logger.info(f"video_id: {video_id} - Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

    if not segments_to_include:
        logger.warning(f"video_id: {video_id} - No segments provided. Aborting.", extra=log_extra)
        return None

    original_video_file_path: Optional[str] = None
    main_video_fps_for_output: float = 24.0
    source_audio_fps: Optional[int] = None

    async with get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, video_id)
        if not video_record or not video_record.original_video_file_path:
            error_msg = "Original video record or path not found in DB for clip generation."
            logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            return None
        if not os.path.exists(video_record.original_video_file_path):
            error_msg = f"Original video file not found at DB path: {video_record.original_video_file_path}"
            logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            return None
        original_video_file_path = video_record.original_video_file_path
    
    logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

    highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
    try:
        ensure_dir(highlights_output_dir)
    except OSError as e_dir:
        logger.exception(f"video_id: {video_id} - Failed to create highlights output directory: {highlights_output_dir}", extra=log_extra)
        async with get_db_session() as error_session:
            await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Failed to create output dir: {str(e_dir)}")
        return None
    
    if not output_filename:
        output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
    final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
    final_clip_path_to_return = None
    processed_sub_clips_for_concat = []
    concatenated_clip_obj = None

    try:
        with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
            main_video_fps_for_output = main_video_clip.fps or 24.0
            logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
            if main_video_clip.audio:
                source_audio_fps = main_video_clip.audio.fps
                logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {source_audio_fps}, Duration: {main_video_clip.audio.duration:.2f}s, Channels: {getattr(main_video_clip.audio, 'nchannels', 'N/A')}", extra=log_extra)
            else:
                logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)

            default_fontsize = 24
            default_text_color = 'white'

            for i, segment_info_loop in enumerate(segments_to_include):
                start = segment_info_loop.get("start_time")
                end = segment_info_loop.get("end_time")
                text_content_loop = segment_info_loop.get("text_content")

                # --- Segment Validation ---
                if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
                    logger.warning(f"video_id: {video_id} - Invalid segment {i+1} times ({start=}, {end=}). Skipping.", extra=log_extra)
                    continue
                current_main_duration = main_video_clip.duration
                if start >= current_main_duration:
                    logger.warning(f"video_id: {video_id} - Segment {i+1} start ({start:.2f}s) at/beyond video duration ({current_main_duration:.2f}s). Skipping.", extra=log_extra)
                    continue
                end = min(end, current_main_duration)
                if start == end:
                    logger.warning(f"video_id: {video_id} - Segment {i+1} zero duration ({start:.2f}s) after capping. Skipping.", extra=log_extra)
                    continue
                # --- End Segment Validation ---

                logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
                sub_clip = main_video_clip.subclip(start, end)
                
                if sub_clip.audio and source_audio_fps is None and sub_clip.audio.fps:
                    source_audio_fps = sub_clip.audio.fps
                    logger.info(f"video_id: {video_id} - Inferred source_audio_fps from subclip {i+1}: {source_audio_fps}", extra=log_extra)

                final_sub_clip_for_this_segment = sub_clip
                if text_content_loop and str(text_content_loop).strip():
                    try:
                        text_to_render = str(text_content_loop)
                        current_bg_color_for_text = 'transparent' # Define the bg_color to check against
                        
                        txt_clip = TextClip(
                            text_to_render, fontsize=default_fontsize, color=default_text_color, font="DejaVu-Sans",
                            method='caption', size=(sub_clip.w * 0.9, None), bg_color=current_bg_color_for_text,
                            stroke_color='black', stroke_width=1
                        )
                        text_h_est = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
                        y_pos = sub_clip.h - text_h_est - (sub_clip.h * 0.05)
                        txt_clip = txt_clip.set_pos(('center', y_pos)).set_duration(sub_clip.duration)
                        
                        video_part = sub_clip.without_audio() if sub_clip.audio else sub_clip
                        
                        # Check against the variable used for bg_color
                        if video_part.mask is None and 'transparent' in current_bg_color_for_text.lower():
                             if not video_part.ismask : video_part = video_part.add_mask()
                        
                        composited_video_with_text = CompositeVideoClip([video_part, txt_clip], use_bgclip=True)
                        
                        if sub_clip.audio:
                            final_sub_clip_for_this_segment = composited_video_with_text.set_audio(sub_clip.audio)
                        else:
                            final_sub_clip_for_this_segment = composited_video_with_text
                        logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
                    except Exception as e_text_main:
                        logger.error(f"video_id: {video_id} - Failed TextClip/Compositing for seg {i+1}: {e_text_main}. Using original subclip (no subtitle).", extra=log_extra)
                
                processed_sub_clips_for_concat.append(final_sub_clip_for_this_segment)
            # --- End regular segment processing loop ---

            if not processed_sub_clips_for_concat:
                error_msg = "No valid sub-clips were generated after processing all segments."
                logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                async with get_db_session() as error_session:
                    await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return None

            # --- EXPLICIT AUDIO CONCATENATION ---
            logger.info(f"video_id: {video_id} - Attempting explicit audio concatenation.", extra=log_extra)
            audio_tracks_to_concat = [sc.audio for sc in processed_sub_clips_for_concat if sc.audio is not None]
            final_combined_audio_track = None

            if audio_tracks_to_concat:
                logger.info(f"video_id: {video_id} - Found {len(audio_tracks_to_concat)} audio tracks for explicit concatenation.", extra=log_extra)
                for idx, audio_track_debug in enumerate(audio_tracks_to_concat):
                    logger.debug(f"video_id: {video_id} - Audio track {idx+1} before concat: Duration={getattr(audio_track_debug, 'duration', 'N/A'):.2f}, FPS={getattr(audio_track_debug, 'fps', 'N/A')}", extra=log_extra)
                try:
                    final_combined_audio_track = concatenate_audioclips(audio_tracks_to_concat)
                    if final_combined_audio_track:
                        current_concat_audio_fps = getattr(final_combined_audio_track, 'fps', None)
                        if not current_concat_audio_fps and source_audio_fps:
                            logger.info(f"video_id: {video_id} - Setting explicit concat audio FPS to source FPS: {source_audio_fps}", extra=log_extra)
                            final_combined_audio_track.fps = source_audio_fps
                        elif not current_concat_audio_fps:
                            final_combined_audio_track.fps = 44100 
                            logger.warning(f"video_id: {video_id} - Explicit concat audio FPS defaulted to 44100", extra=log_extra)
                        
                        logger.info(f"video_id: {video_id} - Explicitly concatenated audio. Duration: {final_combined_audio_track.duration:.2f}s, FPS: {final_combined_audio_track.fps}", extra=log_extra)
                        
                        debug_explicit_concat_audio_path = os.path.join(highlights_output_dir, f"DEBUG_EXPLICIT_CONCAT_AUDIO_{video_id}.aac")
                        try:
                            final_combined_audio_track.write_audiofile(
                                debug_explicit_concat_audio_path, 
                                fps=final_combined_audio_track.fps, 
                                codec='aac', logger=None, verbose=False
                            )
                            logger.info(f"video_id: {video_id} - DEBUG: Successfully wrote EXPLICITLY concatenated audio. PLEASE PLAY: {debug_explicit_concat_audio_path}", extra=log_extra)
                        except Exception as e_audio_write_explicit:
                             logger.error(f"video_id: {video_id} - DEBUG: FAILED to write EXPLICITLY concatenated audio: {e_audio_write_explicit}", extra=log_extra)
                    else:
                        logger.error(f"video_id: {video_id} - concatenate_audioclips returned None! No combined audio track.", extra=log_extra)
                except Exception as e_concat_audio_explicit:
                    logger.exception(f"video_id: {video_id} - Error during explicit audio concatenation: {e_concat_audio_explicit}", extra=log_extra)
            else:
                logger.warning(f"video_id: {video_id} - No audio tracks found in subclips for explicit concatenation. Final clip will be silent.", extra=log_extra)

            logger.info(f"video_id: {video_id} - Concatenating video parts (without their original audio).", extra=log_extra)
            video_parts_only = [sc.without_audio() for sc in processed_sub_clips_for_concat]
            if not video_parts_only:
                 error_msg = "No video parts to concatenate after removing audio."
                 logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                 async with get_db_session() as error_session:
                    await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                 return None
            
            concatenated_video_track_only = concatenate_videoclips(video_parts_only, method="compose")
            logger.info(f"video_id: {video_id} - Concatenated video-only track. Duration: {concatenated_video_track_only.duration:.2f}s", extra=log_extra)

            if final_combined_audio_track and concatenated_video_track_only:
                concatenated_clip_obj = concatenated_video_track_only.set_audio(final_combined_audio_track)
                logger.info(f"video_id: {video_id} - Successfully set explicitly concatenated audio to video-only track.", extra=log_extra)
            elif concatenated_video_track_only:
                concatenated_clip_obj = concatenated_video_track_only
                logger.warning(f"video_id: {video_id} - Proceeding with video-only concatenated clip.", extra=log_extra)
            else:
                error_msg = "Failed to create final concatenated video track."
                logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                async with get_db_session() as error_session:
                     await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return None
            # --- END EXPLICIT AUDIO CONCATENATION LOGIC ---

            has_audio_for_write = concatenated_clip_obj.audio is not None
            audio_fps_for_write = None
            if has_audio_for_write:
                logger.info(f"video_id: {video_id} - Final concatenated clip reports audio. FPS: {getattr(concatenated_clip_obj.audio, 'fps', 'N/A')}", extra=log_extra)
                audio_fps_for_write = getattr(concatenated_clip_obj.audio, 'fps', None) 
                if not audio_fps_for_write:
                    audio_fps_for_write = source_audio_fps or 44100
                    logger.warning(f"video_id: {video_id} - Final audio FPS for write_videofile defaulted to {audio_fps_for_write}", extra=log_extra)
                    if hasattr(concatenated_clip_obj.audio, 'fps'):
                        concatenated_clip_obj.audio.fps = audio_fps_for_write
            else:
                 logger.error(f"video_id: {video_id} - CRITICAL: Final clip object has NO AUDIO before write_videofile.", extra=log_extra)

            logger.info(f"video_id: {video_id} - Preparing to write final clip. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)
            
            write_params = {
                "codec": "libx264",
                "temp_audiofile": f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
                "remove_temp": False, # Keep False to inspect final temp audio if issues persist
                "threads": os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
                "fps": main_video_fps_for_output,
                "logger": 'bar',
                "preset": "medium", 
                "ffmpeg_params": ["-crf", "23"] 
            }
            
            if has_audio_for_write:
                write_params.update({
                    "audio": True, "audio_codec": "aac",
                    "audio_fps": audio_fps_for_write, "audio_bitrate": "128k"
                })
            else:
                write_params["audio"] = False
            
            concatenated_clip_obj.write_videofile(final_clip_path, **write_params)
            logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
            final_clip_path_to_return = final_clip_path
        
        if final_clip_path_to_return:
            async with get_db_session() as session_after_write:
                await database_service.update_video_asset_paths_record(session_after_write, video_id, highlight_clip_path=final_clip_path_to_return)
                await database_service.update_video_status_and_error(session_after_write, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
            logger.info(f"video_id: {video_id} - DB updated with highlight path and status HIGHLIGHT_GENERATED.", extra=log_extra)
            return final_clip_path_to_return
        else:
            error_msg_no_output = "Highlight generation completed but no output path was produced."
            logger.error(f"video_id: {video_id} - {error_msg_no_output}", extra=log_extra)
            async with get_db_session() as error_session_final:
                current_rec = await database_service.get_video_record_by_uuid(error_session_final, video_id)
                if current_rec and current_rec.processing_status != VideoProcessingStatus.PROCESSING_FAILED:
                     await database_service.update_video_status_and_error(error_session_final, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_no_output)
            return None

    except Exception as e_main:
        error_msg_main = f"Highlight generation error: {str(e_main)}"
        logger.exception(f"video_id: {video_id} - Overall error during highlight clip generation process.", extra=log_extra)
        async with get_db_session() as error_session_main:
            await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
        return None
    finally:
        for clip in processed_sub_clips_for_concat:
            if clip: 
                try: clip.close()
                except: pass
        if concatenated_clip_obj:
            try: concatenated_clip_obj.close()
            except: pass
        logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip function (finally block).", extra=log_extra)

# --- if __name__ == "__main__": block for direct testing ---
_original_db_service_module_for_test = database_service # Save before potential patching

if __name__ == "__main__":
    import asyncio
    import subprocess
    
    DIRECT_RUN_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format=DIRECT_RUN_LOG_FORMAT)
    else:
        for handler in logging.getLogger(__name__).handlers or logging.getLogger().handlers:
             handler.setFormatter(logging.Formatter(DIRECT_RUN_LOG_FORMAT))
        # Ensure module's own logger is also set to DEBUG for direct run
        logging.getLogger(__name__).setLevel(logging.DEBUG) 


    async def test_clip_generation_main():
        logger.info("Running clip_builder_service.py (Direct Test with Explicit Audio Concat)...")
        
        test_video_id_main = "clip_builder_direct_test_004" # Use a distinct ID
        
        class MockDBSessionMain:
            async def __aenter__(self): return self
            async def __aexit__(self, exc_type, exc, tb): pass
        
        class MockDatabaseServiceModuleMain:
            VideoProcessingStatus = _original_db_service_module_for_test.VideoProcessingStatus
            @asynccontextmanager
            async def get_db_session(self) -> AsyncGenerator[MockDBSessionMain, None]:
                logger.debug("MOCK DB (clip_builder_test): get_db_session called")
                yield MockDBSessionMain()
            async def get_video_record_by_uuid(self, session, video_id):
                logger.info(f"MOCK DB: get_video_record_by_uuid for video_id: {video_id}")
                dummy_video_filename = "sample_clip_builder_test_video_audio_test.mp4"
                if not os.path.exists(dummy_video_filename):
                    logger.info(f"MOCK DB: Creating dummy video '{dummy_video_filename}' for test...")
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=160x120:rate=24", # Rate matches default main_video_fps
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100",
                            "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", dummy_video_filename
                        ], check=True, capture_output=True, text=True)
                        logger.info(f"MOCK DB: Created dummy: {dummy_video_filename}")
                    except Exception as e: 
                        logger.error(f"MOCK DB: Failed to create dummy: {e}")
                        return None
                class MockRecord: original_video_file_path = os.path.abspath(dummy_video_filename)
                return MockRecord()
            async def update_video_status_and_error(self, s, vid, st, e=None): logger.info(f"MOCK DB: video_id={vid}, status={st.value}, error='{e}'")
            async def update_video_asset_paths_record(self, s, vid, **p): logger.info(f"MOCK DB: video_id={vid}, paths_updated_with: {p}")
        
        global database_service
        database_service = MockDatabaseServiceModuleMain()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..")
        test_output_parent_dir = os.path.join(project_root, "direct_test_output_clip_builder")
        ensure_dir(test_output_parent_dir)
        test_processing_base = os.path.join(test_output_parent_dir, test_video_id_main)
        ensure_dir(test_processing_base)
        logger.info(f"Direct test: Output base path: {test_processing_base}")

        segments_to_test = [
            {"start_time": 0.5, "end_time": 1.5, "text_content": "Test Clip - Part 1"}, # 1s
            {"start_time": 2.0, "end_time": 3.5, "text_content": "Test Clip - Part 2"}  # 1.5s
            # Total expected audio duration ~2.5s
        ]
        logger.info(f"MAIN TEST: Calling generate_highlight_clip for {test_video_id_main}")
        
        final_clip_output_path = None
        try:
            final_clip_output_path = await generate_highlight_clip(
                video_id=test_video_id_main,
                segments_to_include=segments_to_test,
                processing_base_path=test_processing_base
            )
            if final_clip_output_path and os.path.exists(final_clip_output_path):
                logger.info(f"MAIN TEST: Final clip path: {final_clip_output_path}")
                logger.info(">>>> Check logs for 'DEBUG: Successfully wrote EXPLICITLY concatenated audio... PLEASE PLAY THIS FILE.'")
                logger.info(f">>>> Then, PLEASE MANUALLY PLAY THE FINAL MP4: {final_clip_output_path} to check audio and subtitles (if ImageMagick is working).")
                file_size = os.path.getsize(final_clip_output_path)
                logger.info(f">>>> Final clip file size: {file_size} bytes (non-zero size is a good sign).")
            else:
                logger.error("MAIN TEST: Highlight clip generation failed or path not returned.")
        except Exception as e_test_main:
            logger.exception(f"MAIN TEST: Exception during main test call for video_id: {test_video_id_main}")
        finally:
            database_service = _original_db_service_module_for_test
            logger.info("Restored original database_service module after direct test.")
            dummy_video_to_clean = "sample_clip_builder_test_video_audio_test.mp4"
            if os.path.exists(dummy_video_to_clean): 
                try: os.remove(dummy_video_to_clean)
                except Exception as e_clean: logger.warning(f"Could not clean up {dummy_video_to_clean}: {e_clean}")

    asyncio.run(test_clip_generation_main())







