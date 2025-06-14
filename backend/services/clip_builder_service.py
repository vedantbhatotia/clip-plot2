
# # backend/services/clip_builder_service.py
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
#     concatenate_audioclips, # <<< IMPORT THIS
#     AudioFileClip # For potentially loading audio if needed, or for the result of .audio
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
#     logger.info(f"video_id: {video_id} - Input segments for clip builder: {json.dumps(segments_to_include, indent=2)}", extra=log_extra)
#     logger.info(f"video_id: {video_id} - Starting highlight clip generation (MUSIC_ENABLED={MUSIC_ENABLED}) with {len(segments_to_include)} segments.", extra=log_extra)

#     if not segments_to_include:
#         logger.warning(f"video_id: {video_id} - No segments provided. Aborting.", extra=log_extra)
#         return None

#     original_video_file_path: Optional[str] = None
#     main_video_fps_for_output: float = 24.0
#     # source_audio_fps will be determined from the main clip if audio exists

#     async with get_db_session() as session:
#         video_record = await database_service.get_video_record_by_uuid(session, video_id)
#         if not video_record or not video_record.original_video_file_path:
#             # ... (error handling) ...
#             return None
#         if not os.path.exists(video_record.original_video_file_path):
#             # ... (error handling) ...
#             return None
#         original_video_file_path = video_record.original_video_file_path
    
#     logger.info(f"video_id: {video_id} - Using original video path: {original_video_file_path}", extra=log_extra)

#     highlights_output_dir = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
#     try:
#         ensure_dir(highlights_output_dir)
#     except OSError as e_dir: # ... (error handling) ...
#         return None
    
#     if not output_filename:
#         output_filename = f"highlight_{video_id}_{str(uuid.uuid4())[:6]}.mp4"
#     final_clip_path = os.path.join(highlights_output_dir, output_filename)
    
#     final_clip_path_to_return = None
    
#     # Lists to hold parts
#     video_sub_clips_for_concat = [] # Will hold VideoClips, possibly without audio initially
#     audio_sub_clips_for_concat = [] # Will hold AudioClips

#     try:
#         with VideoFileClip(original_video_file_path, audio=True) as main_video_clip:
#             main_video_fps_for_output = main_video_clip.fps or 24.0
#             logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration:.2f}s, Video FPS: {main_video_fps_for_output:.2f}", extra=log_extra)
            
#             original_video_has_audio = main_video_clip.audio is not None
#             if original_video_has_audio:
#                 logger.info(f"video_id: {video_id} - Original video audio detected. FPS: {main_video_clip.audio.fps}, Duration: {main_video_clip.audio.duration:.2f}s", extra=log_extra)
#             else:
#                 logger.warning(f"video_id: {video_id} - Original video clip has NO audio. Highlights will be silent.", extra=log_extra)

#             default_fontsize = 24
#             default_text_color = 'white'

#             for i, segment_info_loop in enumerate(segments_to_include):
#                 start = segment_info_loop.get("start_time")
#                 end = segment_info_loop.get("end_time")
#                 text_content_loop = segment_info_loop.get("text_content")

#                 # --- Segment Validation (looks good) ---
#                 if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start >= end:
#                     logger.warning(f"video_id: {video_id} - Invalid segment {i+1} times ({start=}, {end=}). Skipping.", extra=log_extra); continue
#                 current_main_duration = main_video_clip.duration
#                 if start >= current_main_duration:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} start ({start:.2f}s) at/beyond video duration ({current_main_duration:.2f}s). Skipping.", extra=log_extra); continue
#                 end = min(end, current_main_duration) # Cap end time
#                 if start == end:
#                     logger.warning(f"video_id: {video_id} - Segment {i+1} zero duration ({start:.2f}s) after capping. Skipping.", extra=log_extra); continue
#                 # --- End Segment Validation ---

#                 logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start:.2f}s - {end:.2f}s", extra=log_extra)
                
#                 # Create the raw subclip (video + audio)
#                 raw_sub_clip = main_video_clip.subclip(start, end) # Using .subclip as per your MoviePy version
                
#                 # Store its audio component if it exists
#                 if raw_sub_clip.audio:
#                     audio_sub_clips_for_concat.append(raw_sub_clip.audio)
#                     logger.info(f"video_id: {video_id} - Extracted audio for segment {i+1}. Duration: {raw_sub_clip.audio.duration:.2f}s", extra=log_extra)
#                 else:
#                     logger.warning(f"video_id: {video_id} - Subclip {i+1} has no audio component to extract.", extra=log_extra)

#                 # Prepare the video part (potentially with subtitles, definitely without its original audio for now)
#                 video_part_of_subclip = raw_sub_clip.without_audio() if original_video_has_audio else raw_sub_clip

#                 if text_content_loop and str(text_content_loop).strip():
#                     try:
#                         txt_clip = TextClip(
#                             txt=str(text_content_loop), fontsize=default_fontsize, color=default_text_color, font="Arial", # Ensure Arial or FreeSans (for Docker)
#                             method='caption', size=(video_part_of_subclip.w * 0.9, None), bg_color='transparent',
#                             stroke_color='black', stroke_width=1
#                         )
#                         text_h_est = txt_clip.h if hasattr(txt_clip, 'h') and txt_clip.h is not None else default_fontsize * 1.5
#                         y_pos = video_part_of_subclip.h - text_h_est - (video_part_of_subclip.h * 0.05) # 5% margin from bottom
#                         txt_clip = txt_clip.set_pos(('center', y_pos)).set_duration(video_part_of_subclip.duration)
                        
#                         video_part_of_subclip = CompositeVideoClip([video_part_of_subclip, txt_clip], use_bgclip=True if video_part_of_subclip.mask is None else False)
#                         logger.info(f"video_id: {video_id} - Added subtitle to video part of segment {i+1}.", extra=log_extra)
#                     except Exception as e_text_main:
#                         logger.error(f"video_id: {video_id} - Failed TextClip/Compositing for seg {i+1}: {e_text_main}. Using video part without subtitle.", extra=log_extra)
                
#                 video_sub_clips_for_concat.append(video_part_of_subclip)
#             # --- End segment processing loop ---

#             if not video_sub_clips_for_concat: # Renamed list
#                 # ... (error handling as before) ...
#                 return None

#             # 1. Concatenate Video Parts (should be silent at this stage)
#             logger.info(f"video_id: {video_id} - Concatenating {len(video_sub_clips_for_concat)} video-only sub-clips.", extra=log_extra)
#             final_video_track = concatenate_videoclips(video_sub_clips_for_concat, method="compose")
#             logger.info(f"video_id: {video_id} - Video-only track concatenated. Duration: {final_video_track.duration:.2f}s", extra=log_extra)

#             # 2. Concatenate Audio Parts
#             final_audio_track = None
#             if audio_sub_clips_for_concat:
#                 logger.info(f"video_id: {video_id} - Concatenating {len(audio_sub_clips_for_concat)} audio sub-clips.", extra=log_extra)
#                 try:
#                     final_audio_track = concatenate_audioclips(audio_sub_clips_for_concat)
#                     if final_audio_track:
#                          # Ensure FPS for the concatenated audio track if MoviePy doesn't set it
#                         if not hasattr(final_audio_track, 'fps') or final_audio_track.fps is None:
#                             detected_source_fps = main_video_clip.audio.fps if main_video_clip.audio and main_video_clip.audio.fps else 44100
#                             logger.warning(f"video_id: {video_id} - Final audio track missing FPS, setting to {detected_source_fps}", extra=log_extra)
#                             final_audio_track.fps = detected_source_fps
#                         logger.info(f"video_id: {video_id} - Audio tracks concatenated. Final audio duration: {final_audio_track.duration:.2f}s, FPS: {final_audio_track.fps}", extra=log_extra)
#                     else:
#                         logger.error(f"video_id: {video_id} - concatenate_audioclips returned None!", extra=log_extra)
#                 except Exception as e_audio_concat:
#                     logger.exception(f"video_id: {video_id} - Error during explicit audio concatenation: {e_audio_concat}", extra=log_extra)
#                     final_audio_track = None # Ensure it's None if concatenation failed
#             else:
#                 logger.warning(f"video_id: {video_id} - No audio tracks to concatenate. Final clip will be silent.", extra=log_extra)

#             # 3. Combine Video and Audio
#             if final_audio_track and final_video_track:
#                 # Ensure audio duration matches video, or trim/pad audio
#                 if abs(final_audio_track.duration - final_video_track.duration) > 0.1: # If more than 0.1s difference
#                     logger.warning(f"video_id: {video_id} - Video duration ({final_video_track.duration:.2f}s) and audio duration ({final_audio_track.duration:.2f}s) mismatch. Trimming audio to video duration.", extra=log_extra)
#                     final_audio_track = final_audio_track.subclip(0, final_video_track.duration) # Use .subclip for audio if it exists

#                 final_output_clip = final_video_track.set_audio(final_audio_track)
#                 logger.info(f"video_id: {video_id} - Combined video and audio tracks.", extra=log_extra)
#             elif final_video_track:
#                 final_output_clip = final_video_track # It's already silent
#                 logger.info(f"video_id: {video_id} - Proceeding with silent video track as no final audio was available/created.", extra=log_extra)
#             else: # Should not happen if video_sub_clips_for_concat was not empty
#                 logger.error(f"video_id: {video_id} - No final video track to write.", extra=log_extra)
#                 return None
            
#             # --- Write the Final Combined Clip ---
#             has_audio_for_write = final_output_clip.audio is not None
#             audio_fps_for_write = final_output_clip.audio.fps if has_audio_for_write and final_output_clip.audio.fps else None
            
#             logger.info(f"video_id: {video_id} - Writing final highlight clip to: {final_clip_path}. Has Audio: {has_audio_for_write}, Audio FPS: {audio_fps_for_write}, Video FPS: {main_video_fps_for_output}", extra=log_extra)
            
#             write_params = {
#                 "codec": "libx264", "audio_codec": "aac" if has_audio_for_write else None,
#                 "temp_audiofile": f'temp-audio-{video_id}-{str(uuid.uuid4())[:6]}.m4a',
#                 "remove_temp": True, "threads": os.cpu_count() if os.cpu_count() and os.cpu_count() > 0 else 2,
#                 "fps": main_video_fps_for_output, "logger": 'bar',
#                 "preset": "medium", "ffmpeg_params": ["-crf", "23"]
#             }
#             if has_audio_for_write:
#                 write_params["audio"] = True
#                 if audio_fps_for_write: # Only pass audio_fps if we have it
#                     write_params["audio_fps"] = audio_fps_for_write
#                 write_params["audio_bitrate"] = "128k"
#             else:
#                 write_params["audio"] = False
            
#             final_output_clip.write_videofile(final_clip_path, **write_params)
#             # ... (DB updates and return logic) ...
#             logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)
#             final_clip_path_to_return = final_clip_path
#         # main_video_clip is closed by 'with'

#     except Exception as e_main: # ... (error handling) ...
#         return None
#     finally:
#         # Close individual sub_clips (video parts and audio parts if they were separate objects)
#         # VideoFileClip objects are views and don't hold open file handles themselves after the main clip is closed
#         # The main VideoFileClip is closed by 'with'.
#         # Concatenated clips should be closed if write_videofile fails.
#         if final_output_clip and not final_clip_path_to_return: # If write failed but clip object exists
#             try: final_output_clip.close()
#             except: pass
#         logger.debug(f"video_id: {video_id} - Exiting generate_highlight_clip (finally).", extra=log_extra)
    
#     if final_clip_path_to_return:
#         async with get_db_session() as session_after_write:
#             # ... DB updates for success ...
#             return final_clip_path_to_return
#     else:
#         # ... DB updates for failure if not already set ...
#         return None

# backend/services/clip_builder_service.py
import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager 
import subprocess 
import shutil # For cleaning up processing base path for highlights

from moviepy.editor import (
    VideoFileClip, TextClip, CompositeVideoClip,
    concatenate_videoclips, concatenate_audioclips, AudioFileClip
)

from . import database_service
from .database_service import VideoProcessingStatus, get_db_session 
from . import storage_service 
from .storage_service import download_file_from_supabase

logger = logging.getLogger(__name__)
HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights" 
TEMP_VIDEO_DIR = os.getenv("TEMP_VIDEO_DIR", "/tmp/clippilot_uploads_dev")

MUSIC_ENABLED = False 

def ensure_dir(directory_path):
    try: os.makedirs(directory_path, exist_ok=True)
    except OSError as e: logger.error(f"Failed to create directory {directory_path}: {e}"); raise

async def generate_highlight_clip(
    video_id: str,
    segments_to_include: List[Dict[str, Any]],
    processing_base_path: str, # Local temporary base path for this video_id's processing run
    output_filename: Optional[str] = None
) -> Optional[str]: # Returns Supabase key of the highlight on success, else None
    log_extra = {'video_id': video_id}
    logger.info(f"Input segments: {json.dumps(segments_to_include, indent=2)}", extra=log_extra)
    logger.info(f"Starting highlight clip generation. Segments: {len(segments_to_include)}, Music: {MUSIC_ENABLED}", extra=log_extra)

    if not segments_to_include: logger.warning("No segments provided. Aborting.", extra=log_extra); return None

    original_video_supabase_key: Optional[str] = None
    main_video_fps_for_output: float = 24.0
    
    async with get_db_session() as session:
        video_record = await database_service.get_video_record_by_uuid(session, video_id)
        if not video_record or not video_record.original_video_file_path:
            error_msg = "Original video record or Supabase key not found in DB."
            logger.error(error_msg, extra=log_extra)
            # No need to update status here as it's a pre-condition check, not a processing failure of this task
            return None
        original_video_supabase_key = video_record.original_video_file_path
    
    # --- Setup Local Paths ---
    # Ensure processing_base_path (e.g., /tmp/clippilot_uploads_dev/video_uuid) exists for this specific run
    ensure_dir(processing_base_path)
    
    highlights_output_dir_local = os.path.join(processing_base_path, HIGHLIGHT_CLIPS_SUBDIR)
    ensure_dir(highlights_output_dir_local)
    
    local_base_output_filename = output_filename or f"highlight_{original_video_supabase_key}.mp4"
    final_clip_local_path = os.path.join(highlights_output_dir_local, local_base_output_filename)

    original_filename_from_key = os.path.basename(original_video_supabase_key)
    local_original_video_for_moviepy = os.path.join(processing_base_path, f"temp_moviepy_original_{video_id}_{original_filename_from_key}")

    final_supabase_key_to_return = None
    final_output_clip_obj_moviepy = None

    try:
        logger.info(f"Downloading original video from Supabase. Key: '{original_video_supabase_key}' to Local: '{local_original_video_for_moviepy}'", extra=log_extra)
        downloaded_path = await download_file_from_supabase(
            bucket_name=storage_service.VIDEO_BUCKET_NAME,
            source_path=original_video_supabase_key,
            local_temp_path=local_original_video_for_moviepy
        )
        if not downloaded_path or not os.path.exists(local_original_video_for_moviepy):
            error_msg = "Failed to download original video from Supabase for clip building."
            logger.error(error_msg, extra=log_extra)
            async with get_db_session() as error_session: await database_service.update_video_status_and_error(error_session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            return None
        logger.info(f"Video for MoviePy successfully downloaded: {local_original_video_for_moviepy}", extra=log_extra)

        with VideoFileClip(local_original_video_for_moviepy, audio=True) as main_video_clip:
            main_video_fps_for_output = main_video_clip.fps or 24.0
            original_video_has_audio = main_video_clip.audio is not None
            source_audio_fps_local = main_video_clip.audio.fps if original_video_has_audio and main_video_clip.audio.fps else 44100
            logger.info(f"Loaded local original. Duration:{main_video_clip.duration:.2f}s, HasAudio:{original_video_has_audio}, AudioFPS:{source_audio_fps_local}", extra=log_extra)

            video_sub_clips_for_concat = []
            audio_sub_clips_for_concat = []
            # ... (Your segment processing loop - ensure subclip method is correct and text_content for TextClip)
            for i, segment_info_loop in enumerate(segments_to_include):
                start = segment_info_loop.get("start_time"); end = segment_info_loop.get("end_time")
                text_content_loop = segment_info_loop.get("text_content")
                # Segment validation...
                if start is None or end is None or start >= end or start >= main_video_clip.duration: continue
                end = min(end, main_video_clip.duration)
                if start == end: continue
                
                raw_sub_clip = main_video_clip.subclip(start, end) # OR .subclipped()
                if raw_sub_clip.audio: audio_sub_clips_for_concat.append(raw_sub_clip.audio)
                
                video_part = raw_sub_clip.without_audio() if raw_sub_clip.audio else raw_sub_clip
                if text_content_loop and str(text_content_loop).strip():
                    try:
                        txt_clip = TextClip(txt=str(text_content_loop), font="Arial", # CHANGE FOR DOCKER
                                            fontsize=24, color='white', method='caption', 
                                            size=(video_part.w * 0.9, None), bg_color='transparent',
                                            stroke_color='black', stroke_width=1
                                           ).set_pos('bottom').set_duration(video_part.duration) # Simplified pos
                        video_part = CompositeVideoClip([video_part, txt_clip], use_bgclip=True if video_part.mask is None else False)
                    except Exception as e_txt: logger.error(f"TextClip error for seg {i+1}: {e_txt}", extra=log_extra)
                video_sub_clips_for_concat.append(video_part)


            if not video_sub_clips_for_concat: logger.error("No valid sub-clips generated.", extra=log_extra); return None

            final_video_track = concatenate_videoclips(video_sub_clips_for_concat, method="compose")
            final_audio_track = None
            if audio_sub_clips_for_concat:
                try:
                    final_audio_track = concatenate_audioclips(audio_sub_clips_for_concat)
                    if final_audio_track and (not hasattr(final_audio_track, 'fps') or not final_audio_track.fps):
                        final_audio_track.fps = source_audio_fps_local or 44100
                except Exception as e_ac: logger.error(f"Audio concat error: {e_ac}", extra=log_extra); final_audio_track = None
            
            if final_audio_track:
                if abs(final_audio_track.duration - final_video_track.duration) > 0.2: # Allow slightly more leeway
                     final_audio_track = final_audio_track.set_duration(final_video_track.duration)
                final_output_clip_obj_moviepy = final_video_track.set_audio(final_audio_track)
            else:
                final_output_clip_obj_moviepy = final_video_track # Silent
            
            has_audio = final_output_clip_obj_moviepy.audio is not None
            write_audio_fps = final_output_clip_obj_moviepy.audio.fps if has_audio and final_output_clip_obj_moviepy.audio.fps else (source_audio_fps_local if source_audio_fps_local else None)

            logger.info(f"Writing final clip locally: {final_clip_local_path}. Has Audio: {has_audio}, AudioFPS: {write_audio_fps}", extra=log_extra)
            write_params = {
                "codec": "libx264", "audio_codec": "aac" if has_audio else None,
                "audio": has_audio, "fps": main_video_fps_for_output, "logger": 'bar',
                "temp_audiofile": os.path.join(processing_base_path, f"temp_audio_final_{video_id}.m4a"), "remove_temp": True
            }
            if has_audio and write_audio_fps: write_params["audio_fps"] = write_audio_fps
            
            final_output_clip_obj_moviepy.write_videofile(final_clip_local_path, **write_params)
            logger.info(f"Local highlight generated: {final_clip_local_path}", extra=log_extra)

        if os.path.exists(final_clip_local_path):
            highlight_supabase_key = f"{video_id}/highlights/{output_filename}" # Use the base output_filename
            logger.info(f"Uploading '{final_clip_local_path}' to Supabase key '{highlight_supabase_key}'", extra=log_extra)
            with open(final_clip_local_path, "rb") as f_upload:
                uploaded_highlight_key_supabase = await storage_service.upload_file_to_supabase(
                    file_object=f_upload, bucket_name=storage_service.HIGHLIGHT_BUCKET_NAME, 
                    destination_path_in_bucket=highlight_supabase_key, content_type='video/mp4'
                )
            if uploaded_highlight_key_supabase:
                final_clip_path_to_return = uploaded_highlight_key_supabase
                async with get_db_session() as s_after:
                    await database_service.update_video_asset_paths_record(s_after, video_id, highlight_clip_path=uploaded_highlight_key_supabase)
                    await database_service.update_video_status_and_error(s_after, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
            else: logger.error("Failed to upload highlight to Supabase.", extra=log_extra); return None # DB status FAILED will be set by main try-except
        else: logger.error(f"Local highlight {final_clip_local_path} not found for upload.", extra=log_extra); return None

    except Exception as e_main:
        error_msg_main = f"Highlight generation error: {type(e_main).__name__} - {str(e_main)}"
        logger.exception(f"Overall error in highlight generation.", extra=log_extra)
        async with get_db_session() as error_session_main:
            await database_service.update_video_status_and_error(error_session_main, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_main)
        return None
    finally:
        if os.path.exists(local_original_video_for_moviepy): os.remove(local_original_video_for_moviepy)
        if os.path.exists(final_clip_local_path): os.remove(final_clip_local_path)
        if final_output_clip_obj_moviepy: 
            try: 
                final_output_clip_obj_moviepy.close() 
                                          
            except: pass
        logger.debug(f"Exiting generate_highlight_clip (finally).", extra=log_extra)
    
    return final_clip_path_to_return

# # ... (if __name__ == "__main__": block - ensure it tests this new audio logic)