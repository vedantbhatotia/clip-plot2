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