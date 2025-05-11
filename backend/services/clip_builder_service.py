
import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional

from moviepy import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    AudioFileClip, # For MusicGen output
)
MUSIC_ENABLED = False # Explicitly disable music generation

# --- Import Database Service ---
from . import database_service
from .database_service import VideoProcessingStatus, get_db_session # For updating status/paths

logger = logging.getLogger(__name__)
HIGHLIGHT_CLIPS_SUBDIR = "generated_highlights"
# GENERATED_MUSIC_SUBDIR = "generated_music" # No longer needed

def ensure_dir(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise

# --- get_musicgen_model function is no longer needed ---
# def get_musicgen_model():
#     global musicgen_model
#     # ... (implementation removed) ...
#     return musicgen_model

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

    sub_clips = []
    total_clip_duration = 0
    default_fontsize = 24
    default_text_color = 'white'
    
    # To store the final video object for closing in finally block
    final_video_obj_to_close = None 

    try:
        with VideoFileClip(original_video_file_path) as main_video_clip:
            logger.info(f"video_id: {video_id} - Loaded original video. Duration: {main_video_clip.duration}s", extra=log_extra)
            for i, segment_info in enumerate(segments_to_include):
                start = segment_info.get("start_time")
                end = segment_info.get("end_time")
                text_content = segment_info.get("text_content")

                if start is None or end is None or start >= end:
                    logger.warning(f"video_id: {video_id} - Invalid segment times for segment {i}: start={start}, end={end}. Skipping.", extra=log_extra)
                    continue
                if end > main_video_clip.duration:
                    logger.warning(f"video_id: {video_id} - Segment {i} end time {end}s exceeds video duration {main_video_clip.duration}s. Capping.", extra=log_extra)
                    end = main_video_clip.duration
                if start >= main_video_clip.duration: # Also check if start is beyond or at duration
                    logger.warning(f"video_id: {video_id} - Segment {i} start time {start}s is at or beyond video duration {main_video_clip.duration}s. Skipping.", extra=log_extra)
                    continue
                if start == end: # Skip zero-duration clips
                    logger.warning(f"video_id: {video_id} - Segment {i} has zero duration (start == end). Skipping.", extra=log_extra)
                    continue


                logger.info(f"video_id: {video_id} - Processing segment {i+1}: {start}s - {end}s", extra=log_extra)
                # Corrected method name
                sub_clip = main_video_clip.subclipped(start, end) 
                if sub_clip.audio is None:
                    logger.error(f"video_id: {video_id} - Subclip {i+1} for segment {start}-{end} has NO audio! This will likely cause issues. Main clip audio was: {main_video_clip.audio is not None}", extra=log_extra)
                    # Decide how to handle: skip this subclip? proceed without its audio? For now, it will just be added.
                else:
                    logger.info(f"video_id: {video_id} - Subclip {i+1} for segment {start}-{end} has audio. Duration: {sub_clip.audio.duration}", extra=log_extra)

                
                if text_content and text_content.strip():
                    try:
                        txt_clip = TextClip(
                            text_content,
                            fontsize=default_fontsize,
                            color=default_text_color,
                            font="Arial", 
                            method='caption',
                            size=(sub_clip.w * 0.9, None),
                            bg_color='transparent',
                            stroke_color='black',
                            stroke_width=1
                        )
                        txt_clip = txt_clip.set_pos(('center', 'bottom-10%')).set_duration(sub_clip.duration)
                        sub_clip = CompositeVideoClip([sub_clip, txt_clip], use_bgclip=True if sub_clip.mask is None else False) # Added use_bgclip for safety with transparent TextClip
                        logger.info(f"video_id: {video_id} - Added subtitle to segment {i+1}.", extra=log_extra)
                    except Exception as e_textclip:
                        logger.error(f"video_id: {video_id} - Failed to create TextClip for segment {i+1}: {e_textclip}. Proceeding without subtitle for this segment.", extra=log_extra)

                sub_clips.append(sub_clip)
                total_clip_duration += sub_clip.duration
        
        if not sub_clips:
            logger.error(f"video_id: {video_id} - No valid sub-clips were generated.", extra=log_extra)
            return None

        final_video_obj_to_close = concatenate_videoclips(sub_clips, method="compose")
        logger.info(f"video_id: {video_id} - Concatenated {len(sub_clips)} sub-clips. Total duration: {total_clip_duration:.2f}s", extra=log_extra)

        # --- Music Generation Block is now effectively skipped due to MUSIC_ENABLED = False ---
        # if MUSIC_ENABLED and total_clip_duration > 0:
        #    ... (music logic would go here) ...
        # The original audio from the concatenated clips will be preserved unless explicitly replaced.

        logger.info(f"video_id: {video_id} - Writing final highlight clip (without new music) to: {final_clip_path}", extra=log_extra)
        final_video_obj_to_close.write_videofile(
            final_clip_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f'temp-audio-{str(uuid.uuid4())[:8]}.m4a', # Unique temp audio file
            remove_temp=True,
            threads=os.cpu_count() or 4, # Use available CPUs or default to 4
            fps=24 
        )
        logger.info(f"video_id: {video_id} - Highlight clip successfully generated: {final_clip_path}", extra=log_extra)

        async with get_db_session() as session:
            await database_service.update_video_asset_paths_record(session, video_id, highlight_clip_path=final_clip_path)
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.HIGHLIGHT_GENERATED)
            logger.info(f"video_id: {video_id} - DB updated with highlight path and status.", extra=log_extra)

        return final_clip_path

    except Exception as e:
        logger.exception(f"video_id: {video_id} - Error during highlight clip generation process.", extra=log_extra)
        async with get_db_session() as session: # Try to update status on failure
            await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, f"Highlight generation error: {str(e)}")
        return None
    finally:
        # Close all sub_clips and the final_video if they exist
        for clip in sub_clips:
            try:
                clip.close()
            except Exception as e_close:
                logger.debug(f"video_id: {video_id} - Minor error closing a sub_clip: {e_close}", extra=log_extra)
        if final_video_obj_to_close: # Renamed variable for clarity
            try:
                final_video_obj_to_close.close()
            except Exception as e_close_final:
                logger.debug(f"video_id: {video_id} - Minor error closing final_video: {e_close_final}", extra=log_extra)


if __name__ == "__main__":
    # ... (your existing test block, it should now run without attempting music generation) ...
    # Ensure the mock for database_service.update_video_asset_paths_record also handles highlight_clip_path
    # and that the mock for database_service.update_video_status_and_error handles HIGHLIGHT_GENERATED status.
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    async def test_clip_generation():
        logger.info("Running clip_builder_service.py directly for testing (NO MUSIC)...")
        
        class MockVideoRecord:
            def __init__(self, uuid, path):
                self.video_uuid = uuid
                self.original_video_file_path = path
        
        async def mock_get_video_record(session, uuid):
            test_video_file = "sample_short_video.mp4" 
            if not os.path.exists(test_video_file):
                logger.error(f"Test video '{test_video_file}' not found. Please create it or update path.")
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=24", # Smaller size for faster test
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", test_video_file
                    ], check=True, capture_output=True, text=True)
                    logger.info(f"Created dummy test video: {test_video_file}")
                except Exception as e_ffmpeg_dummy:
                    logger.error(f"Failed to create dummy test video with ffmpeg: {e_ffmpeg_dummy}")
                    return None
            return MockVideoRecord(uuid, os.path.abspath(test_video_file))

        class MockSessionObj: # Simple object to act as session for mock
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
        test_processing_path = f"/tmp/clippilot_test_clipbuilder/{test_vid_id}" 
        ensure_dir(test_processing_path)

        segments = [
            {"start_time": 1.0, "end_time": 3.5, "text_content": "First amazing segment!"},
            {"start_time": 5.2, "end_time": 8.0, "text_content": "Another key moment here."},
        ]

        logger.info(f"Attempting to generate highlight for video_id: {test_vid_id}")
        highlight_path = await generate_highlight_clip(
            video_id=test_vid_id,
            segments_to_include=segments,
            processing_base_path=test_processing_path
        )

        if highlight_path:
            logger.info(f"Highlight clip generated successfully: {highlight_path}")
        else:
            logger.error("Highlight clip generation failed.")

        database_service.get_video_record_by_uuid = original_get_video_record
        database_service.get_db_session = original_get_session
        database_service.update_video_asset_paths_record = original_update_asset_paths
        database_service.update_video_status_and_error = original_update_status
        
        # import shutil
        # if os.path.exists(f"/tmp/clippilot_test_clipbuilder"):
        #     shutil.rmtree(f"/tmp/clippilot_test_clipbuilder")
        # if os.path.exists("sample_short_video.mp4"):
        #    os.remove("sample_short_video.mp4")


    import asyncio
    import subprocess # Make sure subprocess is imported for the test
    asyncio.run(test_clip_generation())