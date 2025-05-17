# backend/services/segment_processor_service.py
import logging
from typing import List, Dict, Any, Optional
import os # For __main__ test block (os.path)

# Get a logger for this module. It will inherit the global config from main.py when run as part of the app.
# When run directly, its config is set in the __main__ block.
logger = logging.getLogger(__name__)

def expand_segments(
    segments: List[Dict[str, Any]],
    padding_start_sec: float = 1.0,
    padding_end_sec: float = 1.0,
    video_duration: Optional[float] = None,
    video_id: Optional[str] = None # For logging context
) -> List[Dict[str, Any]]:
    """
    Adds padding to the start and end of each segment,
    ensuring they don't exceed video_duration if provided.
    """
    log_extra = {'video_id': video_id} if video_id else {} # Prepare extra for logger
    expanded_segments: List[Dict[str, Any]] = []

    if not segments:
        return []

    for i, seg_original in enumerate(segments):
        seg = seg_original.copy() # Work on a copy
        start = seg.get("start_time")
        end = seg.get("end_time")

        if start is None or end is None:
            logger.warning(f"Segment {i} missing start/end times, skipping expansion: {seg}", extra=log_extra)
            expanded_segments.append(seg)
            continue

        new_start = start - padding_start_sec
        new_end = end + padding_end_sec
        new_start = max(0.0, new_start) # Ensure start time is not negative

        if video_duration is not None:
            new_end = min(new_end, video_duration)
            if new_start >= video_duration: # If padded start is beyond video end
                logger.debug(f"video_id: {video_id} - Segment {i} (orig: {start:.2f}-{end:.2f}) new_start ({new_start:.2f}) is beyond video_duration ({video_duration:.2f}). Skipping.", extra=log_extra)
                continue
        
        if new_start >= new_end: # Ensure positive duration
            logger.debug(f"video_id: {video_id} - Segment {i} (orig: {start:.2f}-{end:.2f}) resulted in zero/negative duration (new: {new_start:.2f}-{new_end:.2f}) after padding. Skipping.", extra=log_extra)
            continue
            
        seg["start_time"] = round(new_start, 3)
        seg["end_time"] = round(new_end, 3)
        expanded_segments.append(seg)
        logger.debug(f"video_id: {video_id} - Segment {i} expanded: original=({start:.2f}-{end:.2f}), new=({seg['start_time']:.2f}-{seg['end_time']:.2f})", extra=log_extra)

    return expanded_segments

def merge_overlapping_segments(
    segments: List[Dict[str, Any]],
    max_gap_to_merge_sec: float = 0.5,
    video_id: Optional[str] = None # For logging context
) -> List[Dict[str, Any]]:
    """
    Merges overlapping or closely sequential segments.
    Assumes segments are dictionaries with 'start_time', 'end_time',
    and optionally 'text_content'.
    """
    log_extra = {'video_id': video_id} if video_id else {}
    if not segments:
        return []

    # Sort segments by start time to ensure merging logic works correctly
    sorted_segments = sorted(segments, key=lambda x: x.get("start_time", float('inf')))
    
    merged_segments: List[Dict[str, Any]] = []
    if not sorted_segments:
        return []

    current_merged_segment = sorted_segments[0].copy()

    for i in range(1, len(sorted_segments)):
        next_segment = sorted_segments[i]
        
        current_start = current_merged_segment.get("start_time")
        current_end = current_merged_segment.get("end_time")
        next_start = next_segment.get("start_time")
        next_end = next_segment.get("end_time")

        if any(t is None for t in [current_start, current_end, next_start, next_end]):
            logger.warning(f"video_id: {video_id} - Segment(s) involved in merge check have None times. Finalizing current. Current={current_merged_segment}, Next={next_segment}", extra=log_extra)
            merged_segments.append(current_merged_segment)
            current_merged_segment = next_segment.copy()
            continue

        # Check for overlap or if the gap is small enough to merge
        if next_start <= (current_end + max_gap_to_merge_sec):
            current_merged_segment["end_time"] = max(current_end, next_end)
            
            current_text = current_merged_segment.get("text_content", "")
            next_text = next_segment.get("text_content", "")
            if current_text and next_text and next_text.strip() and current_text.strip() != next_text.strip() and next_text.strip() not in current_text.strip():
                current_merged_segment["text_content"] = f"{current_text.strip()}\n{next_text.strip()}"
            elif next_text and not current_text: # If current_text was empty, just use next_text
                current_merged_segment["text_content"] = next_text.strip()
            logger.debug(f"video_id: {video_id} - Merged with next segment. New current_merged_segment end: {current_merged_segment['end_time']:.2f}. Text: '{current_merged_segment.get('text_content', '')[:30]}...'", extra=log_extra)
        else:
            merged_segments.append(current_merged_segment)
            current_merged_segment = next_segment.copy()
            
    if current_merged_segment: # Add the last processed segment
        merged_segments.append(current_merged_segment)
        
    return merged_segments

def refine_segments_for_clip(
    segments: List[Dict[str, Any]],
    padding_start_sec: float = 1.0,
    padding_end_sec: float = 1.5,
    max_gap_to_merge_sec: float = 0.5,
    video_duration: Optional[float] = None,
    video_id: Optional[str] = None # For logging context
) -> List[Dict[str, Any]]:
    """
    Applies expansion and merging to a list of segments to make them more
    suitable for a highlight clip.
    """
    log_extra = {'video_id': video_id} if video_id else {}
    if not segments:
        logger.info(f"video_id: {video_id or 'N/A'} - No segments provided to refine_segments_for_clip.", extra=log_extra)
        return []

    logger.info(
        f"video_id: {video_id or 'N/A'} - Refining {len(segments)} segments. "
        f"Padding(start/end): {padding_start_sec}s/{padding_end_sec}s. "
        f"Merge Gap: {max_gap_to_merge_sec}s. Video Duration: {video_duration or 'N/A'}s.",
        extra=log_extra
    )
    
    expanded = expand_segments(segments, padding_start_sec, padding_end_sec, video_duration, video_id)
    if not expanded:
        logger.info(f"video_id: {video_id or 'N/A'} - No segments remained after expansion.", extra=log_extra)
        return []
    logger.debug(f"video_id: {video_id or 'N/A'} - Segments after expansion ({len(expanded)}): {expanded}", extra=log_extra)

    merged = merge_overlapping_segments(expanded, max_gap_to_merge_sec, video_id)
    if not merged:
        logger.info(f"video_id: {video_id or 'N/A'} - No segments remained after merging.", extra=log_extra)
        return []
    logger.debug(f"video_id: {video_id or 'N/A'} - Segments after merging ({len(merged)}): {merged}", extra=log_extra)
    
    final_segments: List[Dict[str, Any]] = []
    for seg_original in merged:
        seg = seg_original.copy()
        start = seg.get("start_time")
        end = seg.get("end_time")

        if start is None or end is None:
            logger.warning(f"video_id: {video_id or 'N/A'} - Segment missing time info after merging: {seg}. Keeping as is or consider filtering.", extra=log_extra)
            final_segments.append(seg) # Decide if you want to keep such segments
            continue
        
        start = round(max(0.0, start), 3)
        if video_duration is not None:
            end = round(min(end, video_duration), 3)
        else:
            end = round(end, 3)

        if video_duration is not None and start >= video_duration:
            logger.debug(f"video_id: {video_id or 'N/A'} - Final check: Segment start ({start=}) at/beyond video duration ({video_duration=}). Skipping.", extra=log_extra)
            continue
        if start >= end:
            logger.debug(f"video_id: {video_id or 'N/A'} - Final check: Segment has zero or negative duration ({start=}, {end=}). Skipping.", extra=log_extra)
            continue
            
        seg["start_time"] = start
        seg["end_time"] = end
        final_segments.append(seg)

    logger.info(f"video_id: {video_id or 'N/A'} - Final refined segments count: {len(final_segments)}", extra=log_extra)
    return final_segments


if __name__ == "__main__":
    # This basicConfig is only for when running this script directly.
    # It uses a format that does NOT require 'video_id' in 'extra' for general script logs.
    # Logs from within the functions above WILL use 'video_id' if it's passed and
    # the global formatter (from main.py if this module is imported) expects it.
    if not logging.getLogger().hasHandlers():
        # Simple format for direct script execution logs
        direct_test_log_format = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=direct_test_log_format)

    logger.info("Running segment_processor_service.py directly for testing...") # Will use direct_test_log_format

    test_video_id_for_logs = "seg_proc_direct_test_01" # For logs from inside the functions
    test_video_duration = 60.0

    sample_segments = [
        {"start_time": 5.0, "end_time": 7.0, "text_content": "Alpha segment."},
        {"start_time": 7.3, "end_time": 9.0, "text_content": "Beta segment, close to Alpha."},
        {"start_time": 15.0, "end_time": 16.0, "text_content": "Gamma (short)."},
        {"start_time": 15.8, "end_time": 18.0, "text_content": "Delta (overlaps Gamma with padding)."},
        {"start_time": 25.0, "end_time": 26.0, "text_content": "Epsilon (isolated)."},
        {"start_time": 58.0, "end_time": 62.0, "text_content": "Zeta (will be capped by duration)."},
        {"start_time": 30.0, "end_time": 30.5, "text_content": "Eta."},
        {"start_time": 30.6, "end_time": 31.0, "text_content": "Theta (close to Eta)."}
    ]
    logger.info(f"\n--- Test Case: refine_segments_for_clip ---") # Uses direct_test_log_format
    logger.info(f"Original Segments ({len(sample_segments)}):") # Uses direct_test_log_format
    for idx, s_val in enumerate(sample_segments): logger.info(f"  Original Segment {idx}: {s_val}") # Uses direct_test_log_format

    # Parameters for refinement
    padding_s, padding_e, merge_g = 0.5, 0.5, 0.2
    logger.info(f"\nRefining with: padding_start={padding_s}s, padding_end={padding_e}s, merge_gap={merge_g}s, video_duration={test_video_duration}s") # Uses direct_test_log_format
    
    # Calling the main refinement function
    # Logs INSIDE refine_segments_for_clip and its sub-functions will use test_video_id_for_logs if the global formatter is set
    # by main.py to expect 'video_id' via 'extra'. If not, they will just log without the video_id prefix from formatter.
    refined_segments_output = refine_segments_for_clip(
        sample_segments,
        padding_start_sec=padding_s,
        padding_end_sec=padding_e,
        max_gap_to_merge_sec=merge_g,
        video_duration=test_video_duration,
        video_id=test_video_id_for_logs # Passed for internal logging context
    )
    logger.info(f"\nFinal Refined Segments ({len(refined_segments_output)}):") # Uses direct_test_log_format
    for idx, r_s_val in enumerate(refined_segments_output): logger.info(f"  Refined Segment {idx}: {r_s_val}") # Uses direct_test_log_format

    # Example test: Empty list
    logger.info(f"\n--- Test Case: Empty Segments List ---") # Uses direct_test_log_format
    empty_refined = refine_segments_for_clip([], video_id="empty_test_id")
    logger.info(f"Result for empty list: {empty_refined}") # Uses direct_test_log_format

    # Example test: Segments entirely out of duration
    segments_out = [{"start_time": 70.0, "end_time": 75.0, "text_content": "Segment out of bounds"}]
    logger.info(f"\n--- Test Case: Segments out of bounds ---") # Uses direct_test_log_format
    logger.info(f"Original Segments: {segments_out}") # Uses direct_test_log_format
    out_refined = refine_segments_for_clip(segments_out, video_duration=test_video_duration, video_id="out_of_bounds_test")
    logger.info(f"Result for out of bounds list: {out_refined}") # Uses direct_test_log_format