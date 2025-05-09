# services/frames_extract.py
import argparse
import os
import subprocess # For calling FFmpeg directly
# Or if you prefer MoviePy's abstractions:
# from moviepy.editor import VideoFileClip

# --- Configuration (can be moved to a config file later) ---
# Define default output directories. These could be relative to a base processing path.
DEFAULT_FRAMES_SUBDIR = "extracted_frames"
DEFAULT_AUDIO_SUBDIR = "extracted_audio"
FRAME_RATE_STR = "1/2" # For FFmpeg: 1 frame every 2 seconds
AUDIO_SAMPLE_RATE = "16000" # Good for Whisper
AUDIO_CHANNELS = "1" # Mono

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def extract_media_content(video_path: str, output_base_path: str):
    """
    Extracts frames and audio from the given video file.

    Args:
        video_path (str): Absolute path to the input video file.
        output_base_path (str): Absolute path to a base directory where 
                                'extracted_frames' and 'extracted_audio' subdirectories
                                will be created for this specific video.

    Returns:
        tuple: (frames_directory_path, audio_file_path)
               Returns (None, None) if extraction fails for any part.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None, None

    video_filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]

    # Define specific output paths for this video's content
    frames_output_dir = os.path.join(output_base_path, DEFAULT_FRAMES_SUBDIR)
    audio_output_dir = os.path.join(output_base_path, DEFAULT_AUDIO_SUBDIR)
    
    ensure_dir(frames_output_dir)
    ensure_dir(audio_output_dir)

    # Naming convention for output files
    # Frames will be frame_0001.jpg, frame_0002.jpg etc. in frames_output_dir
    # Audio will be video_filename_no_ext.wav in audio_output_dir
    audio_output_path = os.path.join(audio_output_dir, f"{video_filename_no_ext}.wav")

    print(f"Starting media extraction for: {video_path}")
    print(f"  Output frames to: {frames_output_dir}")
    print(f"  Output audio to: {audio_output_path}")

    try:
        # Extract Frames (e.g., 1 frame every 2 seconds)
        # -vf "fps=1/2": sets the frame rate.
        # -q:v 2: sets quality for JPG output (2-5 is good)
        # -loglevel error: only show errors
        frames_command = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={FRAME_RATE_STR}",
            "-q:v", "2",
            os.path.join(frames_output_dir, "frame_%04d.jpg"),
            "-loglevel", "error",
            "-hide_banner"
        ]
        print(f"  Executing frames command: {' '.join(frames_command)}")
        subprocess.run(frames_command, check=True)
        print(f"  Frame extraction successful.")

        # Extract Audio (to WAV, 16kHz, mono for Whisper)
        # -vn: no video output
        # -acodec pcm_s16le: standard WAV codec
        # -ar: audio sample rate
        # -ac: audio channels
        # -y: overwrite output file without asking
        audio_command = [
            "ffmpeg", "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", AUDIO_SAMPLE_RATE,
            "-ac", AUDIO_CHANNELS,
            audio_output_path,
            "-y", # Overwrite if exists
            "-loglevel", "error",
            "-hide_banner"
        ]
        print(f"  Executing audio command: {' '.join(audio_command)}")
        subprocess.run(audio_command, check=True)
        print(f"  Audio extraction successful.")
        
        return frames_output_dir, audio_output_path

    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during media extraction: {e}")
        return None, None


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Extract frames and audio from a video file.")
#     parser.add_argument("video_path", type=str, help="Path to the input video file.")
#     parser.add_argument("output_base_path", type=str, help="Base directory where 'extracted_frames' and 'extracted_audio' subdirectories will be created.")
    
#     args = parser.parse_args()

#     if not os.path.isfile(args.video_path):
#         print(f"Error: Input video file '{args.video_path}' not found or is not a file.")
#         exit(1)

#     ensure_dir(args.output_base_path) # Ensure the base output path itself exists

#     print(f"Processing video: {args.video_path}")
#     print(f"Output base directory: {args.output_base_path}")
    
#     frames_dir, audio_file = extract_media_content(args.video_path, args.output_base_path)

#     if frames_dir and audio_file:
#         print("\nExtraction Summary:")
#         print(f"  Frames saved in: {frames_dir}")
#         print(f"  Audio saved to: {audio_file}")
#         print("Extraction completed successfully.")
#     else:
#         print("\nExtraction failed.")
#         exit(1)

extract_media_content("file_example_MP4_640_3MG.mp4","test_output")