# Example: services/storage_service.py (New File)
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Optional, List 
import logging

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VIDEO_BUCKET_NAME = os.getenv("VIDEO_BUCKET_NAME", "media")
HIGHLIGHT_BUCKET_NAME = os.getenv("HIGHLIGHT_BUCKET_NAME", "media")

supabase_client: Optional[Client] = None
# print(f"Supabase URL: {SUPABASE_URL}")

logger = logging.getLogger(__name__)
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        pass 

def get_supabase_client() -> Optional[Client]:
    if not supabase_client:
        # This state should ideally not be reached if initialized at module load
        # Or re-attempt initialization if appropriate for your app lifecycle
        raise RuntimeError("Supabase client not initialized. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")
    return supabase_client

# --- Upload ---
# async def upload_file_to_supabase(
#     file_object, 
#     bucket_name: str, 
#     destination_path: str, 
#     content_type: Optional[str] = 'application/octet-stream' 
# ) -> Optional[str]:
#     client = get_supabase_client()
#     if not client: return None
#     try:
       
#         file_object.seek(0) 
#         file_bytes = file_object.read()
        
#         logger.info(f"Uploading to Supabase. Bucket: '{bucket_name}', Path: '{destination_path}', Content-Type: '{content_type}'")
        
#         response = client.storage.from_(bucket_name).upload(
#             path=destination_path,
#             file=file_bytes, 
#             file_options={"content-type": content_type, "x-upsert": "true"} # x-upsert to overwrite if exists
#         )
#         logger.info(f"Supabase upload response: {response.path}")
#         # if hasattr(response, 'text'): # Or response.content for bytes
#         #     logger.debug(f"Supabase upload response text: {response.text}")
#         return response.full_path
#     except Exception as e:
#         # Log e
#         return None
async def upload_file_to_supabase(
    file_object, 
    bucket_name: str, 
    destination_path_in_bucket: str, # e.g., "video_uuid/original_video.mp4"
    content_type: Optional[str] = 'application/octet-stream'
) -> Optional[str]: # Returns the destination_path_in_bucket on success
    client = get_supabase_client()
    log_extra_op = {'video_id': destination_path_in_bucket.split('/')[0] if '/' in destination_path_in_bucket else 'N/A_UPLOAD'}
    
    if not client: 
        logger.error(f"Supabase client not available for upload to {bucket_name}/{destination_path_in_bucket}", extra=log_extra_op)
        return None
    try:
        file_object.seek(0) 
        file_bytes = file_object.read()
        
        logger.info(f"Uploading to Supabase. Bucket: '{bucket_name}', Path in Bucket: '{destination_path_in_bucket}', Size: {len(file_bytes)} bytes", extra=log_extra_op)
        
        response = client.storage.from_(bucket_name).upload(
            path=destination_path_in_bucket, # This is the path within the bucket
            file=file_bytes,
            file_options={"content-type": content_type, "x-upsert": "true"} 
        )
        
        logger.debug(f"Supabase upload response status: {response}", extra=log_extra_op)
        
        if response:
            logger.info(f"Successfully uploaded to Supabase. Bucket: '{bucket_name}', Key In Bucket: '{destination_path_in_bucket}'", extra=log_extra_op)
            return destination_path_in_bucket # Return the path within the bucket
        else:
            # ... (your existing error parsing for response) ...
            error_details = "Unknown Supabase storage error" # Default
            try:
                error_json = response.json(); error_details = error_json.get("message", json.dumps(error_json))
            except: error_details = response.text[:500]
            logger.error(f"Supabase upload failed to {bucket_name}/{destination_path_in_bucket}. Status: {response.status_code}, Error: {error_details}", extra=log_extra_op)
            return None
    except Exception as e:
        logger.exception(f"Exception during Supabase upload to {bucket_name}/{destination_path_in_bucket}", extra=log_extra_op)
        return None
# --- Download ---
async def download_file_from_supabase(
    bucket_name: str, 
    source_path: str, # The S3-like key stored in your DB
    local_temp_path: str # Full path to save locally e.g. /tmp/clippilot_processing/video_uuid/downloaded_video.mp4
) -> Optional[str]:
    client = get_supabase_client()
    if not client: return None
    try:
        response = client.storage.from_(bucket_name).download(source_path)
        if isinstance(response, bytes): # Successful download returns bytes
            with open(local_temp_path, "wb") as f:
                f.write(response)
            return local_temp_path
        else: # Error occurred
            return None
    except Exception as e:
        # Log e
        return None

async def get_public_url_from_supabase(
    bucket_name: str,
    file_path: str # The S3-like key
) -> Optional[str]:
    client = get_supabase_client()
    if not client: return None
    try:
        response = client.storage.from_(bucket_name).get_public_url(file_path)
        return response
    except Exception as e:
        # Log e
        return None

async def create_signed_url_from_supabase(
    bucket_name: str,
    file_path: str,
    expires_in: int = 3600 # URL valid for 1 hour
) -> Optional[str]:
    client = get_supabase_client()
    if not client: return None
    try:
        response = client.storage.from_(bucket_name).create_signed_url(file_path, expires_in)
        # response is typically a dict like {'signedURL': '...'}
        return response.get('signedURL') if isinstance(response, dict) else None
    except Exception as e:
        # Log e
        return None

async def delete_file_from_supabase(bucket_name: str, file_paths: List[str]) -> bool:
    client = get_supabase_client()
    if not client: return False
    try:
        response = client.storage.from_(bucket_name).remove(file_paths)
        return True # Simplified, add proper error checking based on response
    except Exception as e:
        # Log e
        return False
    
if __name__ == "__main__":
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    VIDEO_BUCKET_NAME = os.getenv("VIDEO_BUCKET_NAME", "media")
    HIGHLIGHT_BUCKET_NAME = os.getenv("HIGHLIGHT_BUCKET_NAME", "media") 
    # print(f"Supabase URL: {SUPABASE_URL}")
    # print(f"Supabase Service Key: {SUPABASE_SERVICE_KEY}")
    # print(f"Video Bucket Name: {VIDEO_BUCKET_NAME}")
    # print(f"Highlight Bucket Name: {HIGHLIGHT_BUCKET_NAME}")
