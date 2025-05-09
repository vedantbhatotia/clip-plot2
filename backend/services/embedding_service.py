# services/embedding_service.py
import os
import json
import logging
import time
import glob # For finding frame files
from PIL import Image # For loading images for CLIP

import torch # PyTorch, a dependency for transformers and sentence-transformers
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel # For vision embeddings

import chromadb # Vector Database

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(video_id)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Text Embeddings
DEFAULT_TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Vision Embeddings
DEFAULT_VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
# ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/tmp/clippilot_chroma_db_dev") # Or "./data/chroma_db"
TEXT_EMBEDDINGS_COLLECTION_NAME = "clippilot_text_embeddings"
VISION_EMBEDDINGS_COLLECTION_NAME = "clippilot_vision_embeddings"

# --- ChromaDB Client ---
# Initialize ChromaDB client. In a real app, this might be managed more centrally.
# Using a persistent client so data survives restarts if path is persistent.
persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Get or create collections
try:
    text_embeddings_collection = persistent_client.get_or_create_collection(
        name=TEXT_EMBEDDINGS_COLLECTION_NAME,
        # Optionally, specify metadata for the collection if needed for HNSW indexing later
        # metadata={"hnsw:space": "cosine"} # For cosine distance
    )
    logger.info(f"ChromaDB text embeddings collection '{TEXT_EMBEDDINGS_COLLECTION_NAME}' loaded/created.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB text collection: {e}")
    text_embeddings_collection = None

try:
    vision_embeddings_collection = persistent_client.get_or_create_collection(
        name=VISION_EMBEDDINGS_COLLECTION_NAME,
        # metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"ChromaDB vision embeddings collection '{VISION_EMBEDDINGS_COLLECTION_NAME}' loaded/created.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB vision collection: {e}")
    vision_embeddings_collection = None


# --- Model Loading (Load once per service instantiation or on first use) ---
# This helps avoid reloading models on every call, which is slow.
# In a production app, you might manage model loading more robustly (e.g., singletons, dependency injection).

_text_model_instance = None
def get_text_embedding_model(model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL):
    global _text_model_instance
    if _text_model_instance is None:
        try:
            logger.info(f"Loading text embedding model: {model_name}...")
            _text_model_instance = SentenceTransformer(model_name)
            logger.info(f"Text embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load text embedding model '{model_name}': {e}")
            raise
    return _text_model_instance

_vision_model_instance = None
_vision_processor_instance = None
def get_vision_embedding_model_and_processor(model_name: str = DEFAULT_VISION_EMBEDDING_MODEL):
    global _vision_model_instance, _vision_processor_instance
    if _vision_model_instance is None or _vision_processor_instance is None:
        try:
            logger.info(f"Loading vision embedding model and processor: {model_name}...")
            # Check for CUDA availability for CLIP model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device} for CLIP model")

            _vision_model_instance = CLIPModel.from_pretrained(model_name).to(device)
            _vision_processor_instance = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"Vision embedding model and processor '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vision embedding model/processor '{model_name}': {e}")
            raise
    return _vision_model_instance, _vision_processor_instance


# --- Placeholder for DB Updates ---
def update_db_embedding_status(video_id: str, embedding_type: str, status: str, count: int = 0, error_message: str = None):
    log_extra = {'video_id': video_id}
    db_update_payload = {
        "video_id": video_id, "step": f"{embedding_type}_embedding", "status": status,
        "count": count, "error_message": error_message
    }
    logger.info(f"DB Update: {db_update_payload}", extra=log_extra)
    # TODO: Implement actual database update logic here

# --- Text Embedding Generation ---
def run_text_embedding_pipeline(video_id: str, transcript_file_path: str, processing_output_base_path: str):
    log_extra = {'video_id': video_id}
    logger.info(f"Starting text embedding pipeline for transcript: {transcript_file_path}", extra=log_extra)
    update_db_embedding_status(video_id, "text", "started")

    if not text_embeddings_collection:
        logger.error("Text embeddings collection not available. Aborting.", extra=log_extra)
        update_db_embedding_status(video_id, "text", "failed", error_message="ChromaDB collection unavailable")
        return False
        
    if not os.path.exists(transcript_file_path):
        logger.error(f"Transcript file not found at {transcript_file_path}", extra=log_extra)
        update_db_embedding_status(video_id, "text", "failed", error_message="Transcript file not found")
        return False

    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        
        segments = transcript_data.get("segments", [])
        if not segments:
            logger.warning(f"No segments found in transcript for video_id: {video_id}", extra=log_extra)
            update_db_embedding_status(video_id, "text", "completed_empty")
            return True # No segments to process, but not an error

        model = get_text_embedding_model()
        
        embeddings_to_add = []
        metadata_to_add = []
        ids_to_add = []
        
        logger.info(f"Processing {len(segments)} text segments for embedding...", extra=log_extra)
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue

            start_time = segment.get("start")
            end_time = segment.get("end")
            
            embedding = model.encode(text, convert_to_tensor=False).tolist() # Get NumPy array then to list

            segment_id = f"{video_id}_text_{i}" # Unique ID for this embedding

            embeddings_to_add.append(embedding)
            metadata_to_add.append({
                "video_uuid": video_id,
                "text_content": text,
                "start_time": start_time,
                "end_time": end_time,
                "source_type": "transcript_segment"
            })
            ids_to_add.append(segment_id)

        if embeddings_to_add:
            text_embeddings_collection.add(
                embeddings=embeddings_to_add,
                metadatas=metadata_to_add,
                ids=ids_to_add
            )
            logger.info(f"Added {len(embeddings_to_add)} text embeddings to ChromaDB for video_id: {video_id}", extra=log_extra)
        
        update_db_embedding_status(video_id, "text", "completed", count=len(embeddings_to_add))
        return True

    except Exception as e:
        logger.exception(f"Error during text embedding generation for {video_id}", extra=log_extra)
        update_db_embedding_status(video_id, "text", "failed", error_message=str(e))
        return False

# --- Vision Embedding Generation ---
def run_vision_embedding_pipeline(video_id: str, frames_directory_path: str, processing_output_base_path: str):
    log_extra = {'video_id': video_id}
    logger.info(f"Starting vision embedding pipeline for frames in: {frames_directory_path}", extra=log_extra)
    update_db_embedding_status(video_id, "vision", "started")

    if not vision_embeddings_collection:
        logger.error("Vision embeddings collection not available. Aborting.", extra=log_extra)
        update_db_embedding_status(video_id, "vision", "failed", error_message="ChromaDB collection unavailable")
        return False

    if not os.path.isdir(frames_directory_path):
        logger.error(f"Frames directory not found at {frames_directory_path}", extra=log_extra)
        update_db_embedding_status(video_id, "vision", "failed", error_message="Frames directory not found")
        return False

    try:
        # Check for CUDA availability for CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = get_vision_embedding_model_and_processor()
        
        frame_files = sorted(glob.glob(os.path.join(frames_directory_path, "frame_*.jpg"))) # Assuming JPG
        if not frame_files:
            logger.warning(f"No frame files found in {frames_directory_path} for video_id: {video_id}", extra=log_extra)
            update_db_embedding_status(video_id, "vision", "completed_empty")
            return True

        embeddings_to_add = []
        metadata_to_add = []
        ids_to_add = []

        logger.info(f"Processing {len(frame_files)} frames for vision embedding...", extra=log_extra)
        for i, frame_path in enumerate(frame_files):
            try:
                image = Image.open(frame_path)
                inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad(): # Important for inference
                    image_features = model.get_image_features(**inputs)
                
                embedding = image_features.cpu().numpy().flatten().tolist() # Get NumPy array, flatten, then to list

                # Attempt to extract timestamp from filename (e.g., frame_0001.jpg -> t=0 if 1fps, t=1 if from t=1)
                # This depends on your FRAME_RATE_STR from media_processor
                # For simplicity, using frame index as a proxy for time or sequence.
                # A more robust way would be to store exact timestamps when frames are extracted.
                frame_basename = os.path.basename(frame_path)
                try:
                    # Assuming FRAME_RATE_STR was like "1/1" (1 fps, frame_0001 at t=0)
                    # or "fps=N" where frame_0001 is at t=0
                    # If frame_0001.jpg is t=0, then frame_xxxx.jpg is t = (xxxx-1) / FPS
                    # This is a simplification; real frame timestamps might be more complex.
                    frame_number = int(frame_basename.split('_')[1].split('.')[0])
                    # If FRAME_RATE_STR was "1/X" (1 frame every X seconds), timestamp = (frame_number - 1) * X
                    # If FRAME_RATE_STR was "N/1" (N frames per second), timestamp = (frame_number - 1) / N
                    # Let's assume 1 FPS for now for simplicity from frame_xxxx.jpg -> t = xxxx-1
                    frame_timestamp_sec = float(frame_number - 1) # Simplistic timestamp
                except:
                    frame_timestamp_sec = float(i) # Fallback to index if parsing fails

                frame_id = f"{video_id}_vision_{i}"

                embeddings_to_add.append(embedding)
                metadata_to_add.append({
                    "video_uuid": video_id,
                    "frame_filename": frame_basename,
                    "frame_timestamp_sec": frame_timestamp_sec, # Approximate timestamp
                    "source_type": "video_frame"
                })
                ids_to_add.append(frame_id)
            except Exception as frame_e:
                logger.error(f"Failed to process frame {frame_path}: {frame_e}", extra=log_extra)
                continue # Skip this frame

        if embeddings_to_add:
            vision_embeddings_collection.add(
                embeddings=embeddings_to_add,
                metadatas=metadata_to_add,
                ids=ids_to_add
            )
            logger.info(f"Added {len(embeddings_to_add)} vision embeddings to ChromaDB for video_id: {video_id}", extra=log_extra)

        update_db_embedding_status(video_id, "vision", "completed", count=len(embeddings_to_add))
        return True

    except Exception as e:
        logger.exception(f"Error during vision embedding generation for {video_id}", extra=log_extra)
        update_db_embedding_status(video_id, "vision", "failed", error_message=str(e))
        return False
