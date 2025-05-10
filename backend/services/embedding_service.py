# # services/embedding_service.py
# import os
# import json
# import logging
# import time
# import glob # For finding frame files
# from PIL import Image # For loading images for CLIP

# import torch # PyTorch, a dependency for transformers and sentence-transformers
# from sentence_transformers import SentenceTransformer
# from transformers import CLIPProcessor, CLIPModel # For vision embeddings

# import chromadb # Vector Database

# # Configure basic logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [%(video_id)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # --- Configuration ---
# # Text Embeddings
# DEFAULT_TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# # Vision Embeddings
# DEFAULT_VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
# # ChromaDB
# CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/tmp/clippilot_chroma_db_dev") # Or "./data/chroma_db"
# TEXT_EMBEDDINGS_COLLECTION_NAME = "clippilot_text_embeddings"
# VISION_EMBEDDINGS_COLLECTION_NAME = "clippilot_vision_embeddings"

# # --- ChromaDB Client ---
# # Initialize ChromaDB client. In a real app, this might be managed more centrally.
# # Using a persistent client so data survives restarts if path is persistent.
# persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# # Get or create collections
# try:
#     text_embeddings_collection = persistent_client.get_or_create_collection(
#         name=TEXT_EMBEDDINGS_COLLECTION_NAME,
#         # Optionally, specify metadata for the collection if needed for HNSW indexing later
#         # metadata={"hnsw:space": "cosine"} # For cosine distance
#     )
#     logger.info(f"ChromaDB text embeddings collection '{TEXT_EMBEDDINGS_COLLECTION_NAME}' loaded/created.")
# except Exception as e:
#     logger.error(f"Error initializing ChromaDB text collection: {e}")
#     text_embeddings_collection = None

# try:
#     vision_embeddings_collection = persistent_client.get_or_create_collection(
#         name=VISION_EMBEDDINGS_COLLECTION_NAME,
#         # metadata={"hnsw:space": "cosine"}
#     )
#     logger.info(f"ChromaDB vision embeddings collection '{VISION_EMBEDDINGS_COLLECTION_NAME}' loaded/created.")
# except Exception as e:
#     logger.error(f"Error initializing ChromaDB vision collection: {e}")
#     vision_embeddings_collection = None


# # --- Model Loading (Load once per service instantiation or on first use) ---
# # This helps avoid reloading models on every call, which is slow.
# # In a production app, you might manage model loading more robustly (e.g., singletons, dependency injection).

# _text_model_instance = None
# def get_text_embedding_model(model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL):
#     global _text_model_instance
#     if _text_model_instance is None:
#         try:
#             logger.info(f"Loading text embedding model: {model_name}...")
#             _text_model_instance = SentenceTransformer(model_name)
#             logger.info(f"Text embedding model '{model_name}' loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load text embedding model '{model_name}': {e}")
#             raise
#     return _text_model_instance

# _vision_model_instance = None
# _vision_processor_instance = None
# def get_vision_embedding_model_and_processor(model_name: str = DEFAULT_VISION_EMBEDDING_MODEL):
#     global _vision_model_instance, _vision_processor_instance
#     if _vision_model_instance is None or _vision_processor_instance is None:
#         try:
#             logger.info(f"Loading vision embedding model and processor: {model_name}...")
#             # Check for CUDA availability for CLIP model
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"Using device: {device} for CLIP model")

#             _vision_model_instance = CLIPModel.from_pretrained(model_name).to(device)
#             _vision_processor_instance = CLIPProcessor.from_pretrained(model_name)
#             logger.info(f"Vision embedding model and processor '{model_name}' loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load vision embedding model/processor '{model_name}': {e}")
#             raise
#     return _vision_model_instance, _vision_processor_instance


# # --- Placeholder for DB Updates ---
# def update_db_embedding_status(video_id: str, embedding_type: str, status: str, count: int = 0, error_message: str = None):
#     log_extra = {'video_id': video_id}
#     db_update_payload = {
#         "video_id": video_id, "step": f"{embedding_type}_embedding", "status": status,
#         "count": count, "error_message": error_message
#     }
#     logger.info(f"DB Update: {db_update_payload}", extra=log_extra)
#     # TODO: Implement actual database update logic here

# # --- Text Embedding Generation ---
# def run_text_embedding_pipeline(video_id: str, transcript_file_path: str, processing_output_base_path: str):
#     log_extra = {'video_id': video_id}
#     logger.info(f"Starting text embedding pipeline for transcript: {transcript_file_path}", extra=log_extra)
#     update_db_embedding_status(video_id, "text", "started")

#     if not text_embeddings_collection:
#         logger.error("Text embeddings collection not available. Aborting.", extra=log_extra)
#         update_db_embedding_status(video_id, "text", "failed", error_message="ChromaDB collection unavailable")
#         return False
        
#     if not os.path.exists(transcript_file_path):
#         logger.error(f"Transcript file not found at {transcript_file_path}", extra=log_extra)
#         update_db_embedding_status(video_id, "text", "failed", error_message="Transcript file not found")
#         return False

#     try:
#         with open(transcript_file_path, "r", encoding="utf-8") as f:
#             transcript_data = json.load(f)
        
#         segments = transcript_data.get("segments", [])
#         if not segments:
#             logger.warning(f"No segments found in transcript for video_id: {video_id}", extra=log_extra)
#             update_db_embedding_status(video_id, "text", "completed_empty")
#             return True # No segments to process, but not an error

#         model = get_text_embedding_model()
        
#         embeddings_to_add = []
#         metadata_to_add = []
#         ids_to_add = []
        
#         logger.info(f"Processing {len(segments)} text segments for embedding...", extra=log_extra)
#         for i, segment in enumerate(segments):
#             text = segment.get("text", "").strip()
#             if not text:
#                 continue

#             start_time = segment.get("start")
#             end_time = segment.get("end")
            
#             embedding = model.encode(text, convert_to_tensor=False).tolist() # Get NumPy array then to list

#             segment_id = f"{video_id}_text_{i}" # Unique ID for this embedding

#             embeddings_to_add.append(embedding)
#             metadata_to_add.append({
#                 "video_uuid": video_id,
#                 "text_content": text,
#                 "start_time": start_time,
#                 "end_time": end_time,
#                 "source_type": "transcript_segment"
#             })
#             ids_to_add.append(segment_id)

#         if embeddings_to_add:
#             text_embeddings_collection.add(
#                 embeddings=embeddings_to_add,
#                 metadatas=metadata_to_add,
#                 ids=ids_to_add
#             )
#             logger.info(f"Added {len(embeddings_to_add)} text embeddings to ChromaDB for video_id: {video_id}", extra=log_extra)
        
#         update_db_embedding_status(video_id, "text", "completed", count=len(embeddings_to_add))
#         return True

#     except Exception as e:
#         logger.exception(f"Error during text embedding generation for {video_id}", extra=log_extra)
#         update_db_embedding_status(video_id, "text", "failed", error_message=str(e))
#         return False

# # --- Vision Embedding Generation ---
# def run_vision_embedding_pipeline(video_id: str, frames_directory_path: str, processing_output_base_path: str):
#     log_extra = {'video_id': video_id}
#     logger.info(f"Starting vision embedding pipeline for frames in: {frames_directory_path}", extra=log_extra)
#     update_db_embedding_status(video_id, "vision", "started")

#     if not vision_embeddings_collection:
#         logger.error("Vision embeddings collection not available. Aborting.", extra=log_extra)
#         update_db_embedding_status(video_id, "vision", "failed", error_message="ChromaDB collection unavailable")
#         return False

#     if not os.path.isdir(frames_directory_path):
#         logger.error(f"Frames directory not found at {frames_directory_path}", extra=log_extra)
#         update_db_embedding_status(video_id, "vision", "failed", error_message="Frames directory not found")
#         return False

#     try:
#         # Check for CUDA availability for CLIP model
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model, processor = get_vision_embedding_model_and_processor()
        
#         frame_files = sorted(glob.glob(os.path.join(frames_directory_path, "frame_*.jpg"))) # Assuming JPG
#         if not frame_files:
#             logger.warning(f"No frame files found in {frames_directory_path} for video_id: {video_id}", extra=log_extra)
#             update_db_embedding_status(video_id, "vision", "completed_empty")
#             return True

#         embeddings_to_add = []
#         metadata_to_add = []
#         ids_to_add = []

#         logger.info(f"Processing {len(frame_files)} frames for vision embedding...", extra=log_extra)
#         for i, frame_path in enumerate(frame_files):
#             try:
#                 image = Image.open(frame_path)
#                 inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
                
#                 with torch.no_grad(): # Important for inference
#                     image_features = model.get_image_features(**inputs)
                
#                 embedding = image_features.cpu().numpy().flatten().tolist() # Get NumPy array, flatten, then to list

#                 # Attempt to extract timestamp from filename (e.g., frame_0001.jpg -> t=0 if 1fps, t=1 if from t=1)
#                 # This depends on your FRAME_RATE_STR from media_processor
#                 # For simplicity, using frame index as a proxy for time or sequence.
#                 # A more robust way would be to store exact timestamps when frames are extracted.
#                 frame_basename = os.path.basename(frame_path)
#                 try:
#                     # Assuming FRAME_RATE_STR was like "1/1" (1 fps, frame_0001 at t=0)
#                     # or "fps=N" where frame_0001 is at t=0
#                     # If frame_0001.jpg is t=0, then frame_xxxx.jpg is t = (xxxx-1) / FPS
#                     # This is a simplification; real frame timestamps might be more complex.
#                     frame_number = int(frame_basename.split('_')[1].split('.')[0])
#                     # If FRAME_RATE_STR was "1/X" (1 frame every X seconds), timestamp = (frame_number - 1) * X
#                     # If FRAME_RATE_STR was "N/1" (N frames per second), timestamp = (frame_number - 1) / N
#                     # Let's assume 1 FPS for now for simplicity from frame_xxxx.jpg -> t = xxxx-1
#                     frame_timestamp_sec = float(frame_number - 1) # Simplistic timestamp
#                 except:
#                     frame_timestamp_sec = float(i) # Fallback to index if parsing fails

#                 frame_id = f"{video_id}_vision_{i}"

#                 embeddings_to_add.append(embedding)
#                 metadata_to_add.append({
#                     "video_uuid": video_id,
#                     "frame_filename": frame_basename,
#                     "frame_timestamp_sec": frame_timestamp_sec, # Approximate timestamp
#                     "source_type": "video_frame"
#                 })
#                 ids_to_add.append(frame_id)
#             except Exception as frame_e:
#                 logger.error(f"Failed to process frame {frame_path}: {frame_e}", extra=log_extra)
#                 continue # Skip this frame

#         if embeddings_to_add:
#             vision_embeddings_collection.add(
#                 embeddings=embeddings_to_add,
#                 metadatas=metadata_to_add,
#                 ids=ids_to_add
#             )
#             logger.info(f"Added {len(embeddings_to_add)} vision embeddings to ChromaDB for video_id: {video_id}", extra=log_extra)

#         update_db_embedding_status(video_id, "vision", "completed", count=len(embeddings_to_add))
#         return True

#     except Exception as e:
#         logger.exception(f"Error during vision embedding generation for {video_id}", extra=log_extra)
#         update_db_embedding_status(video_id, "vision", "failed", error_message=str(e))
#         return False



# # services/embedding_service.py
# import os
# import json
# import logging # Standard logging
# import time
# import glob
# from PIL import Image

# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import CLIPProcessor, CLIPModel
# import chromadb

# # --- Loggers ---
# # Logger for general module setup, model loading, etc. (does not expect video_id in format)
# module_logger = logging.getLogger(f"app.{__name__}") # e.g., app.services.embedding_service
# # Logger for messages specific to processing a video (will be used with `extra={'video_id': ...}`)
# processing_logger = logging.getLogger(f"processing.{__name__}")

# # --- Configuration ---
# DEFAULT_TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# DEFAULT_VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
# # CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/tmp/clippilot_chroma_db_dev")
# CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.getcwd(), "data", "chroma_db_dev"))
# TEXT_EMBEDDINGS_COLLECTION_NAME = "clippilot_text_embeddings"
# VISION_EMBEDDINGS_COLLECTION_NAME = "clippilot_vision_embeddings"

# # --- ChromaDB Client (Uses module_logger) ---
# persistent_client = None
# text_embeddings_collection = None
# vision_embeddings_collection = None

# try:
#     module_logger.info(f"Attempting to initialize ChromaDB PersistentClient at path: {CHROMA_DB_PATH}")
#     persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
#     module_logger.info(f"ChromaDB PersistentClient initialized successfully for path: {CHROMA_DB_PATH}")

#     module_logger.info(f"Getting/Creating text collection: {TEXT_EMBEDDINGS_COLLECTION_NAME}")
#     text_embeddings_collection = persistent_client.get_or_create_collection(name=TEXT_EMBEDDINGS_COLLECTION_NAME)
#     module_logger.info(f"Text collection '{TEXT_EMBEDDINGS_COLLECTION_NAME}' ready.")

#     module_logger.info(f"Getting/Creating vision collection: {VISION_EMBEDDINGS_COLLECTION_NAME}")
#     vision_embeddings_collection = persistent_client.get_or_create_collection(name=VISION_EMBEDDINGS_COLLECTION_NAME)
#     module_logger.info(f"Vision collection '{VISION_EMBEDDINGS_COLLECTION_NAME}' ready.")

# except Exception as e:
#     module_logger.exception(f"CRITICAL: Failed to initialize ChromaDB client or collections using path {CHROMA_DB_PATH}. Embedding service may not function.")
#     # In a real app, you might want to raise this or have a health check fail


# # --- Model Loading (Uses module_logger) ---
# _text_model_instance = None
# def get_text_embedding_model(model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL):
#     global _text_model_instance
#     if _text_model_instance is None:
#         try:
#             module_logger.info(f"Loading text embedding model: {model_name}...")
#             _text_model_instance = SentenceTransformer(model_name)
#             module_logger.info(f"Text embedding model '{model_name}' loaded successfully.")
#         except Exception as e:
#             module_logger.exception(f"Failed to load text embedding model '{model_name}'")
#             raise # Re-raise after logging
#     return _text_model_instance

# _vision_model_instance = None
# _vision_processor_instance = None
# def get_vision_embedding_model_and_processor(model_name: str = DEFAULT_VISION_EMBEDDING_MODEL):
#     global _vision_model_instance, _vision_processor_instance
#     if _vision_model_instance is None or _vision_processor_instance is None:
#         try:
#             module_logger.info(f"Loading vision embedding model and processor: {model_name}...")
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             module_logger.info(f"Using device: {device} for CLIP model")
#             _vision_model_instance = CLIPModel.from_pretrained(model_name).to(device)
#             _vision_processor_instance = CLIPProcessor.from_pretrained(model_name)
#             module_logger.info(f"Vision embedding model and processor '{model_name}' loaded successfully.")
#         except Exception as e:
#             module_logger.exception(f"Failed to load vision embedding model/processor '{model_name}'")
#             raise # Re-raise after logging
#     return _vision_model_instance, _vision_processor_instance


# # --- Placeholder for DB Updates (Uses processing_logger) ---
# def update_db_embedding_status(video_id: str, embedding_type: str, status: str, count: int = 0, error_message: str = None):
#     log_extra = {'video_id': video_id}
#     # The 'name' in the log format will show 'processing.services.embedding_service'
#     # The video_id from 'extra' won't appear in the formatted string with the current basicConfig in main.py,
#     # but it's available on the LogRecord object if a custom formatter/filter is used.
#     processing_logger.info(
#         f"DB Update: video_id='{video_id}', status='{status}', type='{embedding_type}', count={count}, error='{error_message}'"
#         # Removed 'extra=log_extra' here if the format string doesn't use it, to avoid confusion.
#         # Or, keep it if you plan to add a custom formatter that uses record.video_id.
#         # For now, explicitly including video_id in the message string for clarity with basicConfig.
#     )

# # --- Text Embedding Generation (Uses processing_logger) ---
# def run_text_embedding_pipeline(video_id: str, transcript_file_path: str, processing_output_base_path: str):
#     log_extra = {'video_id': video_id} # For potential custom formatters/filters
#     processing_logger.info(f"Starting text embedding pipeline for video_id: {video_id}, transcript: {transcript_file_path}")

#     if not text_embeddings_collection:
#         processing_logger.error(f"video_id: {video_id} - Text embeddings collection not available. Aborting text embedding.")
#         update_db_embedding_status(video_id, "text", "failed", error_message="ChromaDB collection unavailable")
#         return False
        
#     if not os.path.exists(transcript_file_path):
#         processing_logger.error(f"video_id: {video_id} - Transcript file not found at {transcript_file_path}")
#         update_db_embedding_status(video_id, "text", "failed", error_message="Transcript file not found")
#         return False

#     try:
#         with open(transcript_file_path, "r", encoding="utf-8") as f:
#             transcript_data = json.load(f)
        
#         segments = transcript_data.get("segments", [])
#         if not segments:
#             processing_logger.warning(f"video_id: {video_id} - No segments found in transcript.")
#             update_db_embedding_status(video_id, "text", "completed_empty")
#             return True

#         model = get_text_embedding_model()
        
#         embeddings_to_add = []
#         metadata_to_add = []
#         ids_to_add = []
        
#         processing_logger.info(f"video_id: {video_id} - Processing {len(segments)} text segments for embedding...")
#         for i, segment in enumerate(segments):
#             text = segment.get("text", "").strip()
#             if not text:
#                 continue
#             start_time = segment.get("start")
#             end_time = segment.get("end")
#             embedding = model.encode(text, convert_to_tensor=False).tolist()
#             segment_id = f"{video_id}_text_{i}"
#             embeddings_to_add.append(embedding)
#             metadata_to_add.append({
#                 "video_uuid": video_id, "text_content": text, "start_time": start_time,
#                 "end_time": end_time, "source_type": "transcript_segment"
#             })
#             ids_to_add.append(segment_id)

#         if embeddings_to_add:
#             text_embeddings_collection.add(embeddings=embeddings_to_add, metadatas=metadata_to_add, ids=ids_to_add)
#             processing_logger.info(f"video_id: {video_id} - Added {len(embeddings_to_add)} text embeddings to ChromaDB.")
        
#         update_db_embedding_status(video_id, "text", "completed", count=len(embeddings_to_add))
#         return True

#     except Exception as e:
#         processing_logger.exception(f"video_id: {video_id} - Error during text embedding generation.")
#         update_db_embedding_status(video_id, "text", "failed", error_message=str(e))
#         return False

# # --- Vision Embedding Generation (Uses processing_logger) ---
# # def run_vision_embedding_pipeline(video_id: str, frames_directory_path: str, processing_output_base_path: str):
# #     log_extra = {'video_id': video_id} # For potential custom formatters/filters
# #     processing_logger.info(f"Starting vision embedding pipeline for video_id: {video_id}, frames in: {frames_directory_path}")

# #     if not vision_embeddings_collection:
# #         processing_logger.error(f"video_id: {video_id} - Vision embeddings collection not available. Aborting vision embedding.")
# #         update_db_embedding_status(video_id, "vision", "failed", error_message="ChromaDB collection unavailable")
# #         return False

# #     if not os.path.isdir(frames_directory_path):
# #         processing_logger.error(f"video_id: {video_id} - Frames directory not found at {frames_directory_path}")
# #         update_db_embedding_status(video_id, "vision", "failed", error_message="Frames directory not found")
# #         return False

# #     try:
# #         device = "cuda" if torch.cuda.is_available() else "cpu"
# #         model, processor = get_vision_embedding_model_and_processor()
        
# #         frame_files = sorted(glob.glob(os.path.join(frames_directory_path, "frame_*.jpg")))
# #         if not frame_files:
# #             processing_logger.warning(f"video_id: {video_id} - No frame files found in {frames_directory_path}.")
# #             update_db_embedding_status(video_id, "vision", "completed_empty")
# #             return True

# #         embeddings_to_add = []
# #         metadata_to_add = []
# #         ids_to_add = []

# #         processing_logger.info(f"video_id: {video_id} - Processing {len(frame_files)} frames for vision embedding...")
# #         for i, frame_path in enumerate(frame_files):
# #             try:
# #                 image = Image.open(frame_path)
# #                 inputs = processor(text=None, images=image, return_t_ensors="pt", padding=True).to(device)
# #                 with torch.no_grad():
# #                     image_features = model.get_image_features(**inputs)
# #                 embedding = image_features.cpu().numpy().flatten().tolist()
# #                 frame_basename = os.path.basename(frame_path)
# #                 try:
# #                     frame_number = int(frame_basename.split('_')[1].split('.')[0])
# #                     frame_timestamp_sec = float(frame_number - 1)
# #                 except:
# #                     frame_timestamp_sec = float(i) # Fallback
# #                 frame_id = f"{video_id}_vision_{i}"
# #                 embeddings_to_add.append(embedding)
# #                 metadata_to_add.append({
# #                     "video_uuid": video_id, "frame_filename": frame_basename,
# #                     "frame_timestamp_sec": frame_timestamp_sec, "source_type": "video_frame"
# #                 })
# #                 ids_to_add.append(frame_id)
# #             except Exception as frame_e:
# #                 processing_logger.error(f"video_id: {video_id} - Failed to process frame {frame_path}: {frame_e}")
# #                 continue

# #         if embeddings_to_add:
# #             vision_embeddings_collection.add(embeddings=embeddings_to_add, metadatas=metadata_to_add, ids=ids_to_add)
# #             processing_logger.info(f"video_id: {video_id} - Added {len(embeddings_to_add)} vision embeddings to ChromaDB.")

# #         update_db_embedding_status(video_id, "vision", "completed", count=len(embeddings_to_add))
# #         return True

# #     except Exception as e:
# #         processing_logger.exception(f"video_id: {video_id} - Error during vision embedding generation.")
# #         update_db_embedding_status(video_id, "vision", "failed", error_message=str(e))
# #         return False

# def run_vision_embedding_pipeline(video_id: str, frames_directory_path: str, processing_output_base_path: str):
#     log_extra = {'video_id': video_id} # For potential custom formatters/filters
#     processing_logger.info(f"Starting vision embedding pipeline for video_id: {video_id}, frames in: {frames_directory_path}")

#     if not vision_embeddings_collection:
#         processing_logger.error(f"video_id: {video_id} - Vision embeddings collection not available. Aborting vision embedding.")
#         update_db_embedding_status(video_id, "vision", "failed", error_message="ChromaDB collection unavailable")
#         return False

#     if not os.path.isdir(frames_directory_path):
#         processing_logger.error(f"video_id: {video_id} - Frames directory not found at {frames_directory_path}")
#         update_db_embedding_status(video_id, "vision", "failed", error_message="Frames directory not found")
#         return False

#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model, processor = get_vision_embedding_model_and_processor()
        
#         frame_files = sorted(glob.glob(os.path.join(frames_directory_path, "frame_*.jpg")))
#         if not frame_files:
#             processing_logger.warning(f"video_id: {video_id} - No frame files found in {frames_directory_path}.")
#             update_db_embedding_status(video_id, "vision", "completed_empty")
#             return True

#         embeddings_to_add = []
#         metadata_to_add = []
#         ids_to_add = []

#         processing_logger.info(f"video_id: {video_id} - Processing {len(frame_files)} frames for vision embedding...")
#         for i, frame_path in enumerate(frame_files):
#             try:
#                 image = Image.open(frame_path)
#                 # --- THIS IS THE CORRECTED LINE ---
#                 inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
#                 # --- END OF CORRECTION ---
                
#                 with torch.no_grad():
#                     image_features = model.get_image_features(**inputs)
                
#                 embedding = image_features.cpu().numpy().flatten().tolist()
#                 frame_basename = os.path.basename(frame_path)
#                 try:
#                     frame_number = int(frame_basename.split('_')[1].split('.')[0])
#                     frame_timestamp_sec = float(frame_number - 1)
#                 except:
#                     frame_timestamp_sec = float(i) # Fallback
#                 frame_id = f"{video_id}_vision_{i}"
#                 embeddings_to_add.append(embedding)
#                 metadata_to_add.append({
#                     "video_uuid": video_id, "frame_filename": frame_basename,
#                     "frame_timestamp_sec": frame_timestamp_sec, "source_type": "video_frame"
#                 })
#                 ids_to_add.append(frame_id)
#             except Exception as frame_e:
#                 processing_logger.error(f"video_id: {video_id} - Failed to process frame {frame_path}: {frame_e}") # Log the specific frame error
#                 # Optionally, re-raise if you want the whole pipeline to stop on one frame error,
#                 # or log and continue as it is now.
#                 continue # Skip this frame

#         if embeddings_to_add:
#             vision_embeddings_collection.add(embeddings=embeddings_to_add, metadatas=metadata_to_add, ids=ids_to_add)
#             processing_logger.info(f"video_id: {video_id} - Added {len(embeddings_to_add)} vision embeddings to ChromaDB.")
#         elif len(frame_files) > 0 : # If there were frames but none were processable
#              processing_logger.warning(f"video_id: {video_id} - No vision embeddings were generated, though {len(frame_files)} frames were found. Check frame processing errors.")


#         update_db_embedding_status(video_id, "vision", "completed", count=len(embeddings_to_add))
#         return True

#     except Exception as e:
#         processing_logger.exception(f"video_id: {video_id} - Error during vision embedding generation.")
#         update_db_embedding_status(video_id, "vision", "failed", error_message=str(e))
#         return False
# # --- Test Block (Uses module_logger for its own setup messages) ---
# if __name__ == "__main__":
#     # This setup ensures that the basicConfig from main.py (if this script is run
#     # in an environment where main.py's config would have been applied) is used,
#     # or it sets up a default if run completely standalone.
#     if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
#             datefmt='%Y-%m-%d %H:%M:%S'
#         )

#     module_logger.info("Running embedding_service.py directly for testing...")
    
#     test_video_id = "embedding-test-direct-001"
#     # For test logs that are video_id specific, include it in the message string
#     # as the 'extra' dict won't be automatically formatted by basicConfig.
    
#     test_processing_base_dir = f"/tmp/clippilot_test_embeddings/{test_video_id}" # Renamed for clarity
#     # Corrected path for transcript file within the test_processing_base_dir
#     test_transcript_dir = os.path.join(test_processing_base_dir, "transcripts")
#     os.makedirs(test_transcript_dir, exist_ok=True)
#     test_transcript_file = os.path.join(test_transcript_dir, "transcript_whisper.json")

#     module_logger.info(f"--- TESTING TEXT EMBEDDINGS for video_id: {test_video_id} ---")
#     dummy_transcript = {
#         "text": "Hello world. This is a test.",
#         "segments": [
#             {"id": 0, "start": 0.0, "end": 1.5, "text": " Hello world."},
#             {"id": 1, "start": 1.5, "end": 3.0, "text": " This is a test."}
#         ]
#     }
#     with open(test_transcript_file, "w") as f:
#         json.dump(dummy_transcript, f)
#     module_logger.info(f"Dummy transcript created at: {test_transcript_file}")
    
#     if os.path.exists(test_transcript_file):
#         text_success = run_text_embedding_pipeline(test_video_id, test_transcript_file, test_processing_base_dir)
#         module_logger.info(f"Text embedding pipeline success for video_id {test_video_id}: {text_success}")
#         if text_success and text_embeddings_collection:
#              results = text_embeddings_collection.get(ids=[f"{test_video_id}_text_0"], include=["metadatas", "documents"])
#              module_logger.info(f"ChromaDB get result for text (video_id {test_video_id}): IDs - {results['ids']}")
#     else:
#         module_logger.error(f"Dummy transcript file {test_transcript_file} not found for testing.")

#     module_logger.info(f"\n--- TESTING VISION EMBEDDINGS for video_id: {test_video_id} ---")
#     test_frames_dir = os.path.join(test_processing_base_dir, "frames")
#     os.makedirs(test_frames_dir, exist_ok=True)

#     try:
#         dummy_image = Image.new('RGB', (60, 30), color = 'white')
#         dummy_image.save(os.path.join(test_frames_dir, "frame_0001.jpg"))
#         dummy_image.save(os.path.join(test_frames_dir, "frame_0002.jpg"))
#         module_logger.info(f"Created dummy frame files in {test_frames_dir}")
#     except Exception as e:
#         module_logger.error(f"Could not create dummy frame image: {e}. Vision embedding test might fail.")

#     if os.path.isdir(test_frames_dir) and len(os.listdir(test_frames_dir)) > 0:
#         vision_success = run_vision_embedding_pipeline(test_video_id, test_frames_dir, test_processing_base_dir)
#         module_logger.info(f"Vision embedding pipeline success for video_id {test_video_id}: {vision_success}")
#         if vision_success and vision_embeddings_collection:
#             results = vision_embeddings_collection.get(ids=[f"{test_video_id}_vision_0"], include=["metadatas"])
#             module_logger.info(f"ChromaDB get result for vision (video_id {test_video_id}): IDs - {results['ids']}")
#     else:
#         module_logger.error(f"Dummy frames directory {test_frames_dir} is empty or not found.")







































































# backend/services/embedding_service.py
import os
import json
import logging
import time
import glob
from PIL import Image
import asyncio # For asyncio.run in __main__
from typing import Optional # For type hinting

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import chromadb

# --- Import Database Service ---
from . import database_service # Assuming services is a package
from .database_service import VideoProcessingStatus, get_db_session

# --- Loggers ---
module_logger = logging.getLogger(f"app.{__name__}")
processing_logger = logging.getLogger(f"processing.{__name__}")

# --- Configuration ---
DEFAULT_TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.getcwd(), "data", "chroma_db_dev")) # Path relative to CWD
TEXT_EMBEDDINGS_COLLECTION_NAME = "clippilot_text_embeddings"
VISION_EMBEDDINGS_COLLECTION_NAME = "clippilot_vision_embeddings"

# --- ChromaDB Client (Uses module_logger) ---
persistent_client = None
text_embeddings_collection = None
vision_embeddings_collection = None

try:
    # Ensure data directory exists if using relative path for ChromaDB
    if "data/chroma_db_dev" in CHROMA_DB_PATH: # A bit of a heuristic for the default relative path
        os.makedirs(os.path.dirname(CHROMA_DB_PATH), exist_ok=True)

    module_logger.info(f"Attempting to initialize ChromaDB PersistentClient at path: {CHROMA_DB_PATH}")
    persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    module_logger.info(f"ChromaDB PersistentClient initialized successfully for path: {CHROMA_DB_PATH}")

    module_logger.info(f"Getting/Creating text collection: {TEXT_EMBEDDINGS_COLLECTION_NAME}")
    text_embeddings_collection = persistent_client.get_or_create_collection(name=TEXT_EMBEDDINGS_COLLECTION_NAME)
    module_logger.info(f"Text collection '{TEXT_EMBEDDINGS_COLLECTION_NAME}' ready.")

    module_logger.info(f"Getting/Creating vision collection: {VISION_EMBEDDINGS_COLLECTION_NAME}")
    vision_embeddings_collection = persistent_client.get_or_create_collection(name=VISION_EMBEDDINGS_COLLECTION_NAME)
    module_logger.info(f"Vision collection '{VISION_EMBEDDINGS_COLLECTION_NAME}' ready.")

except Exception as e:
    module_logger.exception(f"CRITICAL: Failed to initialize ChromaDB client or collections using path {CHROMA_DB_PATH}. Embedding service may not function.")


# --- Model Loading (Synchronous, as it's CPU/GPU bound and typically once) ---
_text_model_instance = None
def get_text_embedding_model(model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL):
    global _text_model_instance
    if _text_model_instance is None:
        try:
            module_logger.info(f"Loading text embedding model: {model_name}...")
            _text_model_instance = SentenceTransformer(model_name)
            module_logger.info(f"Text embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            module_logger.exception(f"Failed to load text embedding model '{model_name}'")
            raise
    return _text_model_instance

_vision_model_instance = None
_vision_processor_instance = None
def get_vision_embedding_model_and_processor(model_name: str = DEFAULT_VISION_EMBEDDING_MODEL):
    global _vision_model_instance, _vision_processor_instance
    if _vision_model_instance is None or _vision_processor_instance is None:
        try:
            module_logger.info(f"Loading vision embedding model and processor: {model_name}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            module_logger.info(f"Using device: {device} for CLIP model")
            _vision_model_instance = CLIPModel.from_pretrained(model_name).to(device)
            _vision_processor_instance = CLIPProcessor.from_pretrained(model_name)
            module_logger.info(f"Vision embedding model and processor '{model_name}' loaded successfully.")
        except Exception as e:
            module_logger.exception(f"Failed to load vision embedding model/processor '{model_name}'")
            raise
    return _vision_model_instance, _vision_processor_instance

# --- Text Embedding Generation ---
async def run_text_embedding_pipeline(video_id: str, transcript_file_path: str, processing_output_base_path: str) -> bool:
    log_extra = {'video_id': video_id}
    processing_logger.info(f"video_id: {video_id} - Starting text embedding pipeline for transcript: {transcript_file_path}", extra=log_extra)

    async with get_db_session() as session:
        try:
            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.TEXT_EMBEDDING
            )

            if not text_embeddings_collection:
                error_msg = "ChromaDB text_embeddings_collection not available."
                processing_logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return False
            
            if not os.path.exists(transcript_file_path):
                error_msg = f"Transcript file not found at {transcript_file_path}"
                processing_logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return False

            with open(transcript_file_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            
            segments = transcript_data.get("segments", [])
            if not segments:
                processing_logger.warning(f"video_id: {video_id} - No segments found in transcript.", extra=log_extra)
                # Still update status to TEXT_EMBEDDED as there was nothing to process
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.TEXT_EMBEDDED, "No segments for text embedding")
                return True # Technically successful, just no embeddings

            model = get_text_embedding_model() # Synchronous model loading
            
            embeddings_to_add = []
            metadata_to_add = []
            ids_to_add = []
            
            processing_logger.info(f"video_id: {video_id} - Processing {len(segments)} text segments for embedding...", extra=log_extra)
            for i, segment_data in enumerate(segments): # Renamed 'segment' to 'segment_data'
                text = segment_data.get("text", "").strip()
                if not text:
                    continue
                start_time = segment_data.get("start")
                end_time = segment_data.get("end")
                # model.encode is synchronous / CPU-bound
                embedding = model.encode(text, convert_to_tensor=False).tolist()
                segment_id_str = f"{video_id}_text_{i}" # Renamed 'segment_id'
                embeddings_to_add.append(embedding)
                metadata_to_add.append({
                    "video_uuid": video_id, "text_content": text, "start_time": start_time,
                    "end_time": end_time, "source_type": "transcript_segment"
                })
                ids_to_add.append(segment_id_str)

            if embeddings_to_add:
                text_embeddings_collection.add(embeddings=embeddings_to_add, metadatas=metadata_to_add, ids=ids_to_add)
                processing_logger.info(f"video_id: {video_id} - Added {len(embeddings_to_add)} text embeddings to ChromaDB.", extra=log_extra)
            
            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.TEXT_EMBEDDED, f"Added {len(embeddings_to_add)} text embeddings"
            )
            return True

        except Exception as e:
            error_msg = f"Text embedding error: {str(e)}"
            processing_logger.exception(f"video_id: {video_id} - {error_msg}", extra=log_extra)
            try:
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            except Exception as db_err:
                processing_logger.error(f"video_id: {video_id} - Additionally, failed to update DB status after text embedding error: {db_err}", extra=log_extra)
            return False

# --- Vision Embedding Generation ---
async def run_vision_embedding_pipeline(video_id: str, frames_directory_path: str, processing_output_base_path: str) -> bool:
    log_extra = {'video_id': video_id}
    processing_logger.info(f"video_id: {video_id} - Starting vision embedding pipeline for frames in: {frames_directory_path}", extra=log_extra)

    async with get_db_session() as session:
        try:
            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.VISION_EMBEDDING
            )

            if not vision_embeddings_collection:
                error_msg = "ChromaDB vision_embeddings_collection not available."
                processing_logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return False

            if not os.path.isdir(frames_directory_path):
                error_msg = f"Frames directory not found at {frames_directory_path}"
                processing_logger.error(f"video_id: {video_id} - {error_msg}", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = get_vision_embedding_model_and_processor() # Synchronous model loading
            
            frame_files = sorted(glob.glob(os.path.join(frames_directory_path, "frame_*.jpg")))
            if not frame_files:
                processing_logger.warning(f"video_id: {video_id} - No frame files found in {frames_directory_path}.", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.VISION_EMBEDDED, "No frames for vision embedding")
                return True # Technically successful, just no embeddings

            embeddings_to_add = []
            metadata_to_add = []
            ids_to_add = []

            processing_logger.info(f"video_id: {video_id} - Processing {len(frame_files)} frames for vision embedding...", extra=log_extra)
            for i, frame_path in enumerate(frame_files):
                try:
                    image = Image.open(frame_path)
                    # processor and model inference are synchronous / CPU-GPU bound
                    inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy().flatten().tolist()
                    
                    frame_basename = os.path.basename(frame_path)
                    try:
                        frame_number = int(frame_basename.split('_')[1].split('.')[0])
                        frame_timestamp_sec = float(frame_number - 1) # Simplistic, assumes 1 FPS and frame_0001 is t=0
                    except:
                        frame_timestamp_sec = float(i)
                    frame_id_str = f"{video_id}_vision_{i}" # Renamed 'frame_id'

                    embeddings_to_add.append(embedding)
                    metadata_to_add.append({
                        "video_uuid": video_id, "frame_filename": frame_basename,
                        "frame_timestamp_sec": frame_timestamp_sec, "source_type": "video_frame"
                    })
                    ids_to_add.append(frame_id_str)
                except Exception as frame_e:
                    processing_logger.error(f"video_id: {video_id} - Failed to process frame {frame_path}: {frame_e}", extra=log_extra)
                    continue

            if embeddings_to_add:
                vision_embeddings_collection.add(embeddings=embeddings_to_add, metadatas=metadata_to_add, ids=ids_to_add)
                processing_logger.info(f"video_id: {video_id} - Added {len(embeddings_to_add)} vision embeddings to ChromaDB.", extra=log_extra)
            elif len(frame_files) > 0:
                 processing_logger.warning(f"video_id: {video_id} - No vision embeddings were generated, though {len(frame_files)} frames were found. Check frame processing errors.", extra=log_extra)

            await database_service.update_video_status_and_error(
                session, video_id, VideoProcessingStatus.VISION_EMBEDDED, f"Added {len(embeddings_to_add)} vision embeddings"
            )
            return True

        except Exception as e:
            error_msg = f"Vision embedding error: {str(e)}"
            processing_logger.exception(f"video_id: {video_id} - {error_msg}", extra=log_extra)
            try:
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg)
            except Exception as db_err:
                processing_logger.error(f"video_id: {video_id} - Additionally, failed to update DB status after vision embedding error: {db_err}", extra=log_extra)
            return False

# --- if __name__ == "__main__": block for direct testing ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    module_logger.info("Running embedding_service.py directly for testing...")
    
    test_video_id = "direct_embedding_test_001"
    # Ensure current_dir and project_root_dir are defined if this block is executed.
    # Typically, os.getcwd() if running from 'backend' folder.
    current_dir_test = os.path.dirname(os.path.abspath(__file__))
    project_root_test = os.path.join(current_dir_test, "..")

    test_processing_base_dir = os.path.join(project_root_test, "direct_test_output", test_video_id, "embeddings_module")
    os.makedirs(test_processing_base_dir, exist_ok=True)
    
    # Mock database_service for direct testing
    class MockDBSession:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def commit(self): module_logger.info("Mock DB (embedding_test): Commit called")
        async def rollback(self): module_logger.info("Mock DB (embedding_test): Rollback called")
        async def close(self): module_logger.info("Mock DB (embedding_test): Session closed")

    original_db_service = database_service # Save original
    class MockDatabaseServiceModule:
        VideoProcessingStatus = original_db_service.VideoProcessingStatus
        async def get_db_session(self): return MockDBSession()
        async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
            module_logger.info(f"Mock DB (embedding_test): video_id={video_id}, status={status.value}, error='{error_msg}'")
        # Add other necessary mock methods if your functions call them (e.g., get_video_record_by_uuid)
    
    database_service = MockDatabaseServiceModule() # Monkey patch

    # --- Test Text Embeddings ---
    module_logger.info(f"--- TESTING TEXT EMBEDDINGS for video_id: {test_video_id} ---")
    test_transcript_dir = os.path.join(test_processing_base_dir, "transcripts")
    os.makedirs(test_transcript_dir, exist_ok=True)
    test_transcript_file = os.path.join(test_transcript_dir, "transcript_whisper.json")
    dummy_transcript = {
        "text": "Hello embedding world. This is a test segment.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": " Hello embedding world."},
            {"id": 1, "start": 2.0, "end": 4.5, "text": " This is a test segment."}
        ]
    }
    with open(test_transcript_file, "w") as f: json.dump(dummy_transcript, f)
    module_logger.info(f"Dummy transcript created at: {test_transcript_file}")
    
    if os.path.exists(test_transcript_file):
        text_success = asyncio.run(run_text_embedding_pipeline(test_video_id, test_transcript_file, test_processing_base_dir))
        module_logger.info(f"Text embedding pipeline success for video_id {test_video_id}: {text_success}")
        if text_success and text_embeddings_collection:
             results = text_embeddings_collection.get(ids=[f"{test_video_id}_text_0"], include=["metadatas", "documents"])
             module_logger.info(f"ChromaDB get result for text (video_id {test_video_id}): IDs - {results['ids']}")

    # --- Test Vision Embeddings ---
    module_logger.info(f"\n--- TESTING VISION EMBEDDINGS for video_id: {test_video_id} ---")
    test_frames_dir = os.path.join(test_processing_base_dir, "frames")
    os.makedirs(test_frames_dir, exist_ok=True)
    try:
        dummy_image = Image.new('RGB', (60, 30), color = 'red')
        dummy_image.save(os.path.join(test_frames_dir, "frame_0001.jpg"))
        dummy_image.save(os.path.join(test_frames_dir, "frame_0002.jpg"))
        module_logger.info(f"Created dummy frame files in {test_frames_dir}")
    except Exception as e_img:
        module_logger.error(f"Could not create dummy frame image: {e_img}. Vision embedding test might fail.")

    if os.path.isdir(test_frames_dir) and len(os.listdir(test_frames_dir)) > 0:
        vision_success = asyncio.run(run_vision_embedding_pipeline(test_video_id, test_frames_dir, test_processing_base_dir))
        module_logger.info(f"Vision embedding pipeline success for video_id {test_video_id}: {vision_success}")
        if vision_success and vision_embeddings_collection:
            results = vision_embeddings_collection.get(ids=[f"{test_video_id}_vision_0"], include=["metadatas"])
            module_logger.info(f"ChromaDB get result for vision (video_id {test_video_id}): IDs - {results['ids']}")
    
    database_service = original_db_service # Restore original service
    # Optional: Clean up test ChromaDB data / test files
    # import shutil
    # if os.path.exists(test_processing_base_dir):
    #     shutil.rmtree(test_processing_base_dir)
    # if text_embeddings_collection: text_embeddings_collection.delete(where={"video_uuid": test_video_id})
    # if vision_embeddings_collection: vision_embeddings_collection.delete(where={"video_uuid": test_video_id})
    module_logger.info("Direct test for embedding_service.py finished.")