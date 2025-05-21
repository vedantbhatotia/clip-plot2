
# backend/services/embedding_service.py
import os
import json
import logging
import time
import glob
from PIL import Image
import asyncio # For asyncio.run in __main__
from typing import Optional, List, Dict, Any, AsyncGenerator # Added AsyncGenerator
from contextlib import asynccontextmanager # For mocking get_db_session

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import chromadb
from dotenv import load_dotenv
load_dotenv()
# --- Import Database Service ---
from . import database_service # Assuming services is a package
from .database_service import VideoProcessingStatus, get_db_session

# --- Loggers ---
module_logger = logging.getLogger(f"app.{__name__}")
log_system_extra = {'video_id': 'SYSTEM'}
processing_logger = logging.getLogger(f"processing.{__name__}")

# --- Configuration ---
DEFAULT_TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.getcwd(), "data", "chroma_db_dev"))
TEXT_EMBEDDINGS_COLLECTION_NAME = "clippilot_text_embeddings"
VISION_EMBEDDINGS_COLLECTION_NAME = "clippilot_vision_embeddings"

# --- ChromaDB Client (Uses module_logger) ---
persistent_client = None
text_embeddings_collection = None
vision_embeddings_collection = None

try:
    if "data/chroma_db_dev" in CHROMA_DB_PATH: # Heuristic for default relative path
        data_dir = os.path.dirname(CHROMA_DB_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            module_logger.info(f"Created data directory for ChromaDB: {data_dir}")

    module_logger.info(f"Attempting to initialize ChromaDB PersistentClient at path: {CHROMA_DB_PATH}")
    persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    module_logger.info(f"ChromaDB PersistentClient initialized successfully for path: {CHROMA_DB_PATH}")

    module_logger.info(f"Getting/Creating text collection: {TEXT_EMBEDDINGS_COLLECTION_NAME} with 'cosine' metric.")
    text_embeddings_collection = persistent_client.get_or_create_collection(
        name=TEXT_EMBEDDINGS_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # <<<--- VERIFIED/ENSURED: Using Cosine Distance
    )
    module_logger.info(f"Text collection '{TEXT_EMBEDDINGS_COLLECTION_NAME}' ready. Metadata: {text_embeddings_collection.metadata}")

    module_logger.info(f"Getting/Creating vision collection: {VISION_EMBEDDINGS_COLLECTION_NAME} with 'cosine' metric.")
    vision_embeddings_collection = persistent_client.get_or_create_collection(
        name=VISION_EMBEDDINGS_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # <<<--- VERIFIED/ENSURED: Using Cosine Distance
    )
    module_logger.info(f"Vision collection '{VISION_EMBEDDINGS_COLLECTION_NAME}' ready. Metadata: {vision_embeddings_collection.metadata}")

except Exception as e:
    module_logger.exception(f"CRITICAL: Failed to initialize ChromaDB client or collections using path {CHROMA_DB_PATH}. Embedding service may not function.")


# --- Model Loading (Synchronous, as it's CPU/GPU bound and typically once) ---
_text_model_instance = None
def get_text_embedding_model(model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL) -> SentenceTransformer:
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
def get_vision_embedding_model_and_processor(model_name: str = DEFAULT_VISION_EMBEDDING_MODEL) -> tuple[CLIPModel, CLIPProcessor]:
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
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.TEXT_EMBEDDED, "No segments for text embedding")
                return True

            model = get_text_embedding_model()
            
            embeddings_to_add: List[List[float]] = []
            metadata_to_add: List[Dict[str, Any]] = []
            ids_to_add: List[str] = []
            documents_for_chroma: List[str] = []  # <<<--- ENSURED: List for documents

            processing_logger.info(f"video_id: {video_id} - Processing {len(segments)} text segments for embedding...", extra=log_extra)
            for i, segment_data in enumerate(segments):
                text = segment_data.get("text", "").strip()
                if not text:
                    continue
                start_time = segment_data.get("start")
                end_time = segment_data.get("end")
                embedding = model.encode(text, convert_to_tensor=False).tolist()
                segment_id_str = f"{video_id}_text_{i}"
                
                embeddings_to_add.append(embedding)
                metadata_item = {
                    "video_uuid": video_id, "text_content": text, "start_time": start_time,
                    "end_time": end_time, "source_type": "transcript_segment"
                }
                metadata_to_add.append(metadata_item)
                ids_to_add.append(segment_id_str)
                documents_for_chroma.append(text)  # <<<--- ENSURED: Populating documents

            if embeddings_to_add:
                if not (len(embeddings_to_add) == len(metadata_to_add) == len(ids_to_add) == len(documents_for_chroma)):
                    error_msg_mismatch = "Internal error: Mismatch in lengths of data lists for ChromaDB text embedding."
                    processing_logger.error(f"video_id: {video_id} - {error_msg_mismatch}", extra=log_extra)
                    await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_mismatch)
                    return False

                text_embeddings_collection.add(
                    embeddings=embeddings_to_add,
                    metadatas=metadata_to_add,
                    documents=documents_for_chroma,  # <<<--- ENSURED: Passing documents
                    ids=ids_to_add
                )
                processing_logger.info(f"video_id: {video_id} - Added {len(embeddings_to_add)} text embeddings (with documents) to ChromaDB.", extra=log_extra)
            
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
async def run_vision_embedding_pipeline(
    video_id: str,
    frames_directory_path: str,
    processing_output_base_path: str,
    effective_fps: float = 1.0 # <<<--- ENSURED: Parameter for accurate timestamps
) -> bool:
    log_extra = {'video_id': video_id}
    processing_logger.info(f"video_id: {video_id} - Starting vision embedding pipeline (Using effective_fps: {effective_fps}) for frames in: {frames_directory_path}", extra=log_extra)

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
            model, processor = get_vision_embedding_model_and_processor()
            
            frame_files = sorted(glob.glob(os.path.join(frames_directory_path, "frame_*.jpg")))
            if not frame_files:
                processing_logger.warning(f"video_id: {video_id} - No frame files found in {frames_directory_path}.", extra=log_extra)
                await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.VISION_EMBEDDED, "No frames for vision embedding")
                return True

            embeddings_to_add: List[List[float]] = []
            metadata_to_add: List[Dict[str, Any]] = []
            ids_to_add: List[str] = []

            processing_logger.info(f"video_id: {video_id} - Processing {len(frame_files)} frames for vision embedding...", extra=log_extra)
            for i, frame_path in enumerate(frame_files):
                try:
                    image = Image.open(frame_path)
                    inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy().flatten().tolist()
                    
                    frame_basename = os.path.basename(frame_path)
                    frame_number_parsed = None # For robust ID generation
                    try:
                        frame_number_parsed = int(frame_basename.split('_')[1].split('.')[0])
                        # Accurate timestamp calculation using effective_fps
                        # frame_number is 1-based (frame_0001.jpg), timestamp is 0-based
                        frame_timestamp_sec = (frame_number_parsed - 1) / effective_fps if effective_fps > 0 else float(i)
                    except (IndexError, ValueError):
                        processing_logger.warning(f"video_id: {video_id} - Could not parse frame number from '{frame_basename}'. Using index {i} for timestamp calc.", extra=log_extra)
                        frame_timestamp_sec = float(i) / effective_fps if effective_fps > 0 else float(i) # Fallback
                    
                    frame_id_str = f"{video_id}_vision_{frame_number_parsed if frame_number_parsed is not None else i}"

                    embeddings_to_add.append(embedding)
                    metadata_to_add.append({
                        "video_uuid": video_id, "frame_filename": frame_basename,
                        "frame_timestamp_sec": round(frame_timestamp_sec, 3),
                        "source_type": "video_frame"
                    })
                    ids_to_add.append(frame_id_str)
                except Exception as frame_e:
                    processing_logger.error(f"video_id: {video_id} - Failed to process frame {frame_path}: {frame_e}", extra=log_extra)
                    continue

            if embeddings_to_add:
                if not (len(embeddings_to_add) == len(metadata_to_add) == len(ids_to_add)):
                    error_msg_mismatch = "Internal error: Mismatch in lengths of data lists for ChromaDB vision embedding."
                    processing_logger.error(f"video_id: {video_id} - {error_msg_mismatch}", extra=log_extra)
                    await database_service.update_video_status_and_error(session, video_id, VideoProcessingStatus.PROCESSING_FAILED, error_msg_mismatch)
                    return False
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
        log_format = '%(asctime)s - %(levelname)s - %(name)s  - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    module_logger.info("Running embedding_service.py directly for testing...")
    
    test_video_id_main = "direct_embedding_main_test_002" # Changed ID for fresh test potentially
    current_dir_test = os.path.dirname(os.path.abspath(__file__))
    project_root_test = os.path.join(current_dir_test, "..")
    test_processing_base_dir_main = os.path.join(project_root_test, "direct_test_output", test_video_id_main, "embeddings_module")
    os.makedirs(test_processing_base_dir_main, exist_ok=True)
    
    # Mock database_service
    class MockDBSession:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
    original_db_service_ref = database_service
    class MockDatabaseServiceModule:
        VideoProcessingStatus = original_db_service_ref.VideoProcessingStatus
        @asynccontextmanager
        async def get_db_session(self) -> AsyncGenerator[MockDBSession, None]:
            yield MockDBSession()
        async def update_video_status_and_error(self, session, video_id, status, error_msg=None):
            module_logger.info(f"MOCK DB (embedding_test): video_id={video_id}, status={status.value}, error='{error_msg}'")
            return True
    database_service = MockDatabaseServiceModule()

    async def run_tests():
        module_logger.info(f"--- TESTING TEXT EMBEDDINGS for video_id: {test_video_id_main} ---")
        test_transcript_dir = os.path.join(test_processing_base_dir_main, "transcripts")
        os.makedirs(test_transcript_dir, exist_ok=True)
        test_transcript_file = os.path.join(test_transcript_dir, "transcript_whisper.json")
        dummy_transcript = {
            "text": "Another hello world. This is another test segment.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": " Another hello world."},
                {"id": 1, "start": 2.5, "end": 5.0, "text": " This is another test segment."}
            ]
        }
        with open(test_transcript_file, "w") as f: json.dump(dummy_transcript, f)
        
        text_success = await run_text_embedding_pipeline(test_video_id_main, test_transcript_file, test_processing_base_dir_main)
        module_logger.info(f"Text embedding pipeline success for video_id {test_video_id_main}: {text_success}")
        if text_success and text_embeddings_collection:
             results = text_embeddings_collection.get(ids=[f"{test_video_id_main}_text_0"], include=["metadatas", "documents"])
             module_logger.info(f"ChromaDB get result for text (video_id {test_video_id_main}): Docs - {results.get('documents', [])}")

        module_logger.info(f"\n--- TESTING VISION EMBEDDINGS for video_id: {test_video_id_main} ---")
        test_frames_dir = os.path.join(test_processing_base_dir_main, "frames")
        os.makedirs(test_frames_dir, exist_ok=True)
        try:
            Image.new('RGB', (60, 30), color = 'cyan').save(os.path.join(test_frames_dir, "frame_0001.jpg"))
            Image.new('RGB', (60, 30), color = 'magenta').save(os.path.join(test_frames_dir, "frame_0002.jpg"))
        except Exception as e_img: module_logger.error(f"Could not create dummy frame image: {e_img}")

        if os.path.isdir(test_frames_dir) and len(os.listdir(test_frames_dir)) > 0:
            vision_success = await run_vision_embedding_pipeline(test_video_id_main, test_frames_dir, test_processing_base_dir_main, effective_fps=2.0) # Test with 2 FPS
            module_logger.info(f"Vision embedding pipeline success for video_id {test_video_id_main}: {vision_success}")
            if vision_success and vision_embeddings_collection:
                results = vision_embeddings_collection.get(ids=[f"{test_video_id_main}_vision_0"], include=["metadatas"]) # Check first frame
                if results and results.get('metadatas') and results['metadatas']:
                     module_logger.info(f"ChromaDB get result for vision (video_id {test_video_id_main}): Meta - {results['metadatas'][0]}")

    asyncio.run(run_tests())
    database_service = original_db_service_ref # Restore original service
    module_logger.info("Direct test for embedding_service.py finished.")