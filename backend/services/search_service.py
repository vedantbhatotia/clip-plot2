
# backend/services/search_service.py
import logging
from typing import List, Dict, Any, Optional
import os # For __main__ block
import json # For __main__ block
import asyncio # For __main__ block
from dotenv import load_dotenv
load_dotenv()

import torch # For perform_semantic_visual_search

# Import models and ChromaDB collections from embedding_service
# Assuming embedding_service.py is in the same directory or a discoverable path
from .embedding_service import (
    get_text_embedding_model,
    get_vision_embedding_model_and_processor,
    text_embeddings_collection,
    vision_embeddings_collection
)

# Use consistent logger naming
logger = logging.getLogger(__name__)
module_logger = logging.getLogger(f"app.{__name__}") # For module-level info if any
processing_logger = logging.getLogger(f"processing.{__name__}") # For per-search operation logs


async def perform_semantic_text_search(
    query_text: str,
    video_uuid: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs a semantic text search in ChromaDB.
    Returns results with a 'similarity_score' (1 - cosine_distance).
    """
    # Corrected log_extra initialization
    log_extra = {'video_id': video_uuid if video_uuid else "all_videos"}
    processing_logger.info(
        f"video_id: {video_uuid or 'all'} - Performing TEXT search for query: '{query_text}', top_k: {top_k}",
        extra=log_extra
    )

    if not text_embeddings_collection:
        processing_logger.error(f"video_id: {video_uuid or 'all'} - Text embeddings collection is not available.", extra=log_extra)
        return []
    try:
        text_model = get_text_embedding_model()
    except Exception as e:
        # Using processing_logger for consistency in request-path errors,
        # module_logger might be for setup/initialization issues not tied to a specific request.
        processing_logger.error(f"video_id: {video_uuid or 'all'} - Failed to get text embedding model for search. Error: {e}", extra=log_extra)
        return []

    try:
        query_embedding = text_model.encode(query_text, convert_to_tensor=False).tolist()

        where_filter = {}
        if video_uuid:
            where_filter["video_uuid"] = video_uuid
        
        processing_logger.info(f"video_id: {video_uuid or 'all'} - Querying ChromaDB (text) with filter: {where_filter or 'None'}", extra=log_extra)
        
        results_dict = text_embeddings_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None, # Pass None if filter is empty
            include=['metadatas', 'documents', 'distances']
        )
        
        # Robust check for results
        if not results_dict or not results_dict.get('ids') or not results_dict['ids'][0]:
            processing_logger.info(f"video_id: {video_uuid or 'all'} - No results from ChromaDB text query.", extra=log_extra)
            return []

        num_results_from_db = len(results_dict['ids'][0])
        processing_logger.info(f"video_id: {video_uuid or 'all'} - ChromaDB text query returned {num_results_from_db} items.", extra=log_extra)

        formatted_results = []
        # Safely access lists from results_dict, providing empty lists as defaults
        ids_list = results_dict.get('ids', [[]])[0]
        metadatas_list = results_dict.get('metadatas', [[]])[0] or []
        documents_list = results_dict.get('documents', [[]])[0] or []
        distances_list = results_dict.get('distances', [[]])[0] or []

        for i in range(len(ids_list)):
            doc_id = ids_list[i]
            # Ensure index is within bounds for potentially shorter lists
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            doc_text_from_chroma = documents_list[i] if i < len(documents_list) else ""
            dist = distances_list[i] if i < len(distances_list) else None
            
            segment_text_to_use = doc_text_from_chroma if doc_text_from_chroma else meta.get("text_content", "")
            
            similarity_score = (1 - dist) if dist is not None else None
            # Assuming collection uses cosine distance, where Chroma distance = 1 - cosine_similarity
            
            formatted_results.append({
                "id": doc_id,
                "video_uuid": meta.get("video_uuid"),
                "segment_text": segment_text_to_use,
                "start_time": meta.get("start_time"),
                "end_time": meta.get("end_time"),
                "score": similarity_score,
                "score_type": "cosine_similarity" if dist is not None else "unknown_distance",
                "_raw_distance": dist
            })
        
        processing_logger.info(f"video_id: {video_uuid or 'all'} - Processed {len(formatted_results)} text search results.", extra=log_extra)
        return formatted_results

    except Exception as e:
        processing_logger.exception(f"video_id: {video_uuid or 'all'} - Error during semantic text search for query '{query_text}'", extra=log_extra)
        return []


# async def perform_semantic_visual_search(
#     query_text_for_visual: str,
#     video_uuid: Optional[str] = None,
#     top_k: int = 5
# ) -> List[Dict[str, Any]]:
#     log_extra = {'video_id': video_uuid if video_uuid else "all_videos"}
#     processing_logger.info(
#         f"video_id: {video_uuid or 'all'} - Performing VISUAL search for text query: '{query_text_for_visual}', top_k: {top_k}",
#         extra=log_extra
#     )

#     if not vision_embeddings_collection:
#         processing_logger.error(f"video_id: {video_uuid or 'all'} - Vision embeddings collection is not available.", extra=log_extra)
#         return []
#     try:
#         vision_model, vision_processor = get_vision_embedding_model_and_processor()
#         clip_device = "cuda" if torch.cuda.is_available() else "cpu"
#         if not torch.cuda.is_available() and torch.backends.mps.is_available():
#             clip_device = "mps"
#     except Exception as e:
#         processing_logger.error(f"video_id: {video_uuid or 'all'} - Failed to get vision embedding model/processor for search. Error: {e}", extra=log_extra)
#         return []

#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu" # Added MPS check
#         if not torch.cuda.is_available() and torch.backends.mps.is_available():
#             device = "mps"
        
#         # text_inputs = vision_processor(text=[query_text_for_visual],
#         #                                return_tensors="pt", padding=True, truncation=True).to(device)
#         # with torch.no_grad():
#         #     text_features = vision_model.get_text_features(**text_inputs)
#         # query_embedding = text_features.cpu().numpy().flatten().tolist()

#         text_inputs_on_cpu = vision_processor(
#         text=[query_text_for_visual], 
#         return_tensors="pt", 
#         padding=True, 
#         truncation=True
#         ) # Keep inputs on CPU initially

#         with torch.no_grad():
            
#             original_model_device = next(vision_model.parameters()).device
#             if original_model_device.type == 'mps':
#                 processing_logger.info(f"Temporarily moving CLIP model to CPU for text feature extraction to avoid MPS issue.")
#                 vision_model.to('cpu')
#                 text_inputs_on_cpu = text_inputs_on_cpu.to('cpu') # Ensure inputs are also on CPU
#                 text_features = vision_model.get_text_features(**text_inputs_on_cpu)
#                 vision_model.to(original_model_device) # Move model back to original device (MPS)
#                 processing_logger.info(f"Moved CLIP model back to {original_model_device}.")
#             else: # If on CUDA or CPU already, proceed as normal
#                 text_inputs = text_inputs_on_cpu.to(clip_device)
#                 text_features = vision_model.get_text_features(**text_inputs)

#         query_embedding = text_features.cpu().numpy().flatten().tolist()

#         where_filter = {}
#         if video_uuid:
#             where_filter["video_uuid"] = video_uuid
        
#         processing_logger.info(f"video_id: {video_uuid or 'all'} - Querying ChromaDB (vision) with filter: {where_filter or 'None'}", extra=log_extra)
#         results_dict = vision_embeddings_collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#             where=where_filter if where_filter else None,
#             include=['metadatas', 'distances'] # No 'documents' for vision usually
#         )

#         if not results_dict or not results_dict.get('ids') or not results_dict['ids'][0]:
#             processing_logger.info(f"video_id: {video_uuid or 'all'} - No results from ChromaDB vision query.", extra=log_extra)
#             return []
        
#         num_results_from_db = len(results_dict['ids'][0])
#         processing_logger.info(f"video_id: {video_uuid or 'all'} - ChromaDB vision query returned {num_results_from_db} items.", extra=log_extra)

#         formatted_results = []
#         ids_list = results_dict.get('ids', [[]])[0]
#         metadatas_list = results_dict.get('metadatas', [[]])[0] or []
#         distances_list = results_dict.get('distances', [[]])[0] or []

#         for i in range(len(ids_list)):
#             doc_id = ids_list[i]
#             meta = metadatas_list[i] if i < len(metadatas_list) else {}
#             dist = distances_list[i] if i < len(distances_list) else None
            
#             similarity_score = (1 - dist) if dist is not None else None # Assuming cosine distance
#             score_type = "cosine_similarity" if dist is not None else "unknown_distance"
            
#             formatted_results.append({
#                 "id": doc_id,
#                 "video_uuid": meta.get("video_uuid"),
#                 "frame_filename": meta.get("frame_filename"), # Common metadata for vision
#                 "frame_timestamp_sec": meta.get("frame_timestamp_sec"), # Common metadata
#                 "score": similarity_score,
#                 "score_type": score_type,
#                 "_raw_distance": dist
#             })
        
#         processing_logger.info(f"video_id: {video_uuid or 'all'} - Processed {len(formatted_results)} visual search results.", extra=log_extra)
#         return formatted_results

#     except Exception as e:
#         processing_logger.exception(f"video_id: {video_uuid or 'all'} - Error during semantic visual search for query '{query_text_for_visual}'", extra=log_extra)
#         return []

async def perform_semantic_visual_search(
    query_text_for_visual: str, # Text describing the visual content
    video_uuid: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    log_extra = {'video_id': video_uuid if video_uuid else "all_videos"}
    logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - Performing semantic VISUAL search for query: '{query_text_for_visual}', top_k: {top_k}", extra=log_extra)

    if not vision_embeddings_collection:
        logger.error("ChromaDB vision_embeddings_collection is not available.")
        return []
    
    try:
        # Get CLIP model and processor
        # Note: Model loading itself is synchronous and cached in embedding_service
        vision_model, vision_processor = get_vision_embedding_model_and_processor()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Encode the text query using CLIP's text encoder
        # This creates an embedding in the same space as your image embeddings
        text_inputs = vision_processor(text=[query_text_for_visual], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embedding = vision_model.get_text_features(**text_inputs)
        query_embedding = text_embedding.cpu().numpy().flatten().tolist()


        where_filter = None
        if video_uuid:
            where_filter = {"video_uuid": video_uuid}
            logger.info(f"Applying filter for video_uuid: {video_uuid} to visual search", extra=log_extra)

        # 2. Query the vision_embeddings_collection
        results = vision_embeddings_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"] # You stored frame_filename, frame_timestamp_sec
        )
        logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - ChromaDB visual query returned {len(results.get('ids', [[]])[0]) if results else 0} potential results.", extra=log_extra)

        # 3. Process results
        processed_results = []
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for i in range(len(ids)):
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                dist = distances[i] if distances and i < len(distances) else None
                
                processed_results.append({
                    "id": ids[i], # Unique ID of the vision embedding
                    "video_uuid": meta.get("video_uuid"),
                    "frame_filename": meta.get("frame_filename"),
                    "frame_timestamp_sec": meta.get("frame_timestamp_sec"),
                    "score_type": "distance", # CLIP often uses cosine similarity; distance might be 1-similarity or L2
                    "score": dist,
                    # "metadata_debug": meta 
                })
            logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - Processed {len(processed_results)} visual search results.", extra=log_extra)
        else:
            logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - No visual results found in ChromaDB for the query.", extra=log_extra)

        return processed_results
    except Exception as e:
        logger.exception(f"video_id: {video_uuid if video_uuid else 'all'} - Error during semantic visual search for query '{query_text_for_visual}'", extra=log_extra)
        return []
    

# async def _add_mock_data_for_search_test(video_id_for_mock: str):
#     """Helper to add mock data to ChromaDB if it's empty for the test UUID."""
#     if not text_embeddings_collection or not vision_embeddings_collection:
#         module_logger.error("Cannot add mock data: ChromaDB collections not available from embedding_service.")
#         return False
    
#     mock_data_added_this_run = False

#     # --- Mock Text Data ---
#     try:
#         existing_text_check = text_embeddings_collection.get(where={"video_uuid": video_id_for_mock}, limit=1)
#         if existing_text_check and existing_text_check.get('ids') and existing_text_check['ids'][0]:
#             module_logger.info(f"Mock text data for video_uuid '{video_id_for_mock}' seems to already exist. Skipping text addition.")
#         else:
#             module_logger.info(f"Adding MOCK TEXT data to ChromaDB for video_uuid: {video_id_for_mock} for search testing...")
#             mock_texts = [
#                 "The old lighthouse stood tall against the stormy sky.",
#                 "Delicious pasta with rich tomato sauce and fresh basil.",
#                 "A detailed discussion about quarterly financial reports and future projections.",
#                 "The cat lazily stretched in a warm sunbeam by the window.",
#                 "Thank you for your insightful presentation on market trends."
#             ]
#             text_model_for_mock = get_text_embedding_model()
#             mock_text_embeddings = text_model_for_mock.encode(mock_texts).tolist()
#             mock_text_metadatas = [
#                 {"video_uuid": video_id_for_mock, "text_content": t, "start_time": round(i*10.0, 2), "end_time": round(i*10.0 + 5.23, 2), "source_type": "transcript_segment"}
#                 for i, t in enumerate(mock_texts)
#             ]
#             mock_text_ids = [f"{video_id_for_mock}_text_mock_{i}" for i in range(len(mock_texts))]
#             text_embeddings_collection.add(
#                 embeddings=mock_text_embeddings,
#                 metadatas=mock_text_metadatas,
#                 documents=mock_texts, # Store original text in documents field
#                 ids=mock_text_ids
#             )
#             module_logger.info(f"Added {len(mock_text_ids)} mock TEXT entries for {video_id_for_mock}")
#             mock_data_added_this_run = True
#     except Exception as e_text_add:
#         module_logger.exception(f"Error adding/checking mock text data for {video_id_for_mock}: {e_text_add}")
#         # Decide if this is critical enough to stop further mock data addition
#         # return False 

#     # --- Mock Vision Data ---
#     try:
#         existing_vision_check = vision_embeddings_collection.get(where={"video_uuid": video_id_for_mock}, limit=1)
#         if existing_vision_check and existing_vision_check.get('ids') and existing_vision_check['ids'][0]:
#             module_logger.info(f"Mock vision data for video_uuid '{video_id_for_mock}' seems to already exist. Skipping vision addition.")
#         else:
#             module_logger.info(f"Adding MOCK VISION data to ChromaDB for video_uuid: {video_id_for_mock} for search testing...")
#             mock_vision_descs_for_embedding = [ # These are text descriptions to generate CLIP embeddings from
#                 "a bright red apple on a wooden table",
#                 "a computer screen showing lines of code",
#                 "two people shaking hands in an office setting"
#             ]
#             vision_model_for_mock, vision_processor_for_mock = get_vision_embedding_model_and_processor()
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             if not torch.cuda.is_available() and torch.backends.mps.is_available():
#                 device = "mps"

#             mock_vision_embeddings = []
#             for desc in mock_vision_descs_for_embedding:
#                 inputs = vision_processor_for_mock(text=[desc], return_tensors="pt", padding=True, truncation=True).to(device)
#                 with torch.no_grad():
#                     features = vision_model_for_mock.get_text_features(**inputs) # Use text features for mocking image embeddings from text
#                 mock_vision_embeddings.append(features.cpu().numpy().flatten().tolist())

#             mock_vision_metadatas = [
#                 {"video_uuid": video_id_for_mock, "frame_filename": f"mock_frame_00{i+1}.jpg", 
#                  "frame_timestamp_sec": round(i*15.33, 2), "source_type": "video_frame", 
#                  "mock_description_for_embedding": desc} # Store the text used for embedding
#                 for i, desc in enumerate(mock_vision_descs_for_embedding)
#             ]
#             mock_vision_ids = [f"{video_id_for_mock}_vision_mock_{i}" for i in range(len(mock_vision_descs_for_embedding))]
#             # For vision embeddings, 'documents' are not typically stored in ChromaDB unless it's some textual representation.
#             # If you do store something, ensure it's consistent. Here, we omit 'documents'.
#             vision_embeddings_collection.add(
#                 embeddings=mock_vision_embeddings, metadatas=mock_vision_metadatas, ids=mock_vision_ids
#             )
#             module_logger.info(f"Added {len(mock_vision_ids)} mock VISION entries for {video_id_for_mock}")
#             mock_data_added_this_run = True
#     except Exception as e_vision_add:
#         module_logger.exception(f"Error adding/checking mock vision data for {video_id_for_mock}: {e_vision_add}")
    
#     return mock_data_added_this_run


if __name__ == "__main__":
    # Basic logging setup for direct script execution
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    # This setup needs to happen within an async function or after the event loop is running
    # if embedding_service.py initializes ChromaDB client at module level.
    # For simplicity, assume embedding_service.py handles its own ChromaDB client setup correctly.

    # async def main_test_search_service():
    #     module_logger.info("Running search_service.py directly for comprehensive testing...")
        
    #     # Attempt to pre-load models and check collections
    #     try:
    #         module_logger.info("Pre-loading embedding models for test...")
    #         get_text_embedding_model() # This will initialize the model if not already
    #         get_vision_embedding_model_and_processor() # Initializes if not already
    #         module_logger.info("Embedding models pre-loading attempt complete.")
    #         if not text_embeddings_collection or not vision_embeddings_collection:
    #             module_logger.error("ChromaDB collections (text or vision) not initialized from embedding_service.py. Cannot proceed with tests.")
    #             return
    #     except Exception as model_load_err:
    #         module_logger.exception(f"Failed to load embedding models for test: {model_load_err}")
    #         return

    #     MOCK_VIDEO_UUID_FOR_TEST = "search_test_mock_vid_002" # Changed to avoid conflicts if run multiple times
    #     mock_data_added = await _add_mock_data_for_search_test(MOCK_VIDEO_UUID_FOR_TEST)
    #     if not mock_data_added and not (text_embeddings_collection.get(where={"video_uuid": MOCK_VIDEO_UUID_FOR_TEST}, limit=1)['ids']):
    #         module_logger.warning("Mock data was not added and did not previously exist. Search tests might yield no results.")


    #     # --- Test Text Search ---
    #     query_text_financial = "financial reports"
    #     module_logger.info(f"\n--- Testing Text Search (video: {MOCK_VIDEO_UUID_FOR_TEST}) for query: '{query_text_financial}' ---")
    #     text_results1 = await perform_semantic_text_search(query_text_financial, video_uuid=MOCK_VIDEO_UUID_FOR_TEST, top_k=3)
    #     if text_results1:
    #         for res in text_results1:
    #             score = res.get('score')
    #             score_str = f"{score:.4f}" if score is not None else "N/A"
    #             module_logger.info(f"  TEXT RES: Score={score_str}, Type={res.get('score_type')}, Start={res.get('start_time')}, Text='{res.get('segment_text', '')[:60]}...'")
    #     else:
    #         module_logger.info("  No text results found for this specific video query.")

    #     query_text_sky = "stormy sky"
    #     module_logger.info(f"\n--- Testing Text Search (ALL videos, targeting '{MOCK_VIDEO_UUID_FOR_TEST}') for query: '{query_text_sky}' ---")
    #     text_results2 = await perform_semantic_text_search(query_text_sky, top_k=3) # No video_uuid filter
    #     if text_results2:
    #         found_specific_mock = False
    #         for res in text_results2:
    #             score = res.get('score')
    #             score_str = f"{score:.4f}" if score is not None else "N/A"
    #             module_logger.info(f"  TEXT RES (all): Video={res.get('video_uuid')}, Score={score_str}, Type={res.get('score_type')}, Text='{res.get('segment_text', '')[:60]}...'")
    #             if res.get('video_uuid') == MOCK_VIDEO_UUID_FOR_TEST:
    #                 found_specific_mock = True
    #         if not found_specific_mock:
    #              module_logger.info(f"  Mock data for video '{MOCK_VIDEO_UUID_FOR_TEST}' was not found in 'all videos' search for '{query_text_sky}'.")
    #     else:
    #         module_logger.info("  No text results found for 'all videos' query.")

    #     # --- Test Visual Search ---
    #     query_visual_code = "computer code lines"
    #     module_logger.info(f"\n--- Testing Visual Search (video: {MOCK_VIDEO_UUID_FOR_TEST}) for query: '{query_visual_code}' ---")
    #     visual_results1 = await perform_semantic_visual_search(query_visual_code, video_uuid=MOCK_VIDEO_UUID_FOR_TEST, top_k=3)
    #     if visual_results1:
    #         for res in visual_results1:
    #             score = res.get('score')
    #             score_str = f"{score:.4f}" if score is not None else "N/A"
    #             module_logger.info(f"  VISUAL RES: Score={score_str}, Type={res.get('score_type')}, Frame='{res.get('frame_filename')}', Time={res.get('frame_timestamp_sec')}")
    #     else:
    #         module_logger.info("  No visual results found for this specific video query.")
            
    #     module_logger.info("\n--- Search service direct test finished. ---")
        
    #     # Optional: Cleanup mock data after tests
    #     # try:
    #     #     if text_embeddings_collection:
    #     #         text_embeddings_collection.delete(where={"video_uuid": MOCK_VIDEO_UUID_FOR_TEST})
    #     #     if vision_embeddings_collection:
    #     #         vision_embeddings_collection.delete(where={"video_uuid": MOCK_VIDEO_UUID_FOR_TEST})
    #     #     module_logger.info(f"Attempted cleanup of mock data for video_uuid: {MOCK_VIDEO_UUID_FOR_TEST}")
    #     # except Exception as cleanup_err:
    #     #     module_logger.error(f"Error during mock data cleanup: {cleanup_err}")


    # asyncio.run(main_test_search_service())