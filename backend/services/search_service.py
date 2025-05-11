
import logging
from typing import List, Dict, Any, Optional

from .embedding_service import get_text_embedding_model, text_embeddings_collection 
from .database_service import get_db_session, get_video_record_by_uuid 

from .embedding_service import get_vision_embedding_model_and_processor, vision_embeddings_collection
import torch
logger = logging.getLogger(__name__) 

async def perform_semantic_text_search(
    query_text: str,
    video_uuid: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs a semantic text search in ChromaDB.
    """
    log_extra = {'video_id': video_uuid if video_uuid else "all_videos"}
    logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - Performing semantic search for query: '{query_text}', top_k: {top_k}", extra=log_extra)

    if not text_embeddings_collection:
        logger.error("ChromaDB text_embeddings_collection is not available.")

        return []

    try:
        text_model = get_text_embedding_model() # Load the SentenceTransformer model
        query_embedding = text_model.encode(query_text, convert_to_tensor=False).tolist()

        where_filter = None
        if video_uuid:
            where_filter = {"video_uuid": video_uuid}
            logger.info(f"Applying filter for video_uuid: {video_uuid}", extra=log_extra)

        results = text_embeddings_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"] 
        )
        logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - ChromaDB query returned {len(results.get('ids', [[]])[0]) if results else 0} potential results before processing.", extra=log_extra)


        # Process results
        
        processed_results = []
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]
            metadatas = results.get('metadatas', [[]])[0]
            documents = results.get('documents', [[]])[0] 
            distances = results.get('distances', [[]])[0]

            for i in range(len(ids)):
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                doc = documents[i] if documents and i < len(documents) else "" 
                dist = distances[i] if distances and i < len(distances) else None
                
                processed_results.append({
                    "id": ids[i], 
                    "video_uuid": meta.get("video_uuid"),
                    "segment_text": doc, 
                    "start_time": meta.get("start_time"),
                    "end_time": meta.get("end_time"),
                    "score_type": "distance", 
                    "score": dist,
                })
            logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - Processed {len(processed_results)} search results.", extra=log_extra)
        else:
            logger.info(f"video_id: {video_uuid if video_uuid else 'all'} - No results found in ChromaDB for the query.", extra=log_extra)
            
        return processed_results

    except Exception as e:
        logger.exception(f"video_id: {video_uuid if video_uuid else 'all'} - Error during semantic text search for query '{query_text}'", extra=log_extra)
        return []

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

if __name__ == "__main__":

    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    async def test_search():
        logger.info("Running search_service.py directly for testing...")
        try:
            get_text_embedding_model()
            if not text_embeddings_collection:
                logger.error("ChromaDB text collection not initialized in embedding_service. Test will fail.")
                return
        except Exception as model_err:
            logger.error(f"Failed to load embedding model for test: {model_err}")
            return

        test_video_uuid_for_search = "search-test-vid-001"
        if text_embeddings_collection.count() == 0 or text_embeddings_collection.get(where={"video_uuid": test_video_uuid_for_search})['ids'] == []:
            logger.info("ChromaDB is empty or no data for test UUID, adding mock data for search test...")
            mock_texts = [
                "The weather is sunny and bright today.",
                "I enjoy eating apples and bananas for breakfast.",
                "Machine learning is a fascinating field of study.",
                "A happy customer is the best business strategy of all.",
                "Thank you for your excellent customer support, it was very helpful."
            ]
            mock_embeddings = get_text_embedding_model().encode(mock_texts).tolist()
            mock_metadatas = [
                {"video_uuid": test_video_uuid_for_search, "text_content": t, "start_time": i*10, "end_time": i*10 + 5, "source_type": "transcript_segment"}
                for i, t in enumerate(mock_texts)
            ]
            mock_ids = [f"{test_video_uuid_for_search}_text_mock_{i}" for i in range(len(mock_texts))]
            try:
                text_embeddings_collection.add(
                    embeddings=mock_embeddings,
                    metadatas=mock_metadatas,
                    documents=mock_texts, 
                    ids=mock_ids
                )
                logger.info(f"Added {len(mock_ids)} mock entries to ChromaDB for {test_video_uuid_for_search}")
            except Exception as chroma_add_err:
                logger.error(f"Failed to add mock data to ChromaDB: {chroma_add_err}")
                return



        query1 = "positive customer feedback"
        logger.info(f"\n--- Test Case 1: Query: '{query1}' ---")
        results1 = await perform_semantic_text_search(query_text=query1, video_uuid=test_video_uuid_for_search, top_k=3)
        if results1:
            for res in results1:
                logger.info(f"  Result: Score={res['score']:.4f}, Start={res['start_time']}, Text='{res['segment_text'][:50]}...'")
        else:
            logger.info("  No results found.")

        query2 = "quantum physics lecture"
        logger.info(f"\n--- Test Case 2: Query: '{query2}' ---")
        results2 = await perform_semantic_text_search(query_text=query2, video_uuid=test_video_uuid_for_search, top_k=3)
        if results2:
            for res in results2:
                logger.info(f"  Result: Score={res['score']:.4f}, Start={res['start_time']}, Text='{res['segment_text'][:50]}...'")
        else:
            logger.info("  No results found.")
        
        query3 = "sunny weather"
        logger.info(f"\n--- Test Case 3: Query: '{query3}' (all videos) ---")
        results3 = await perform_semantic_text_search(query_text=query3, top_k=2) 
        if results3:
            for res in results3:
                 logger.info(f"  Result: Video={res['video_uuid']}, Score={res['score']:.4f}, Start={res['start_time']}, Text='{res['segment_text'][:50]}...'")
        else:
            logger.info("  No results found.")


    if __name__ == "__main__":
        import asyncio
        asyncio.run(test_search())