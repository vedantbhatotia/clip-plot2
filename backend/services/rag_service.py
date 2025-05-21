# backend/services/rag_service.py
import os
import logging
import asyncio # For running blocking LangChain calls in threads
from typing import List, Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # For more complex chains

# Loggers
module_logger = logging.getLogger(f"app.{__name__}")
processing_logger = logging.getLogger(f"processing.{__name__}")

# --- RAG Configuration ---
# These can be moved to .env or a config file
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_RAG_MODEL", "llama2:7b") # Or "gemma:7b", etc.
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

RAG_PROMPT_TEMPLATE_STR = """
You are an AI assistant for the ClipPilot system. Your task is to analyze video content segments
retrieved based on a user's query and provide a concise, helpful answer or summary.
Only use the information present in the "Context from video segments" provided below.
Do not make up information. If the context is insufficient, say so.

User Query:
"{query_str}"

Context from video segments:
{context_str}

Based *only* on the provided context and user query, generate a helpful response:
"""

_llm_instance = None
_rag_chain_instance = None

def get_llm_and_rag_chain():
    global _llm_instance, _rag_chain_instance
    if _rag_chain_instance is None:
        try:
            module_logger.info(f"Initializing Ollama LLM with model: {DEFAULT_OLLAMA_MODEL} at URL: {DEFAULT_OLLAMA_BASE_URL}")
            _llm_instance = Ollama(
                base_url=DEFAULT_OLLAMA_BASE_URL,
                model=DEFAULT_OLLAMA_MODEL,
                # You can add other parameters like temperature, top_k, top_p if needed
                # temperature=0.7 
            )
            # Test the LLM connection (optional but good for startup)
            # _llm_instance.invoke("Hi") # This is a blocking call, careful at module level if not handled
            
            rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)
            output_parser = StrOutputParser()
            
            _rag_chain_instance = (
                {"context_str": RunnablePassthrough(), "query_str": RunnablePassthrough()} 
                | rag_prompt 
                | _llm_instance 
                | output_parser
            )
            module_logger.info("RAG chain initialized successfully.")
        except Exception as e:
            module_logger.exception(f"Failed to initialize Ollama LLM or RAG chain. RAG features will be unavailable. Error: {e}")
            _llm_instance = None
            _rag_chain_instance = None
    return _llm_instance, _rag_chain_instance

async def refine_search_results_with_rag(
    query_str: str,
    search_results: List[Dict[str, Any]], # Expects list of dicts like your TextSearchResultItem
    max_context_segments: int = 3, # How many top search results to use as context
    video_id_for_log: Optional[str] = None # For logging
) -> Optional[str]:
    log_extra = {'video_id': video_id_for_log if video_id_for_log else "rag_refinement"}
    processing_logger.info(f"video_id: {video_id_for_log or 'N/A'} - Attempting RAG refinement for query: '{query_str}'", extra=log_extra)

    llm, rag_chain = get_llm_and_rag_chain() # Ensure chain is initialized

    if not llm or not rag_chain:
        processing_logger.warning(f"video_id: {video_id_for_log or 'N/A'} - RAG chain/LLM not available. Cannot refine search.", extra=log_extra)
        return "RAG processing is currently unavailable." # Or return None

    if not search_results:
        processing_logger.info(f"video_id: {video_id_for_log or 'N/A'} - No initial search results provided to RAG.", extra=log_extra)
        return "No initial search results were found to refine." # Or None

    # --- Prepare context string from top search results ---
    # We prefer text segments for RAG context.
    context_segments_text = []
    for i, res in enumerate(search_results[:max_context_segments]):
        # Prioritize 'segment_text', then 'text_content' from metadata
        text = res.get("segment_text") or res.get("text_content")
        if text:
            # Including timestamps can be helpful for context
            start_time = res.get("start_time")
            end_time = res.get("end_time")
            time_info = ""
            if start_time is not None and end_time is not None:
                time_info = f"(from {start_time:.2f}s to {end_time:.2f}s)"
            elif start_time is not None: # For visual hit with only start time
                time_info = f"(around {start_time:.2f}s)"
            
            context_segments_text.append(f"Segment {i+1} {time_info}: \"{text.strip()}\"")
    
    if not context_segments_text:
        processing_logger.info(f"video_id: {video_id_for_log or 'N/A'} - No usable text content found in search results for RAG context.", extra=log_extra)
        return "Relevant context could not be extracted from search results for RAG." # Or None

    context_str = "\n\n".join(context_segments_text)
    
    processing_logger.info(f"video_id: {video_id_for_log or 'N/A'} - Invoking RAG chain with {len(context_segments_text)} context segments. Query: '{query_str}'", extra=log_extra)
    processing_logger.debug(f"video_id: {video_id_for_log or 'N/A'} - RAG Context:\n{context_str}", extra=log_extra)
    
    try:
        # LangChain's Ollama invoke is blocking. Run it in a thread pool for async FastAPI.
        # If your LangChain Ollama wrapper supports .ainvoke(), prefer that.
        # As of recent LangChain versions, some LLM wrappers have .ainvoke()
        if hasattr(rag_chain, 'ainvoke'):
             refined_answer = await rag_chain.ainvoke({"context_str": context_str, "query_str": query_str})
        else: # Fallback to running blocking invoke in a thread
            loop = asyncio.get_event_loop()
            refined_answer = await loop.run_in_executor(
                None,  # Uses default thread pool executor
                rag_chain.invoke,
                {"context_str": context_str, "query_str": query_str}
            )

        processing_logger.info(f"video_id: {video_id_for_log or 'N/A'} - RAG chain invocation successful. Response length: {len(refined_answer)}", extra=log_extra)
        processing_logger.debug(f"video_id: {video_id_for_log or 'N/A'} - RAG Response:\n{refined_answer}", extra=log_extra)
        return refined_answer
    except Exception as e:
        processing_logger.exception(f"video_id: {video_id_for_log or 'N/A'} - Error during RAG chain invocation for query: '{query_str}'", extra=log_extra)
        return f"Error while refining search results with RAG: {str(e)}" # Or None

# --- if __name__ == "__main__": for direct testing ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=log_format) # DEBUG for more verbose output

    async def test_rag_service():
        module_logger.info("Running RAG service direct test...")
        
        # Try to initialize LLM and chain
        llm, chain = get_llm_and_rag_chain()
        if not llm or not chain:
            module_logger.error("LLM or RAG chain failed to initialize. Aborting test.")
            return

        sample_query = "What are the key financial highlights?"
        sample_search_results = [
            {"segment_text": "Our revenue grew by 20% this quarter, reaching $5 million.", "start_time": 10.5, "end_time": 15.2},
            {"segment_text": "Net profit increased to $1.2 million, a 15% rise.", "start_time": 16.0, "end_time": 20.0},
            {"segment_text": "However, operating expenses also went up by 10%.", "start_time": 22.1, "end_time": 25.5},
            {"segment_text": "We launched three new products in the last period.", "start_time": 30.0, "end_time": 33.0}
        ]

        module_logger.info(f"\n--- Test Case 1: RAG with financial data ---")
        rag_response = await refine_search_results_with_rag(sample_query, sample_search_results, video_id_for_log="rag_test_001")
        module_logger.info(f"Query: {sample_query}")
        module_logger.info(f"RAG Response:\n{rag_response}")

        sample_query_2 = "What is this video about?"
        sample_search_results_2 = [
            {"segment_text": "This clip shows a tutorial on how to bake a chocolate cake.", "start_time": 5.0, "end_time": 10.0},
            {"segment_text": "We first mix the dry ingredients, then add the wet ones.", "start_time": 12.0, "end_time": 18.0}
        ]
        module_logger.info(f"\n--- Test Case 2: RAG with cake tutorial ---")
        rag_response_2 = await refine_search_results_with_rag(sample_query_2, sample_search_results_2, video_id_for_log="rag_test_002")
        module_logger.info(f"Query: {sample_query_2}")
        module_logger.info(f"RAG Response:\n{rag_response_2}")

        sample_query_3 = "Any mention of cats?"
        sample_search_results_3 = [] # No search results
        module_logger.info(f"\n--- Test Case 3: RAG with no search results ---")
        rag_response_3 = await refine_search_results_with_rag(sample_query_3, sample_search_results_3, video_id_for_log="rag_test_003")
        module_logger.info(f"Query: {sample_query_3}")
        module_logger.info(f"RAG Response:\n{rag_response_3}")


    asyncio.run(test_rag_service())