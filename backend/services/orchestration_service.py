# backend/services/orchestration_service.py
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel, Field as PydanticField

# Corrected LangChain import for Ollama
from langchain_ollama import OllamaLLM # Use this instead of langchain_community.llms.Ollama
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool # For creating custom tools
from langchain import hub # For pre-built ReAct prompts

# Import your actual service functions
from .search_service import perform_semantic_text_search, perform_semantic_visual_search
from .segment_processor_service import refine_segments_for_clip
from .database_service import get_video_record_by_uuid, get_db_session
from services import database_service
from moviepy.editor import VideoFileClip # For GetVideoDurationTool

logger = logging.getLogger(__name__) # This will be services.orchestration_service
log_system_extra = {'video_id': 'AGENT_ORCHESTRATION'} # For logs originating from this module's setup

LLM_MODEL_NAME = os.getenv("OLLAMA_AGENT_MODEL", "llama2:7b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# TEMP_VIDEO_DIR is not directly used here, but tools might need paths relative to it via DB

_agent_llm_instance: Optional[OllamaLLM] = None
_agent_executor_instance: Optional[AgentExecutor] = None


# --- Tool Input Schemas (using Pydantic) ---
class TextSearchToolInput(BaseModel):
    video_uuid: str = PydanticField(description="The UUID of the video to search within.")
    search_query: str = PydanticField(description="The text query for searching video transcripts.")
    top_k: int = PydanticField(5, description="Number of results to return.")

class VisualSearchToolInput(BaseModel):
    video_uuid: str = PydanticField(description="The UUID of the video to search within.")
    visual_description_query: str = PydanticField(description="A text description of the visual content to search for.")
    top_k: int = PydanticField(5, description="Number of results to return.")

class GetVideoDurationToolInput(BaseModel):
    video_uuid: str = PydanticField(description="The UUID of the video to get duration for.")

class RefineSegmentsToolInput(BaseModel):
    segments: List[Dict[str, Any]] = PydanticField(description="List of raw segments (dicts with 'start_time', 'end_time', 'text_content').")
    video_duration: Optional[float] = PydanticField(None, description="Total duration of the video in seconds, for capping.")
    video_id: str = PydanticField(description="The UUID of the video these segments belong to (for logging).")
    padding_start_sec: float = PydanticField(0.5, description="Padding to add to start of segments.")
    padding_end_sec: float = PydanticField(0.5, description="Padding to add to end of segments.")
    max_gap_to_merge_sec: float = PydanticField(0.2, description="Max gap between segments to merge.")


# --- Tool Implementations with _run and _arun ---
class VideoTextSearchTool(BaseTool):
    name: str = "VideoTextSearchTool"
    description: str = (
        "Use this tool to search for text segments within a specific video based on a query. "
        "Input must be a JSON string that conforms to TextSearchToolInput schema, e.g., "
        "{\"video_uuid\": \"<uuid>\", \"search_query\": \"<query>\", \"top_k\": 5}"
    )
    args_schema: Type[BaseModel] = TextSearchToolInput

    def _run(self, video_uuid: str, search_query: str, top_k: int = 5) -> str:
        logger.info(f"Agent Tool SYNC: VideoTextSearch. UUID: {video_uuid}, Query: '{search_query}'", extra=log_system_extra)
        # For synchronous _run, if called from a context that might not have an event loop,
        # or if the agent executor specifically needs a sync path.
        # Running an async function from a sync function in an already async app can be tricky.
        # This is a simplified approach; proper async-to-sync bridging might be needed
        # if LangChain's AgentExecutor calls _run from an async context without handling it.
        # However, the primary way an async agent should use this is via _arun.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If called from an async context, this is problematic.
                # Langchain should ideally use the coroutine.
                # This is a fallback / indicates a potential misconfiguration in how agent uses tools.
                logger.warning("VideoTextSearchTool._run called from a running event loop. This is unexpected for an async agent.")
                # This might block or error. Let's attempt to run it anyway for now.
                return asyncio.run(self._arun(video_uuid=video_uuid, search_query=search_query, top_k=top_k))

            else: # No loop running, so we can create one.
                return asyncio.run(self._arun(video_uuid=video_uuid, search_query=search_query, top_k=top_k))
        except RuntimeError as e: # Catches "cannot be called from a running event loop"
             logger.error(f"RuntimeError in VideoTextSearchTool._run (likely loop issue): {e}. Returning error string.")
             return f"Error: Could not execute text search due to async loop issue: {e}"


    async def _arun(self, video_uuid: str, search_query: str, top_k: int = 5) -> str:
        logger.info(f"Agent Tool ASYNC: VideoTextSearch. UUID: {video_uuid}, Query: '{search_query}', TopK: {top_k}", extra=log_system_extra)
        try:
            results = await perform_semantic_text_search(query_text=search_query, video_uuid=video_uuid, top_k=top_k)
            return json.dumps(results) if results else "No relevant text segments found."
        except Exception as e:
            logger.exception("Error in VideoTextSearchTool _arun", extra=log_system_extra)
            return f"Error during text search: {str(e)}"

class VideoVisualSearchTool(BaseTool):
    name: str = "VideoVisualSearchTool"
    description: str = (
        "Use this tool to search for visual scenes/frames within a specific video based on a textual description. "
        "Input must be a JSON string conforming to VisualSearchToolInput schema, e.g., "
        "{\"video_uuid\": \"<uuid>\", \"visual_description_query\": \"<query>\", \"top_k\": 5}"
    )
    args_schema: Type[BaseModel] = VisualSearchToolInput

    def _run(self, video_uuid: str, visual_description_query: str, top_k: int = 5) -> str:
        logger.info(f"Agent Tool SYNC: VideoVisualSearch. UUID: {video_uuid}, Query: '{visual_description_query}'", extra=log_system_extra)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("VideoVisualSearchTool._run called from a running event loop.")
                return asyncio.run(self._arun(video_uuid=video_uuid, visual_description_query=visual_description_query, top_k=top_k))
            else:
                return asyncio.run(self._arun(video_uuid=video_uuid, visual_description_query=visual_description_query, top_k=top_k))
        except RuntimeError as e:
            logger.error(f"RuntimeError in VideoVisualSearchTool._run: {e}. Returning error string.")
            return f"Error: Could not execute visual search due to async loop issue: {e}"


    async def _arun(self, video_uuid: str, visual_description_query: str, top_k: int = 5) -> str:
        logger.info(f"Agent Tool ASYNC: VideoVisualSearch. UUID: {video_uuid}, Query: '{visual_description_query}', TopK: {top_k}", extra=log_system_extra)
        try:
            results = await perform_semantic_visual_search(query_text_for_visual=visual_description_query, video_uuid=video_uuid, top_k=top_k)
            return json.dumps(results) if results else "No relevant visual segments found."
        except Exception as e:
            logger.exception("Error in VideoVisualSearchTool _arun", extra=log_system_extra)
            return f"Error during visual search: {str(e)}"

class GetVideoDurationTool(BaseTool):
    name: str = "GetVideoDurationTool"
    description: str = "Use this to find out the total duration in seconds of a specific video. Input must be a JSON string conforming to GetVideoDurationToolInput schema, e.g., {\"video_uuid\": \"<uuid>\"}"
    args_schema: Type[BaseModel] = GetVideoDurationToolInput
    
    def _run(self, video_uuid: str) -> str:
        logger.info(f"Agent Tool SYNC: GetVideoDuration for UUID: {video_uuid}", extra=log_system_extra)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("GetVideoDurationTool._run called from a running event loop.")
                return asyncio.run(self._arun(video_uuid=video_uuid))
            else:
                return asyncio.run(self._arun(video_uuid=video_uuid))
        except RuntimeError as e:
            logger.error(f"RuntimeError in GetVideoDurationTool._run: {e}. Returning error string.")
            return f"Error: Could not get video duration due to async loop issue: {e}"

    async def _arun(self, video_uuid: str) -> str:
        logger.info(f"Agent Tool ASYNC: GetVideoDuration called for: {video_uuid}", extra=log_system_extra)
        path = None
        try:
            async with get_db_session() as session:
                record = await get_video_record_by_uuid(session, video_uuid)
                if record and record.original_video_file_path:
                    path = record.original_video_file_path
            
            if path and os.path.exists(path):
                def get_dur_sync(p):
                    with VideoFileClip(p) as clip: return clip.duration
                duration = await asyncio.to_thread(get_dur_sync, path)
                return f"Video duration is {duration:.2f} seconds."
            else:
                logger.warning(f"Could not find video file or its path to get duration for {video_uuid}", extra=log_system_extra)
                return "Could not find video or its path to get duration."
        except Exception as e:
            logger.exception(f"Error getting duration for {video_uuid}", extra=log_system_extra)
            return f"Error getting video duration: {str(e)}"


class RefineVideoSegmentsTool(BaseTool):
    name: str = "RefineVideoSegmentsTool"
    description: str = (
        "Use this tool to process a list of raw video segments (with 'start_time', 'end_time', 'text_content') "
        "by adding padding and merging overlaps. Input must be a JSON string conforming to RefineSegmentsToolInput schema. "
        "Example: {\"segments\": [{\"start_time\": 10.0, \"end_time\": 12.0, \"text_content\": \"...\"}], "
        "\"video_duration\": 120.0, \"video_id\": \"<uuid>\", \"padding_start_sec\": 0.5, ...}"
    )
    args_schema: Type[BaseModel] = RefineSegmentsToolInput

    def _run(self, segments: List[Dict[str, Any]], video_id: str, video_duration: Optional[float] = None,
             padding_start_sec: float = 0.5, padding_end_sec: float = 0.5, max_gap_to_merge_sec: float = 0.2) -> str:
        logger.info(f"Agent Tool SYNC: RefineVideoSegments for video_id: {video_id}", extra=log_system_extra)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("RefineVideoSegmentsTool._run called from a running event loop.")
                return asyncio.run(self._arun(segments=segments, video_id=video_id, video_duration=video_duration,
                                          padding_start_sec=padding_start_sec, padding_end_sec=padding_end_sec,
                                          max_gap_to_merge_sec=max_gap_to_merge_sec))
            else:
                 return asyncio.run(self._arun(segments=segments, video_id=video_id, video_duration=video_duration,
                                          padding_start_sec=padding_start_sec, padding_end_sec=padding_end_sec,
                                          max_gap_to_merge_sec=max_gap_to_merge_sec))
        except RuntimeError as e:
            logger.error(f"RuntimeError in RefineVideoSegmentsTool._run: {e}. Returning error string.")
            return f"Error: Could not refine segments due to async loop issue: {e}"


    async def _arun(self, segments: List[Dict[str, Any]], video_id: str, video_duration: Optional[float] = None,
                    padding_start_sec: float = 0.5, padding_end_sec: float = 0.5, max_gap_to_merge_sec: float = 0.2) -> str:
        logger.info(f"Agent Tool ASYNC: RefineVideoSegments for video_id: {video_id} with {len(segments)} segments.", extra=log_system_extra)
        try:
            refined = refine_segments_for_clip(
                segments=segments, video_duration=video_duration, video_id=video_id,
                padding_start_sec=padding_start_sec, padding_end_sec=padding_end_sec,
                max_gap_to_merge_sec=max_gap_to_merge_sec
            )
            return json.dumps(refined) if refined else "No segments remained after refinement."
        except Exception as e:
            logger.exception(f"Error in RefineVideoSegmentsTool _arun for video_id: {video_id}", extra=log_system_extra)
            return f"Error during segment refinement: {str(e)}"


def initialize_agent_executor():
    global _agent_llm_instance, _agent_executor_instance
    if _agent_executor_instance is not None:
        logger.debug("Agent Executor already initialized.", extra=log_system_extra)
        return _agent_executor_instance

    try:
        logger.info(f"Initializing AGENT LLM with model: {LLM_MODEL_NAME} at {DEFAULT_OLLAMA_BASE_URL}", extra=log_system_extra)
        _agent_llm_instance = OllamaLLM(base_url=DEFAULT_OLLAMA_BASE_URL, model=LLM_MODEL_NAME)

        tools = [
            VideoTextSearchTool(),
            VideoVisualSearchTool(),
            GetVideoDurationTool(),
            RefineVideoSegmentsTool(),
        ]
        
        # Using a standard ReAct prompt from LangChain Hub. This prompt might need to be
        # customized to better instruct the LLM to format its final answer as a JSON list.
        prompt = hub.pull("hwchase17/react")

        agent = create_react_agent(_agent_llm_instance, tools, prompt)
        _agent_executor_instance = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors="Check your output and make sure it conforms to the Action/Action Input format. If you think you are finished, ensure your Final Answer is ONLY the JSON list of segments or the string 'NO_SEGMENTS_FOUND'.", # More instructive error handling
            max_iterations=10 
        )
        logger.info("Agent Executor initialized successfully with ReAct agent.", extra=log_system_extra)
    except Exception as e:
        logger.exception("CRITICAL: Failed to initialize Agent LLM or Executor. Orchestration will not work.", extra=log_system_extra)
        _agent_llm_instance = None
        _agent_executor_instance = None
    return _agent_executor_instance


async def run_clip_orchestration_agent(
    user_query: str,
    video_uuid: str
) -> Optional[List[Dict[str, Any]]]:
    log_extra = {'video_id': video_uuid}
    logger.info(f"Starting AGENT-BASED clip orchestration. Query: '{user_query}'", extra=log_extra)
    
    agent_executor = initialize_agent_executor()
    if not agent_executor:
        logger.error("Agent Executor not available. Cannot orchestrate clip.", extra=log_extra)
        return None

    agent_input_str = (
        f"User Query: '{user_query}'. "
        f"The video_uuid you must use for all relevant tools is: '{video_uuid}'. "
        "Your goal is to identify and refine a list of video segments (each a dictionary with 'start_time', 'end_time', 'text_content') "
        "that best satisfy the user's query for this video, suitable for creating a highlight clip. "
        "Follow these steps: "
        "1. Get the video duration using GetVideoDurationTool. "
        "2. Use VideoTextSearchTool and/or VideoVisualSearchTool (if the query implies visual content) to find relevant raw segments. "
        "3. If you found segments, use the RefineVideoSegmentsTool with the video duration and the search results to get refined segments. "
        "Your final answer MUST be ONLY the JSON string representation of the list of these refined segments, "
        "for example: '[{\"start_time\": 10.0, \"end_time\": 15.0, \"text_content\": \"...\"}, ...]', "
        "or the exact string 'NO_SEGMENTS_FOUND' if no suitable segments can be identified after trying the tools."
    )
    
    try:
        logger.info(f"Invoking Agent Executor with input string (length: {len(agent_input_str)})", extra=log_extra)
        logger.debug(f"Full agent input string: {agent_input_str}", extra=log_extra)
        
        if hasattr(agent_executor, 'ainvoke'):
            response_dict = await agent_executor.ainvoke({"input": agent_input_str})
        else:
            loop = asyncio.get_event_loop()
            response_dict = await loop.run_in_executor(None, agent_executor.invoke, {"input": agent_input_str})
        
        logger.info(f"Agent Executor raw response dictionary: {response_dict}", extra=log_extra)
        agent_final_output_str = response_dict.get("output")

        if not agent_final_output_str:
            logger.error("Agent did not produce an 'output' string in its response.", extra=log_extra)
            return None

        if agent_final_output_str.strip().upper() == "NO_SEGMENTS_FOUND":
            logger.info("Agent determined no suitable segments found.", extra=log_extra)
            return []
        
        try:
            final_segments = json.loads(agent_final_output_str)
            if isinstance(final_segments, list):
                # Basic validation of segment structure
                valid_segments = []
                for seg in final_segments:
                    if isinstance(seg, dict) and "start_time" in seg and "end_time" in seg:
                        valid_segments.append(seg)
                    else:
                        logger.warning(f"Agent output contained an invalid segment structure: {seg}", extra=log_extra)
                
                if valid_segments:
                    logger.info(f"Agent successfully returned and parsed {len(valid_segments)} segments for clip building.", extra=log_extra)
                    return valid_segments
                else:
                    logger.error(f"Agent output was a list, but contained no valid segments: '{agent_final_output_str}'", extra=log_extra)
                    return None
            else:
                logger.error(f"Agent output was parsed as JSON but is not a list: '{agent_final_output_str}'", extra=log_extra)
                return None
        except json.JSONDecodeError:
            logger.error(f"Agent final output was NOT valid JSON: '{agent_final_output_str}'. This was its thought process: {response_dict.get('intermediate_steps', 'N/A')}", extra=log_extra)
            return None

    except Exception as e:
        logger.exception("Error during agent execution for clip orchestration.", extra=log_extra)
        return None

if __name__ == "__main__":
    # Ensure main_logger exists for the filter if it's used globally or adapt
    if not hasattr(logging.getLogger(__name__), 'filters') or not any(isinstance(f, logging.Filter) for f in logging.getLogger(__name__).filters): # Simple check
        class VideoIDLogFilter(logging.Filter): # Define it for standalone test
            def filter(self, record):
                if not hasattr(record, 'video_id'): record.video_id = 'N/A_standalone'
                return True
        logging.getLogger().addFilter(VideoIDLogFilter())

    if not logging.getLogger().handlers: # Check if basicConfig was already called (e.g. by importing main)
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - [%(video_id)s] - %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    async def test_orchestration_service():
        logger.info("Running orchestration_service.py directly for testing...", extra={'video_id': 'ORCH_TEST_MAIN'})
        
        test_video_uuid = "agent_direct_test_vid_002"
        
        # --- Mock database_service for tools ---
        class MockVideoRecord:
            def __init__(self, uuid, path, duration=70.0): # Longer duration for testing
                self.video_uuid = uuid; self.original_video_file_path = path; self.duration = duration
        
        async def mock_get_video_record_by_uuid(session, video_uuid_arg): # Renamed arg
            mock_path = f"/tmp/fake_video_{video_uuid_arg}.mp4"
            if not os.path.exists(os.path.dirname(mock_path)): os.makedirs(os.path.dirname(mock_path), exist_ok=True)
            if not os.path.exists(mock_path):
                with open(mock_path, 'w') as f: f.write(f"fake video data for {video_uuid_arg}")
            return MockVideoRecord(video_uuid_arg, mock_path)

        original_db_get_record = database_service.get_video_record_by_uuid
        database_service.get_video_record_by_uuid = mock_get_video_record_by_uuid
        # --- End Mocking DB ---

        # --- Mock Search Service for tools ---
        original_text_search = perform_semantic_text_search
        original_visual_search = perform_semantic_visual_search

        async def mock_perform_text_search(query_text: str, video_uuid: Optional[str] = None, top_k: int = 5):
            logger.info(f"MOCK Text Search: query='{query_text}', uuid='{video_uuid}', top_k={top_k}", extra={'video_id': video_uuid})
            if "financial" in query_text:
                return [{"id": "text_s1", "video_uuid": video_uuid, "segment_text": "The company's revenue grew significantly.", "start_time": 10.0, "end_time": 15.0, "score": 0.9}]
            return []
        
        perform_semantic_text_search = mock_perform_text_search # Monkey patch
        # --- End Mocking Search ---

        test_query = "Show me the financial highlights."
        logger.info(f"\n--- Test Case: Agent Clip Orchestration for video {test_video_uuid} ---", extra={'video_id': test_video_uuid})
        logger.info(f"Query: {test_query}", extra={'video_id': test_video_uuid})
        
        executor = initialize_agent_executor() # This will initialize Ollama
        if not executor:
            logger.error("Agent executor failed to initialize in direct test. Aborting.", extra={'video_id': test_video_uuid})
            database_service.get_video_record_by_uuid = original_db_get_record # Restore
            return

        final_segments = await run_clip_orchestration_agent(test_query, test_video_uuid)

        if final_segments is not None:
            logger.info(f"Agent orchestration successful for test. Segments ({len(final_segments)}):", extra={'video_id': test_video_uuid})
            for seg_idx, seg_val in enumerate(final_segments): logger.info(f"  - Segment {seg_idx}: {seg_val}", extra={'video_id': test_video_uuid})
        else:
            logger.error("Agent orchestration failed or returned no segments in test.", extra={'video_id': test_video_uuid})

        # Restore original functions
        database_service.get_video_record_by_uuid = original_db_get_record
        perform_semantic_text_search = original_text_search
        perform_semantic_visual_search = original_visual_search # If you had mocked it

    if os.getenv("OLLAMA_BASE_URL") and os.getenv("OLLAMA_AGENT_MODEL"):
        asyncio.run(test_orchestration_service())
    else:
        logger.warning("OLLAMA_BASE_URL or OLLAMA_AGENT_MODEL not set. Skipping direct agent orchestration test.", extra=log_system_extra)