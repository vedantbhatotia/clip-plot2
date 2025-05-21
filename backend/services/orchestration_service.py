# backend/services/orchestration_service.py
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate # Or ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent # Or other agent types
from langchain_core.tools import Tool

# Import your actual service functions that will be wrapped by tools
from .search_service import perform_semantic_text_search # Needs to be callable by agent
from .segment_processor_service import refine_segments_for_clip # Needs to be callable by agent
# from .clip_builder_service import generate_highlight_clip # The agent will output segments FOR this
from .database_service import get_video_record_by_uuid, get_db_session
from moviepy.editor import VideoFileClip # Only for GetVideoDurationTool's implementation

logger = logging.getLogger(__name__)
LLM_MODEL_NAME = os.getenv("OLLAMA_AGENT_MODEL", "llama3:8b") # Could be same or different model
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMP_VIDEO_DIR = os.getenv("TEMP_VIDEO_DIR", "/tmp/clippilot_uploads_dev") # Needed for clip builder call

_agent_llm_instance = None
_agent_executor_instance = None

# --- Tool Implementations (Wrappers around your services) ---

async def _text_search_tool_wrapper(query_and_uuid_str: str) -> str:
    """
    Input should be a string like 'video_uuid: <uuid>, query: <search_text>'.
    Searches video transcripts for relevant text segments.
    """
    logger.info(f"Agent Tool: TextSearch called with: {query_and_uuid_str}")
    try:
        # Basic parsing, make this more robust
        parts = query_and_uuid_str.split(", query: ")
        uuid_part = parts[0].replace("video_uuid: ", "").strip()
        query_part = parts[1].strip() if len(parts) > 1 else ""
        if not query_part: return "Error: Query text is missing."

        results = await perform_semantic_text_search(query_text=query_part, video_uuid=uuid_part, top_k=5)
        return json.dumps(results) if results else "No text segments found."
    except Exception as e:
        return f"Error in TextSearchTool: {e}"

async def _get_video_duration_tool_wrapper(video_uuid: str) -> str:
    logger.info(f"Agent Tool: GetVideoDuration called for: {video_uuid}")
    path = None
    async with get_db_session() as session:
        record = await get_video_record_by_uuid(session, video_uuid)
        if record and record.original_video_file_path:
            path = record.original_video_file_path
    if path and os.path.exists(path):
        try:
            def get_dur_sync(p):
                with VideoFileClip(p) as clip: return clip.duration
            duration = await asyncio.to_thread(get_dur_sync, path)
            return f"Video duration is {duration:.2f} seconds."
        except Exception as e:
            return f"Error getting duration: {e}"
    return "Could not find video or its path to get duration."

async def _refine_segments_tool_wrapper(input_json_str: str) -> str:
    """
    Input is a JSON string: 
    '{"segments": [{"start_time": s, "end_time": e, "text_content": t}, ...], "video_duration": dur, "video_id": id}'
    """
    logger.info(f"Agent Tool: RefineSegments called with input length: {len(input_json_str)}")
    try:
        data = json.loads(input_json_str)
        refined = refine_segments_for_clip(
            segments=data.get("segments", []),
            video_duration=data.get("video_duration"),
            video_id=data.get("video_id"),
            # Agent could also specify these if needed:
            # padding_start_sec=data.get("padding_start_sec", 0.5), 
            # padding_end_sec=data.get("padding_end_sec", 0.5),
            # max_gap_to_merge_sec=data.get("max_gap_to_merge_sec", 0.2)
        )
        return json.dumps(refined) if refined else "No segments remained after refinement."
    except Exception as e:
        return f"Error in RefineSegmentsTool: {e}"


def get_agent_executor():
    global _agent_llm_instance, _agent_executor_instance
    if _agent_executor_instance is None:
        try:
            logger.info(f"Initializing AGENT LLM with model: {LLM_MODEL_NAME}")
            _agent_llm_instance = Ollama(base_url=DEFAULT_OLLAMA_BASE_URL, model=LLM_MODEL_NAME)

            tools = [
                Tool(
                    name="VideoTextSearch",
                    coroutine=_text_search_tool_wrapper, # Use the async wrapper
                    description="Use this tool to search for text segments within a specific video. Input should be a string like 'video_uuid: <the_video_uuid>, query: <your_search_query_for_text_content>'."
                ),
                Tool(
                    name="GetVideoDuration",
                    coroutine=_get_video_duration_tool_wrapper,
                    description="Use this to find out the total duration in seconds of a specific video. Input is the video_uuid string."
                ),
                Tool(
                    name="RefineVideoSegments",
                    coroutine=_refine_segments_tool_wrapper,
                    description="Use this tool to process a list of raw video segments (with start_time, end_time, text_content) by adding padding and merging overlaps. Input must be a JSON string: '{\"segments\": [...], \"video_duration\": 60.0, \"video_id\": \"uuid\"}'. Returns a JSON string of refined segments."
                ),
                # The "ClipBuilderTool" would be the final action, but the agent's goal is to *select segments for it*.
                # The agent's final output could be the refined list of segments.
            ]
            
            # Example ReAct prompt structure (you'll need to customize heavily)
            # This prompt needs to guide the LLM to output a list of segments for the clip builder
            # OR to call a final tool that triggers the clip builder.
            agent_prompt_template = """
            You are ClipPilot, an expert video highlight generation assistant.
            Your goal: Given a user's request and a video_uuid, identify the best segments to create a highlight clip.
            Output the final list of selected and refined segments as a JSON list like:
            [{"start_time": s1, "end_time": e1, "text_content": "text1"}, {"start_time": s2, "end_time": e2, "text_content": "text2"}]
            or state "NO_SEGMENTS_FOUND" if no suitable segments can be identified.

            TOOLS:
            ------
            You have access to the following tools:
            {tools}

            To use a tool, use the following format:
            ```
            Thought: Do I need to use a tool? Yes
            Action: The action to take, should be one of [{tool_names}]
            Action Input: The input to the action
            Observation: The result of the action
            ```

            When you have a final list of segments to recommend for the clip, or if you decide no segments are suitable,
            provide it in the specified JSON format as your Final Answer, or "NO_SEGMENTS_FOUND".
            Do not make up segments; they must come from search results and be refined.
            Consider video duration when refining segments.

            Begin!

            User Request: {input}
            Video UUID: {video_uuid}
            Thought:
            {agent_scratchpad}
            """
            # This is a simplified prompt. Real agent prompts are more complex.
            # You might need a specific prompt for LangGraph nodes if you go that route.
            
            # For ReAct, you'd typically use a pre-built prompt or a more structured one from langchain.prompts
            # from langchain import hub
            # prompt = hub.pull("hwchase17/react") # This is a generic ReAct prompt
            # For this example, we'll just show a conceptual template.
            # A proper agent setup with create_react_agent would require the prompt to fit its expected format.

            # This is a placeholder for where you'd create the actual agent and executor
            # For now, let's assume the goal is to get the executor working, even if the prompt is simple.
            # agent = create_react_agent(_agent_llm_instance, tools, prompt) # This needs a compatible prompt
            # _agent_executor_instance = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            
            logger.warning("Agent Executor is NOT fully implemented/configured in this example. get_agent_executor() will return None.")
            _agent_executor_instance = None # Placeholder until full agent setup

        except Exception as e:
            logger.exception(f"Failed to initialize Agent LLM or Executor. Error: {e}")
            _agent_llm_instance = None
            _agent_executor_instance = None
    return _agent_executor_instance # Return the executor


async def run_clip_orchestration_agent(
    user_query: str,
    video_uuid: str
) -> Optional[List[Dict[str, Any]]]: # Should return the list of segments for the builder
    log_extra = {'video_id': video_uuid}
    logger.info(f"Starting AGENT-BASED clip orchestration for query: '{user_query}'", extra=log_extra)
    
    agent_executor = get_agent_executor()
    if not agent_executor:
        logger.error("Agent Executor not available. Cannot orchestrate clip.", extra=log_extra)
        return None # Or raise an exception

    try:
        # The agent's input needs to be structured according to its main prompt
        agent_input_str = f"User query: '{user_query}'. Video UUID is '{video_uuid}'."
        
        logger.info(f"Invoking Agent Executor with input: {agent_input_str}", extra=log_extra)
        # Agent execution is blocking, run in thread
        # The actual output parsing depends heavily on how your agent is prompted to give its final answer.
        # For this conceptual example, let's assume it directly outputs the JSON string of segments.
        agent_response_str = await asyncio.to_thread(agent_executor.invoke, {"input": agent_input_str})
        
        logger.info(f"Agent Executor raw response: {agent_response_str}", extra=log_extra)
        output_value = agent_response_str.get("output", "")

        if output_value == "NO_SEGMENTS_FOUND":
            logger.info("Agent determined no suitable segments found.", extra=log_extra)
            return []
        try:
            final_segments = json.loads(output_value)
            if isinstance(final_segments, list):
                logger.info(f"Agent successfully returned {len(final_segments)} segments for clip building.", extra=log_extra)
                return final_segments
            else:
                logger.error(f"Agent output was not a list after JSON parsing: {output_value}", extra=log_extra)
                return None
        except json.JSONDecodeError:
            logger.error(f"Agent output was not valid JSON: {output_value}", extra=log_extra)
            # Fallback or error handling: Maybe just pass the raw string if it's descriptive.
            # Or attempt to extract segments from it if it's a natural language list.
            return None # Or handle error by trying to parse the text

    except Exception as e:
        logger.exception("Error during agent execution for clip orchestration.", extra=log_extra)
        return None

if __name__ == "__main__":
    # ... (Your __main__ for rag_service.py testing RAG summary can remain)
    # ... (Or you can add tests for the conceptual agent parts if you build them out)
    pass