
import os
import enum
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional # For type hinting

from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SAEnum, UniqueConstraint
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.future import select

from dotenv import load_dotenv
load_dotenv()
# --- Loggers ---
module_logger = logging.getLogger(f"app.{__name__}")
db_processing_logger = logging.getLogger(f"db.processing.{__name__}")

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")

async_engine = None
AsyncSessionLocal: Optional[sessionmaker] = None # Type hint for clarity

if not DATABASE_URL:
    module_logger.critical("DATABASE_URL environment variable not set. Database service CANNOT function.")
else:
    try:
        # Ensure the URL is using the asyncpg driver
        if DATABASE_URL.startswith("postgresql://"):
            actual_db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif DATABASE_URL.startswith("postgresql+asyncpg://"):
            actual_db_url = DATABASE_URL
        else:
            module_logger.error(f"Unsupported DATABASE_URL format: {DATABASE_URL}. Must start with postgresql:// or postgresql+asyncpg://")
            raise ValueError("Invalid DATABASE_URL format")

        async_engine = create_async_engine(actual_db_url, echo=False) # Set echo=True for debugging SQL
        AsyncSessionLocal = sessionmaker(
            bind=async_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
        # Log only host to avoid exposing password if DATABASE_URL contains it
        db_host_port = actual_db_url.split('@')[-1].split('/')[0] if '@' in actual_db_url else 'Unknown host'
        module_logger.info(f"Async database engine configured for: {db_host_port}")
    except Exception as e:
        module_logger.exception("Failed to create async database engine or session maker.")
        async_engine = None
        AsyncSessionLocal = None


Base = declarative_base()

# --- Enum for Processing Status ---
class VideoProcessingStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    MEDIA_EXTRACTING = "media_extracting"
    MEDIA_EXTRACTED = "media_extracted"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    TEXT_EMBEDDING = "text_embedding"
    TEXT_EMBEDDED = "text_embedded"
    VISION_EMBEDDING = "vision_embedding"
    VISION_EMBEDDED = "vision_embedded"
    READY_FOR_SEARCH = "ready_for_search"
    HIGHLIGHT_GENERATING = "highlight_generating"
    HIGHLIGHT_GENERATED = "highlight_generated"
    PROCESSING_FAILED = "processing_failed"
    PARTIAL_FAILURE = "partial_failure"

# --- SQLAlchemy Model ---
class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    video_uuid = Column(String, unique=True, index=True, nullable=False)
    # UniqueConstraint("video_uuid", name="uq_video_uuid") # unique=True on Column implies this

    original_filename_server = Column(String, nullable=True)
    original_video_file_path = Column(Text, nullable=True)
    audio_file_path = Column(Text, nullable=True)
    frames_directory_path = Column(Text, nullable=True)
    transcript_file_path = Column(Text, nullable=True)
    generated_highlight_path = Column(Text, nullable=True)

    processing_status = Column(SAEnum(VideoProcessingStatus, name="video_processing_status_enum", create_constraint=True),
                               nullable=False, default=VideoProcessingStatus.UPLOADED)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Video(video_uuid='{self.video_uuid}', status='{self.processing_status.value}')>"

# --- Database Initialization Function ---
async def init_db_tables():
    if not async_engine:
        module_logger.error("Database engine not initialized. Skipping table creation.")
        return
    try:
        async with async_engine.begin() as conn:
            module_logger.info("Creating database tables if they don't exist (using Base.metadata.create_all)...")
            # await conn.run_sync(Base.metadata.drop_all) # CAUTION: Deletes all data! For dev only.
            await conn.run_sync(Base.metadata.create_all)
            module_logger.info("Database tables checked/created successfully.")
    except Exception as e:
        module_logger.exception("Failed to create/check database tables.")


# --- Async Session Context Manager ---
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not AsyncSessionLocal:
        db_processing_logger.critical("AsyncSessionLocal not initialized. Cannot provide DB session.")
        raise RuntimeError("Database session factory not available.")
    
    session: AsyncSession = AsyncSessionLocal()
    try:
        yield session
        await session.commit() # Commit if no exceptions during 'yield'
    except Exception as e:
        db_processing_logger.exception(f"Exception in DB session for (likely) video_uuid involved in operation, rolling back.")
        await session.rollback() # Rollback on exception
        raise # Re-raise the exception after rollback
    finally:
        await session.close()
        # db_processing_logger.debug("DB session closed.")

# --- CRUD Operations ---
async def add_new_video_record(
    session: AsyncSession,
    video_uuid: str,
    original_filename_server: str,
    original_video_file_path: str
) -> Optional[Video]:
    db_processing_logger.info(f"video_id: {video_uuid} - Adding new video record to DB.")
    try:
        # Check if video_uuid already exists
        existing_video_stmt = select(Video).where(Video.video_uuid == video_uuid)
        result = await session.execute(existing_video_stmt)
        existing_video = result.scalar_one_or_none()
        if existing_video is not None:
            db_processing_logger.warning(f"video_id: {video_uuid} - Video record already exists with DB ID {existing_video.id}. Returning existing.")
            return existing_video

        new_video = Video(
            video_uuid=video_uuid,
            original_filename_server=original_filename_server,
            original_video_file_path=original_video_file_path,
            processing_status=VideoProcessingStatus.UPLOADED
        )
        session.add(new_video)
        await session.flush() # Assigns ID if autoincrement, makes it available before commit
        await session.refresh(new_video) # Ensure all attributes are loaded from DB after flush
        db_processing_logger.info(f"video_id: {video_uuid} - New video record added with DB ID: {new_video.id}.")
        return new_video
    except Exception as e:
        db_processing_logger.exception(f"video_id: {video_uuid} - Error adding new video record.")
        # Rollback will be handled by the get_db_session context manager if an exception is raised from here
        return None


async def update_video_status_and_error(
    session: AsyncSession,
    video_uuid: str,
    new_status: VideoProcessingStatus,
    error_msg: Optional[str] = None
) -> bool:
    db_processing_logger.info(f"video_id: {video_uuid} - Updating status to '{new_status.value}'. Error: '{error_msg if error_msg else 'None'}'")
    try:
        stmt = select(Video).where(Video.video_uuid == video_uuid)
        result = await session.execute(stmt)
        video_record = result.scalar_one_or_none()

        if video_record:
            video_record.processing_status = new_status
            video_record.error_message = error_msg
            video_record.updated_at = datetime.utcnow() # Explicitly set for onupdate
            await session.flush() # Make changes available for refresh
            await session.refresh(video_record)
            db_processing_logger.info(f"video_id: {video_uuid} - Status updated to {new_status.value} in DB.")
            return True
        else:
            db_processing_logger.warning(f"video_id: {video_uuid} - Video record not found for status update.")
            return False
    except Exception as e:
        db_processing_logger.exception(f"video_id: {video_uuid} - Error updating status.")
        return False

async def update_video_asset_paths_record(
    session: AsyncSession,
    video_uuid: str,
    audio_path: Optional[str] = None,
    frames_dir: Optional[str] = None,
    transcript_path: Optional[str] = None,
    highlight_clip_path: Optional[str] = None # <--- ADD THIS
) -> bool:
    updates_log = []
    if audio_path: updates_log.append(f"audio_path='{audio_path}'")
    if frames_dir: updates_log.append(f"frames_dir='{frames_dir}'")
    if transcript_path: updates_log.append(f"transcript_path='{transcript_path}'")
    if highlight_clip_path: updates_log.append(f"highlight_clip_path='{highlight_clip_path}'") 
    db_processing_logger.info(f"video_id: {video_uuid} - Updating asset paths: {', '.join(updates_log) if updates_log else 'No paths to update'}")

    if not updates_log: # No actual paths provided to update
        return True 

    try:
        stmt = select(Video).where(Video.video_uuid == video_uuid)
        result = await session.execute(stmt)
        video_record = result.scalar_one_or_none()

        if video_record:
            if audio_path is not None:
                video_record.audio_file_path = audio_path
            if frames_dir is not None:
                video_record.frames_directory_path = frames_dir
            if transcript_path is not None:
                video_record.transcript_file_path = transcript_path
            if highlight_clip_path is not None: # <--- ADD THIS
                video_record.generated_highlight_path = highlight_clip_path
            video_record.updated_at = datetime.utcnow()
            await session.flush()
            await session.refresh(video_record)
            db_processing_logger.info(f"video_id: {video_uuid} - Asset paths updated in DB.")
            return True
        else:
            db_processing_logger.warning(f"video_id: {video_uuid} - Video record not found for asset path update.")
            return False
    except Exception as e:
        db_processing_logger.exception(f"video_id: {video_uuid} - Error updating asset paths.")
        return False

async def get_video_record_by_uuid(session: AsyncSession, video_uuid: str) -> Optional[Video]:
    # This function is mostly for internal use by other services if they need to fetch the full record
    # db_processing_logger.debug(f"video_id: {video_uuid} - Fetching video record by UUID.")
    try:
        stmt = select(Video).where(Video.video_uuid == video_uuid)
        result = await session.execute(stmt)
        video_record = result.scalar_one_or_none()
        # if video_record:
        #     db_processing_logger.debug(f"video_id: {video_uuid} - Video record found.")
        # else:
        #     db_processing_logger.debug(f"video_id: {video_uuid} - Video record not found.")
        return video_record
    except Exception as e:
        db_processing_logger.exception(f"video_id: {video_uuid} - Error fetching video record by UUID.")
        return None