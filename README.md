# TubeSensei - Phase 1A: Database Foundation

## Overview
TubeSensei is an automated YouTube content analysis platform that discovers, transcribes, and extracts business ideas from video content across multiple channels.

## Phase 1A Completion Status
✅ **All Phase 1A tasks have been completed:**
- Project structure created
- All database models implemented (Channel, Video, Transcript, ProcessingJob, ProcessingSession)
- Async database connectivity configured
- Alembic migrations set up
- CRUD utility functions created
- Test suite implemented

## Project Structure
```
tubesensei/
├── app/
│   ├── models/           # Database models
│   ├── services/          # Business logic
│   ├── integrations/      # External integrations
│   ├── workers/           # Background workers
│   └── utils/             # Utility functions (CRUD, etc.)
├── alembic/               # Database migrations
├── tests/                 # Test suite
├── docker-compose.yml     # Docker configuration
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Database Services
```bash
docker-compose up -d
```
This will start PostgreSQL on port 5433 and Redis on port 6379.

### 3. Create Database Migration
```bash
cd tubesensei
alembic revision --autogenerate -m "Initial database schema"
alembic upgrade head
```

### 4. Verify Database Setup
```bash
cd tubesensei
python init_db.py
```

### 5. Run Tests
```bash
cd tubesensei
pytest tests/ -v
```

## Database Models

### Channel
- Tracks YouTube channels with monitoring configuration
- Status: active, paused, inactive
- Includes priority levels and check frequency

### Video
- Stores video metadata and processing status
- Status: discovered, queued, processing, completed, failed
- Tracks valuable content with scoring

### Transcript
- Stores video transcripts from various sources
- Supports multiple languages
- Includes metadata and segments

### ProcessingJob
- Tracks individual processing tasks
- Includes retry logic and progress tracking
- Polymorphic entity references

### ProcessingSession
- Groups related jobs for bulk operations
- Tracks overall progress and statistics
- Supports pause/resume functionality

## Key Features Implemented

1. **UUID Primary Keys**: All models use UUID for primary keys
2. **Async Support**: Full async/await support with SQLAlchemy 2.0
3. **JSONB Fields**: Flexible metadata storage using PostgreSQL JSONB
4. **Relationships**: Proper foreign keys and bidirectional relationships
5. **Indexes**: Performance indexes on frequently queried fields
6. **CRUD Operations**: Generic and model-specific CRUD utilities
7. **Test Coverage**: Comprehensive test suite for models and CRUD operations

## Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (with asyncpg driver)
- `DATABASE_POOL_SIZE`: Connection pool size (default: 20)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DEBUG`: Debug mode flag (default: False)

## Next Steps
With Phase 1A complete, you can proceed to:
- **Phase 1B**: YouTube Integration & Discovery
- **Phase 1C**: Transcript Processing
- **Phase 1D**: Job Queue & Testing

## Common Commands

### Database Migrations
```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_models.py -v
```

## Troubleshooting

### Docker Issues
If PostgreSQL fails to start, check if port 5433 is already in use:
```bash
docker-compose down
docker-compose up -d
```

### Migration Issues
If Alembic can't detect models, ensure all models are imported in `app/models/__init__.py`

### Async Database Errors
Make sure to use `postgresql+asyncpg://` in DATABASE_URL, not `postgresql://`