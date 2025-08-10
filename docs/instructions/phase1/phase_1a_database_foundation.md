# Phase 1A: Database Foundation & Setup

## Overview
This phase focuses on establishing the core database infrastructure and project structure for TubeSensei. Expected duration: 1 week.

## Prerequisites
- Python 3.11+ installed
- Docker and Docker Compose installed
- PostgreSQL client tools (optional but recommended)
- Basic understanding of SQLAlchemy and FastAPI

## Goals
- Set up project structure and development environment
- Create all database models with proper relationships
- Implement database migrations with Alembic
- Establish async database connectivity

## Task 1: Project Setup

### 1.1 Initialize Project Structure
```bash
mkdir -p tubesensei/app/{models,services,integrations,workers,utils}
mkdir -p tubesensei/{alembic,tests}
```

### 1.2 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install Core Dependencies
Create `requirements.txt`:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1
pydantic==2.5.2
pydantic-settings==2.1.0
python-dotenv==1.0.0
psycopg2-binary==2.9.9
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
ruff==0.1.6
mypy==1.7.1
```

### 1.4 Docker Compose Setup
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_USER: tubesensei
      POSTGRES_PASSWORD: tubesensei_dev
      POSTGRES_DB: tubesensei
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 1.5 Environment Configuration
Create `.env.example`:
```env
DATABASE_URL=postgresql+asyncpg://tubesensei:tubesensei_dev@localhost:5433/tubesensei
DATABASE_POOL_SIZE=20
LOG_LEVEL=INFO
```

## Task 2: Database Models Implementation

### 2.1 Create Base Model
`app/models/base.py`:
- UUID primary key base
- Created/updated timestamps
- SQLAlchemy declarative base

### 2.2 Channel Model
`app/models/channel.py`:
- All fields as specified in requirements
- Status enum (active, paused, inactive)
- JSONB fields for metadata and processing_config
- Relationship to videos

### 2.3 Video Model
`app/models/video.py`:
- All fields as specified
- Status enum (discovered, queued, processing, completed, failed)
- Foreign key to channels
- Relationship to transcripts
- Tags as array field

### 2.4 Transcript Model
`app/models/transcript.py`:
- Content storage
- Source enum (youtube_auto, youtube_manual)
- Foreign key to videos
- JSONB metadata field

### 2.5 Processing Models
`app/models/processing_job.py`:
- Job tracking with status enum
- Entity polymorphic reference
- Retry logic fields
- JSONB result storage

`app/models/processing_session.py`:
- Session tracking
- Progress metrics
- Configuration storage

## Task 3: Database Setup & Migrations

### 3.1 Database Configuration
`app/database.py`:
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    echo=settings.DEBUG
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)
```

### 3.2 Alembic Configuration
```bash
alembic init alembic
```
Configure for async operations and auto-generate migrations

### 3.3 Create Initial Migration
```bash
alembic revision --autogenerate -m "Initial database schema"
alembic upgrade head
```

### 3.4 Database Indexes
Add performance indexes:
- Channel: youtube_channel_id (unique)
- Video: youtube_video_id (unique), channel_id, status
- Transcript: video_id
- ProcessingJob: entity_id, status, job_type

## Testing Checklist

### Unit Tests
- [ ] All models instantiate correctly
- [ ] Relationships work bidirectionally
- [ ] Enum validations function properly
- [ ] JSONB fields serialize/deserialize correctly

### Integration Tests
- [ ] Database connects successfully
- [ ] Migrations run without errors
- [ ] CRUD operations work for all models
- [ ] Async sessions handle properly

## Deliverables
1. Complete project structure with all directories
2. All 5 database models implemented with relationships
3. Working Alembic migrations
4. Async database connectivity
5. Basic CRUD utility functions
6. Test suite for models (80% coverage minimum)

## Success Criteria
- [ ] Docker Compose starts PostgreSQL and Redis successfully
- [ ] All database models created with proper fields and types
- [ ] Alembic migrations run without errors
- [ ] Can perform basic CRUD operations on all models
- [ ] All relationships function correctly
- [ ] Tests pass with >80% coverage

## Common Issues & Solutions

### Issue: Alembic can't detect models
**Solution**: Ensure all models are imported in `app/models/__init__.py`

### Issue: Async database connection errors
**Solution**: Use `asyncpg` driver in DATABASE_URL, not `psycopg2`

### Issue: JSONB fields not working
**Solution**: Use `sqlalchemy.dialects.postgresql.JSONB` type

## Next Steps
After completing Phase 1A:
1. Verify all database operations work correctly
2. Run the full test suite
3. Document any deviations from the plan
4. Proceed to Phase 1B: YouTube Integration & Discovery