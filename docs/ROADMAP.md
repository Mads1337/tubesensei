# TubeSensei - Project Roadmap

> A comprehensive record of everything built in TubeSensei — a YouTube content analysis platform with AI-driven discovery and idea extraction.

---

## Phase 1: Core Foundation

### Database & Models

- **Base model** with UUID primary keys and automatic `created_at`/`updated_at` timestamps
- **User** model with JWT auth, role-based access (admin/user/viewer), 2FA support, API key generation, login tracking, and account locking
- **Channel** model storing YouTube channel metadata — subscriber/view counts, priority levels, status tracking, processing config
- **Video** model with YouTube ID, duration, view counts, transcript/idea relationships, valuable scoring
- **Transcript** model supporting multiple sources (YouTube auto/manual, Whisper, custom), 10+ languages, word/char counts, segment data
- **Filters** model for domain/channel content exclusion
- Async SQLAlchemy 2.0 with PostgreSQL, Alembic migration system

### YouTube Integration

- Complete YouTube Data API v3 wrapper — search, channel videos, video details, similar videos
- Quota management and tracking
- Proxy support for transcript extraction
- Rate limiting and retry logic with error handling

### Transcript Processing Pipeline

- `TranscriptProcessor` — cleaning, chunking, validation
- `TranscriptAnalyzer` — content analysis for insights
- `TranscriptCache` — performance caching layer
- `TranscriptCleaner` — text normalization
- Compatibility with youtube-transcript-api v0.10.0+

---

## Phase 2: Agent-Based Discovery System

### Agent Framework

- `BaseAgent` abstract class with event emission, rate limiting, and cancellation support
- `AgentContext` shared execution context with campaign, database, rate limiters, and event callbacks
- `AgentResult` with metrics — items processed, API/LLM calls, tokens, cost, duration
- `AgentEvent` system for progress tracking (started, progress, item_discovered, rate_limited, error, completed)
- `AgentRun` model tracking all agent executions with status, metrics, checkpoints, and cost

### Discovery Agents (7 agents)

1. **CoordinatorAgent** — orchestrates the full pipeline, manages discovery loop, monitors limits, handles pause/resume/cancel
2. **SearchAgent** — YouTube search for initial videos matching a topic
3. **ChannelExpansionAgent** — fetches all videos from discovered channels
4. **SimilarVideosAgent** — finds related videos recursively (configurable depth)
5. **TopicFilterAgent** — AI-based relevance filtering using LLM scoring
6. **TranscriptionAgent** — bulk transcript extraction with batch processing
7. **IdeaExtractionAgent** — extracts business ideas from transcripts with configurable chunking

### Campaign System

- **TopicCampaign** model with full lifecycle — draft, running, paused, completed, failed, cancelled
- Configurable discovery limits (`total_video_limit`, `per_channel_limit`, `search_limit`, `similar_videos_depth`)
- Progress tracking — videos discovered/relevant/filtered, channels explored, transcripts extracted
- Checkpoint/resume support so campaigns can pause and pick up where they left off
- Heartbeat mechanism for stale campaign detection
- API/LLM call usage tracking with cost estimation and ETA calculation
- **CampaignVideo** junction table — discovery source tracking, AI relevance scoring, transcript & idea extraction status
- **CampaignChannel** junction table — expansion status, per-channel video counts

### Background Processing (Celery + Redis)

- Task routing to dedicated queues (discovery, transcripts, batch)
- `run_topic_campaign_task` — main campaign runner
- `run_transcription_campaign_task` — bulk transcription
- `process_campaign_transcripts_task` — transcript processing
- `extract_campaign_ideas_task` — idea extraction
- Agent-specific tasks for search, channel expansion, filtering, similar videos
- `discover_channel_videos_task`, `extract_transcript_task`, `extract_transcript_batch_task`
- Worker manager for lifecycle and task distribution
- Monitoring with health checks, performance metrics, and error rate tracking

---

## Phase 3: AI & LLM Infrastructure

### LLM Manager

- Multi-model support — fast, balanced, and quality model tiers
- Token usage tracking and cost estimation
- Retry strategies with exponential backoff
- Rate limiting for LLM API calls

### Prompt Templates

- Idea extraction prompts (14+ templates) with few-shot examples
- Topic filtering prompts
- Quality assessment prompts
- Idea categorization prompts
- Structured output parsing

### Idea Model

- **Idea** model with status workflow (extracted → reviewed → selected/rejected)
- Priority levels, confidence scores, complexity scoring
- Market size, monetization strategies, competitive advantage fields
- Target audience and related ideas tracking
- `content_hash` for deduplication

---

## Phase 4: Admin Dashboard

### Core Admin Pages

- **Dashboard** (`/admin/`) — overview with summary stats for campaigns, videos, ideas, channels, and recent activity
- **Topic Campaigns** — list, create, edit, detail views with tabs for videos, channels, agent runs, and ideas
- **Videos** — list and detail views with transcript info
- **Ideas** — list with filtering by status/priority/campaign, expandable details, bulk actions
- **Channels** — list with statistics, add channel form
- **Transcripts** — list and detail views with reprocessing
- **Jobs** — queue management with retry/cancel controls
- **Monitoring** — real-time system monitoring dashboard
- **Settings** — configuration management with runtime overrides and save functionality
- 47 HTML templates total, built with HTMX for partial loading

### Authentication

- JWT-based auth with bcrypt password hashing
- Role-based permissions system
- Session management with activity tracking

### Real-Time Updates

- WebSocket endpoint for live campaign progress
- Agent execution status streaming
- HTMX partial loading for responsive UI without full-page reloads

---

## Phase 5: Quick Analysis

### Video Quick Analysis

- YouTube URL parser supporting all formats (`youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`)
- `POST /admin/quick-analysis/analyze-video` — accepts URL, fetches metadata, extracts transcript, runs idea extraction
- Dashboard UI with URL input, loading state, and inline results display
- Error handling for invalid URLs, private videos, missing captions

### Channel Quick Analysis

- Channel URL parser for all formats (`@handle`, `/channel/ID`, `/c/name`)
- `POST /admin/quick-analysis/analyze-channel` — resolves channel, discovers videos, batch-processes transcripts and ideas
- Background processing with progress polling (`GET /channel/{job_id}/progress`)
- Results page with video and idea summaries
- Configurable video limit (default 50)

---

## Phase 6: Idea Generation Improvements

### Deduplication

- `content_hash` column on the Idea model
- Exact duplicate prevention (same title + description) per video
- Near-duplicate detection across videos
- Dedup stats logged per extraction run

### Quality & Filtering

- Confidence threshold filtering (default 0.5, configurable per campaign)
- Low-confidence ideas discarded or marked as low-quality
- Quality assessment and categorization using previously unused prompt templates
- Ideas scored for complexity, market size, and competitive advantage

### Transcript Chunking

- Configurable overlap between chunks (e.g., 200 tokens)
- Chunk boundaries prefer sentence or paragraph breaks
- Better idea coverage for long videos

### Extraction Prompts

- Few-shot examples added to extraction prompts
- Guidance distinguishing actionable business ideas from general observations
- Tunable temperature parameter

### Error Handling

- Failed videos no longer incorrectly marked as `ideas_extracted = True`
- `idea_extraction_retry_count` and `idea_extraction_last_error` columns on CampaignVideo
- Re-running the agent picks up previously failed videos

---

## Phase 7: Custom Investigation Agents

### Models

- **InvestigationAgent** — name, description, system prompt, user prompt template, config (model, temperature, max_tokens), active status
- **InvestigationRun** — links agent + idea, tracks status (pending/running/completed/failed), stores raw and structured results, token usage, cost estimation

### Investigation Runner Service

- Formats prompts with idea context, calls LLM, stores results
- Handles errors gracefully (timeout, LLM failure)
- Tracks token usage and estimated cost

### Pre-Built Agent Templates (5 seeded)

1. **Financial Analysis** — revenue potential, cost structure, break-even analysis
2. **Feasibility Study** — technical feasibility, resource requirements, timeline
3. **Market Research** — market size, competition, target demographics
4. **Technical Complexity** — tech stack, dependencies, development effort
5. **Competitive Analysis** — existing solutions, differentiation, market gaps

### Management UI

- List page showing all agents with name, description, and run count
- Create/edit forms for name, description, system prompt, user prompt template, config
- Delete with confirmation
- Idea detail page with agent dropdown and "Run Investigation" button
- Past investigation results displayed per idea with markdown rendering

---

## Phase 8: REST API & Integrations

### Public REST API (v1)

- **Topic Campaigns** — full CRUD + start/pause/resume/cancel + progress/videos/channels/agent-runs/export
- **Videos** — list (with filtering), detail, update, transcripts
- **Ideas** — list (with filtering), detail, review/select/reject
- **Channels** — list, detail, add
- **Webhooks** — create, list, delete
- **Export** — ideas as JSON or CSV with filtering by campaign, status, priority, date range
- **Auth** — register, login (JWT), refresh, logout, me, change-password
- Consistent response format with pagination and proper HTTP status codes
- Auto-generated OpenAPI docs at `/docs` (Swagger) and `/redoc`

### Rate Limiting

- Configurable rate limits per endpoint group
- 429 responses with `Retry-After` header
- `X-RateLimit-Remaining` response headers

### Webhooks

- Webhook subscription model with event type filters
- Events: `idea.extracted`, `campaign.completed`, `campaign.failed`
- Retry on failure with exponential backoff
- Webhook management via API

### Export Service

- Export ideas as JSON or CSV
- Filterable by campaign, status, priority, date range
- Downloadable from admin UI and via API
- Scheduled daily and weekly exports via Celery Beat with webhook dispatch

---

## Phase 9: Testing & Production Readiness

### Test Suite

- **Unit tests** for all models — TopicCampaign, Video, Channel, Idea, Transcript, AgentRun, InvestigationAgent, InvestigationRun
- **Agent tests** — TopicFilterAgent, IdeaExtractionAgent
- **Integration tests** — YouTube API, Transcript API
- **Service tests** — ChannelManager, IdeaService, InvestigationRunner
- **End-to-end tests** — full campaign flow, quick analysis flow (video + channel), investigation agent flow
- Mock fixtures for external APIs (YouTube, LLM providers)

### Production Deployment

- **Dockerfile** — multi-stage build, Python 3.12-slim, non-root user, Gunicorn + Uvicorn (4 workers)
- **docker-compose.prod.yml** — FastAPI app, Celery workers (discovery/transcripts/batch queues), PostgreSQL, Redis, Flower monitoring, Prometheus metrics, Nginx reverse proxy
- **nginx.conf** — reverse proxy configuration
- **`.env.production.example`** — production environment template
- **`docs/DEPLOYMENT.md`** — deployment guide for Docker and bare-metal
- Health check endpoints for monitoring
- Idempotent Alembic migrations for safe re-runs

---

## Architecture Summary

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI (async) |
| Database | PostgreSQL + SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Task queue | Celery + Redis |
| Frontend | Jinja2 templates + HTMX |
| Real-time | WebSocket |
| Auth | JWT + bcrypt |
| AI/LLM | Multi-model support (fast/balanced/quality) |
| YouTube | Data API v3 + transcript extraction |
| Monitoring | Flower + Prometheus |
| Reverse proxy | Nginx |
| Containerization | Docker + Docker Compose |

### Key Architectural Decisions

- **Async throughout** — all database and HTTP operations are non-blocking
- **Agent-based pipeline** — modular, reusable discovery agents with event emission
- **Checkpoint/resume** — campaigns and agents preserve state for pause and resume
- **Cost tracking** — token usage and LLM cost estimation at every layer
- **Multi-source discovery** — search, channel expansion, and similar videos feed the pipeline
- **Distributed processing** — Celery workers with dedicated queues for different workload types
- **Runtime configuration** — settings modifiable without restart via the admin UI
