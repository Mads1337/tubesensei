# TubeSensei - Project Roadmap & TODO

> Last updated: 2026-02-23

## Priority Legend

| Priority | Meaning | Timeline |
|----------|---------|----------|
| **P0** | Critical - blocks core functionality | Next sprint |
| **P1** | High - key features and important fixes | 1-2 sprints |
| **P2** | Medium - improvements and new capabilities | 2-4 sprints |
| **P3** | Nice-to-have - polish and future ideas | Backlog |

---

## P0: Dashboard Fixes

The admin dashboard is currently broken/incomplete and blocks day-to-day use of the application.

### Re-enable the Dashboard Router

- **Description:** The dashboard router is commented out in `tubesensei/app/api/admin/__init__.py` (lines 8, 32). The admin root (`/admin/`) redirects to `/admin/channels/` instead of showing a proper overview dashboard.
- **Files:**
  - `tubesensei/app/api/admin/__init__.py` - Uncomment dashboard router import and inclusion
  - `tubesensei/app/api/admin/dashboard.py` - Fix whatever caused it to be disabled
  - `templates/admin/dashboard/index.html` - Build or fix the dashboard template
- **Acceptance Criteria:**
  - `/admin/` renders a dashboard page with summary stats (campaigns, videos, ideas, channels)
  - Dashboard shows recent activity and system health at a glance
  - No redirect to `/admin/channels/`

### Fix Settings Page - Add Save Functionality

- **Description:** The settings page at `templates/admin/settings/index.html` is entirely read-only. Users can see configuration but cannot modify anything.
- **Files:**
  - `templates/admin/settings/index.html` - Add form inputs and save button
  - `tubesensei/app/api/admin/settings.py` - Add `POST`/`PUT` endpoint for saving settings
  - `tubesensei/app/core/config.py` - Support runtime config updates (or document which settings require restart)
- **Acceptance Criteria:**
  - Editable settings have form inputs instead of plain text
  - Save button persists changes (at minimum: feature flags, topic discovery limits, logging level)
  - Read-only settings (DB connection, API key status) remain non-editable but clearly marked
  - Success/error feedback on save

### Audit All Admin Pages

- **Description:** Systematically check every admin page for broken links, non-functional buttons, incorrect redirects, and HTMX partial-loading issues.
- **Files:**
  - `templates/admin/` - All template files
  - `tubesensei/app/api/admin/` - All admin routers
- **Acceptance Criteria:**
  - Every link in the admin UI navigates to a working page
  - All action buttons (start, pause, delete, etc.) function correctly
  - HTMX partials load without full-page flashes or errors
  - No 404s or 500s during normal navigation

---

## P1: Single Video URL Analysis (Quick Analysis)

Accept a YouTube video URL, fetch its transcript, and extract ideas - without creating a full campaign.

### Parse & Validate YouTube Video URLs

- **Description:** Accept URLs in all common formats (`youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`, etc.) and extract the video ID.
- **Files:**
  - `tubesensei/app/utils/` - New `url_parser.py` or add to existing utils
- **Acceptance Criteria:**
  - Handles `youtube.com/watch?v=VIDEO_ID`, `youtu.be/VIDEO_ID`, `youtube.com/shorts/VIDEO_ID`
  - Returns clean video ID string
  - Rejects invalid/non-YouTube URLs with clear error

### Quick Analysis API Endpoint

- **Description:** New endpoint that takes a video URL, fetches metadata via `YouTubeAPIClient.get_video_details()` (`tubesensei/app/integrations/youtube_api.py`), extracts the transcript via `TranscriptProcessor.extract_transcript()` (`tubesensei/app/services/transcript_processor.py`), and runs idea extraction via `IdeaExtractionAgent` (`tubesensei/app/agents/idea_extraction_agent.py`).
- **Files:**
  - `tubesensei/app/api/admin/quick_analysis.py` - New router
  - `tubesensei/app/api/admin/__init__.py` - Register the new router
  - `tubesensei/app/models/topic_campaign.py` - Consider adding a `QUICK_VIDEO` campaign type or a lightweight alternative
- **Acceptance Criteria:**
  - `POST /admin/quick-analysis/video` accepts `{ "url": "..." }`
  - Creates a Video record (or reuses existing if already discovered)
  - Extracts transcript and ideas
  - Returns extracted ideas in the response (or provides a job ID for polling)
  - Errors (no captions, private video, invalid URL) return clear messages

### Quick Analysis Dashboard UI

- **Description:** Add a "Quick Analysis" section to the dashboard with a URL input field and submit button.
- **Files:**
  - `templates/admin/dashboard/index.html` - Add quick analysis input section
  - `templates/admin/quick_analysis/` - New templates for results display
  - `static/` - Any needed JS for the async flow
- **Acceptance Criteria:**
  - Input field with "Analyze" button on the dashboard
  - Shows loading state while processing
  - Displays extracted ideas inline or navigates to a results page
  - Error states shown clearly (invalid URL, no transcript available)

---

## P1: Single Channel URL Analysis

Accept a YouTube channel URL, resolve it, list its videos, and batch-process transcripts and ideas.

### Parse & Validate YouTube Channel URLs

- **Description:** Accept channel URLs in all formats (`youtube.com/@handle`, `youtube.com/channel/ID`, `youtube.com/c/name`) and resolve to a channel ID.
- **Files:**
  - `tubesensei/app/utils/url_parser.py` - Extend URL parser (or create if not yet done)
  - `tubesensei/app/integrations/youtube_api.py` - Use `get_channel_by_handle()` (line ~92) for handle resolution
- **Acceptance Criteria:**
  - Handles `@handle`, `/channel/UC...`, `/c/name` formats
  - Resolves handles to channel IDs via the YouTube API
  - Rejects invalid URLs with clear error

### Channel Analysis API Endpoint

- **Description:** Takes a channel URL, resolves to channel ID, calls `YouTubeAPIClient.list_channel_videos()` to discover videos, then batch-processes transcripts and ideas. Internally creates a campaign (type: `QUICK_CHANNEL`) for tracking.
- **Files:**
  - `tubesensei/app/api/admin/quick_analysis.py` - Add channel endpoint to same router
  - `tubesensei/app/models/topic_campaign.py` - Add `QUICK_CHANNEL` type handling
  - `tubesensei/app/services/transcript_processor.py` - Use `batch_process_transcripts()`
  - `tubesensei/app/agents/idea_extraction_agent.py` - Reuse for batch idea extraction
- **Acceptance Criteria:**
  - `POST /admin/quick-analysis/channel` accepts `{ "url": "...", "max_videos": 50 }`
  - Creates Channel record (or reuses existing)
  - Discovers and stores video records
  - Processes transcripts and extracts ideas in background (Celery task)
  - Returns job/campaign ID for progress tracking
  - Configurable video limit (default 50)

### Channel Analysis Dashboard UI

- **Description:** Dashboard section for submitting a channel URL and tracking analysis progress.
- **Files:**
  - `templates/admin/dashboard/index.html` - Add channel analysis section
  - `templates/admin/quick_analysis/` - Channel results templates
- **Acceptance Criteria:**
  - Channel URL input with configurable max videos
  - Progress indicator showing videos discovered / transcribed / ideas extracted
  - Link to view results when complete

---

## P1: Idea Generation System Improvements

The current idea extraction pipeline has several quality issues that reduce the usefulness of extracted ideas.

### Add Idea Deduplication

- **Description:** Currently no deduplication exists - the same idea can appear across chunks of the same video, across videos, and across campaigns. Need deduplication at storage time.
- **Files:**
  - `tubesensei/app/agents/idea_extraction_agent.py` - Add dedup logic in `_process_video()` (line ~153)
  - `tubesensei/app/models/idea.py` - Consider adding a `content_hash` or `embedding` column for similarity matching
- **Acceptance Criteria:**
  - Exact duplicates (same title + description) are never stored twice per video
  - Near-duplicates across videos are flagged or merged (fuzzy title match or embedding similarity)
  - Dedup stats logged per extraction run

### Add Confidence Threshold Filtering

- **Description:** Low-confidence ideas are stored alongside high-quality ones. The `confidence_score` field exists on the Idea model (`tubesensei/app/models/idea.py`, line ~39) but no threshold is enforced.
- **Files:**
  - `tubesensei/app/agents/idea_extraction_agent.py` - Add threshold check before storing
  - `tubesensei/app/models/topic_campaign.py` - Add `confidence_threshold` to campaign config JSONB
- **Acceptance Criteria:**
  - Ideas below the threshold (default 0.5) are discarded or marked as low-quality
  - Threshold is configurable per campaign
  - Filtered-out ideas are logged for debugging

### Improve Transcript Chunking

- **Description:** Current chunking splits transcripts naively without overlap, losing context at chunk boundaries. Ideas spanning two chunks are missed or incomplete.
- **Files:**
  - `tubesensei/app/agents/idea_extraction_agent.py` - Improve chunking in `_extract_ideas_from_text()` (line ~188)
- **Acceptance Criteria:**
  - Chunks have configurable overlap (e.g., 200 tokens)
  - Chunk boundaries prefer sentence or paragraph breaks
  - Long videos produce better idea coverage than before

### Improve Extraction Prompts

- **Description:** The idea extraction prompt (`tubesensei/app/ai/prompt_templates.py`, lines 103-140) is generic. Needs few-shot examples and better guidance for actionable ideas.
- **Files:**
  - `tubesensei/app/ai/prompt_templates.py` - Enhance `IDEA_EXTRACTION` prompt
- **Acceptance Criteria:**
  - Prompt includes 2-3 few-shot examples of well-extracted ideas
  - Guidance distinguishes "actionable business idea" from "general observation"
  - Temperature parameter tunable (currently too low for creative extraction)

### Wire Up Unused Prompt Templates

- **Description:** `QUALITY_ASSESSMENT` and `IDEA_CATEGORIZATION` prompt types exist (`tubesensei/app/ai/prompt_templates.py`, lines 142-218) but are never called anywhere in the codebase.
- **Files:**
  - `tubesensei/app/agents/idea_extraction_agent.py` - Call categorization after extraction
  - New: `tubesensei/app/agents/idea_quality_agent.py` or extend extraction agent - Run quality assessment as post-processing step
  - `tubesensei/app/ai/prompt_templates.py` - Refine prompts as needed
- **Acceptance Criteria:**
  - Every extracted idea gets a quality assessment score
  - Ideas are categorized by industry, business model, revenue model
  - Assessment results stored in the Idea model fields (`complexity_score`, `target_audience`, etc.)

### Fix Error Handling for Failed Videos

- **Description:** When idea extraction fails for a video, it's marked as done (`ideas_extracted = True` on `CampaignVideo`), preventing future retries.
- **Files:**
  - `tubesensei/app/agents/idea_extraction_agent.py` - Fix error handling in `_process_video()` and `run()`
  - `tubesensei/app/models/campaign_video.py` - Ensure failed state is distinguishable from completed
- **Acceptance Criteria:**
  - Failed videos are NOT marked as `ideas_extracted = True`
  - A `retry_count` or `last_error` field tracks failures
  - Re-running the agent picks up previously failed videos

---

## P2: Custom Investigation Agents

Allow users to define custom AI agents that investigate specific ideas in depth.

### Investigation Agent Model

- **Description:** Store agent definitions in PostgreSQL so users can create, edit, and manage investigation agents via the UI.
- **Files:**
  - New: `tubesensei/app/models/investigation_agent.py`
  - `tubesensei/app/models/__init__.py` - Register new model
  - Alembic migration for new table
- **Fields:**
  - `id` (UUID PK)
  - `name` (String) - e.g., "Financial Analysis Agent"
  - `description` (Text)
  - `system_prompt` (Text) - The agent's system prompt
  - `user_prompt_template` (Text) - Template with `{idea_title}`, `{idea_description}`, etc.
  - `config` (JSONB) - Model, temperature, max_tokens, etc.
  - `is_active` (Boolean)
  - `created_at`, `updated_at`
- **Acceptance Criteria:**
  - Model created with migration
  - CRUD operations via SQLAlchemy work correctly

### Investigation Run Model

- **Description:** Track each run of an investigation agent against a specific idea.
- **Files:**
  - New: `tubesensei/app/models/investigation_run.py`
  - `tubesensei/app/models/__init__.py` - Register new model
  - Alembic migration for new table
- **Fields:**
  - `id` (UUID PK)
  - `agent_id` (FK to InvestigationAgent)
  - `idea_id` (FK to Idea)
  - `status` (enum: PENDING, RUNNING, COMPLETED, FAILED)
  - `result` (Text) - The agent's output
  - `result_structured` (JSONB) - Parsed/structured output
  - `tokens_used`, `estimated_cost_usd`
  - `started_at`, `completed_at`
- **Acceptance Criteria:**
  - Model created with migration
  - Linked to both InvestigationAgent and Idea

### Investigation Agent Runner

- **Description:** Service that executes an investigation agent against an idea using the LLM manager.
- **Files:**
  - New: `tubesensei/app/services/investigation_runner.py`
  - `tubesensei/app/ai/llm_manager.py` - Reuse for LLM calls
- **Acceptance Criteria:**
  - Takes an agent definition + idea, formats the prompt, calls LLM
  - Stores result in InvestigationRun
  - Handles errors gracefully (timeout, LLM failure)
  - Tracks token usage and cost

### Pre-built Agent Templates

- **Description:** Seed the database with useful default investigation agents.
- **Templates:**
  - **Financial Analysis** - Revenue potential, cost structure, break-even analysis
  - **Feasibility Study** - Technical feasibility, resource requirements, timeline
  - **Market Research** - Market size, competition, target demographics
  - **Technical Complexity** - Tech stack, dependencies, development effort
  - **Competitive Analysis** - Existing solutions, differentiation, market gaps
- **Files:**
  - New: `tubesensei/app/services/seed_agents.py` or Alembic data migration
- **Acceptance Criteria:**
  - 5 pre-built agents created on first run or via migration
  - Each has well-crafted system and user prompts
  - Users can modify or delete the templates

### Investigation Agent Management UI

- **Description:** Dashboard pages to create, edit, list, and delete investigation agents.
- **Files:**
  - New: `tubesensei/app/api/admin/investigation_agents.py` - Admin router
  - New: `templates/admin/investigation_agents/` - Templates (list, create, edit)
  - `tubesensei/app/api/admin/__init__.py` - Register router
- **Acceptance Criteria:**
  - List page shows all agents with name, description, run count
  - Create/edit form with fields for name, description, system prompt, user prompt template, config
  - Delete with confirmation
  - "Test" button to run agent against a sample idea

### Idea Investigation UI

- **Description:** From an idea's detail page, allow users to select an investigation agent and run it.
- **Files:**
  - `templates/admin/ideas/` - Add "Investigate" section to idea detail
  - `tubesensei/app/api/admin/ideas.py` - Add endpoints for triggering investigations
  - New: `templates/admin/ideas/investigation_results.html` - Results display
- **Acceptance Criteria:**
  - Idea detail page shows dropdown of available agents + "Run Investigation" button
  - Shows past investigation results for the idea
  - Loading state while agent runs
  - Results rendered with markdown formatting

---

## P2: REST API Completion

### Finalize CRUD Endpoints

- **Description:** Ensure all resources have complete CRUD endpoints via the v1 API.
- **Files:**
  - `tubesensei/app/api/v1/topic_campaigns.py` - Existing, expand as needed
  - New: `tubesensei/app/api/v1/ideas.py`
  - New: `tubesensei/app/api/v1/videos.py`
  - New: `tubesensei/app/api/v1/channels.py`
  - `tubesensei/app/main.py` - Has some endpoints at root level (lines 38-325); consider migrating to v1
- **Acceptance Criteria:**
  - All resources (campaigns, ideas, videos, channels) have GET (list + detail), POST, PUT/PATCH, DELETE
  - Consistent response format with pagination
  - Proper HTTP status codes

### API Documentation

- **Description:** Auto-generated API docs via FastAPI's built-in OpenAPI support, plus any needed customization.
- **Files:**
  - `tubesensei/app/main.py` - FastAPI app configuration
  - API router files - Add response models and descriptions
- **Acceptance Criteria:**
  - `/docs` (Swagger UI) and `/redoc` accessible and complete
  - All endpoints have descriptions, request/response examples
  - Authentication documented

### Rate Limiting

- **Description:** Enforce rate limits on API endpoints. Some rate-limiting config exists in settings but may not be enforced.
- **Files:**
  - `tubesensei/app/core/` - Rate limiting middleware
  - `tubesensei/app/core/config.py` - Rate limit settings exist
- **Acceptance Criteria:**
  - Configurable rate limits per endpoint or endpoint group
  - 429 responses with `Retry-After` header
  - Rate limit status in response headers (`X-RateLimit-Remaining`, etc.)

---

## P2: Export & Integration

### IdeaHunter-Compatible Export

- **Description:** Export ideas in a format compatible with IdeaHunter or similar tools.
- **Files:**
  - New: `tubesensei/app/services/export_service.py`
  - New: `tubesensei/app/api/admin/export.py` or `tubesensei/app/api/v1/export.py`
- **Acceptance Criteria:**
  - Export ideas as JSON or CSV
  - Filterable by campaign, status, priority, date range
  - Downloadable from admin UI
  - API endpoint for programmatic access

### Webhooks

- **Description:** Notify external systems when key events occur (new ideas extracted, campaign completed, etc.).
- **Files:**
  - New: `tubesensei/app/models/webhook.py` - Webhook subscription model
  - New: `tubesensei/app/services/webhook_service.py` - Dispatch logic
- **Acceptance Criteria:**
  - Register webhook URLs with event type filters
  - Events: `idea.extracted`, `campaign.completed`, `campaign.failed`
  - Retry on failure with exponential backoff
  - Webhook management in admin UI

### Scheduled Exports

- **Description:** Automatically export ideas on a schedule (daily/weekly digest).
- **Files:**
  - New: Celery beat task for scheduled exports
  - `tubesensei/app/services/export_service.py` - Reuse export logic
- **Acceptance Criteria:**
  - Configurable schedule (daily, weekly)
  - Export to file or send via webhook
  - Email digest option (if email service configured)

---

## P3: Testing & Production Readiness

### Expand Test Coverage

- **Description:** Current test suite exists at `tests/` but coverage needs improvement toward 95%+.
- **Files:**
  - `tests/` - Expand existing test files
  - New test files for untested modules
- **Acceptance Criteria:**
  - Unit tests for all models, services, and agents
  - Integration tests for API endpoints
  - Mock external APIs (YouTube, LLM providers)
  - Coverage report generated and tracked

### End-to-End Workflow Tests

- **Description:** Test complete flows: URL submission -> video discovery -> transcription -> idea extraction -> display.
- **Files:**
  - New: `tests/e2e/` - End-to-end test suite
- **Acceptance Criteria:**
  - Full quick-analysis flow (video + channel)
  - Full campaign flow (create -> run -> complete)
  - Investigation agent flow
  - Tests run in CI

### Production Deployment

- **Description:** Ensure the application is ready for production deployment.
- **Files:**
  - New or update: `docker-compose.prod.yml`
  - New: deployment documentation
  - `tubesensei/app/core/config.py` - Audit for production defaults
- **Acceptance Criteria:**
  - Secret management (no hardcoded keys, environment-based config)
  - HTTPS configuration documented
  - Database migration strategy documented
  - Monitoring and alerting setup (health endpoints already exist)
  - Deployment guide for Docker and bare-metal

---

## Architecture Notes

### Key Files Reference

| Component | Path |
|-----------|------|
| App entrypoint | `tubesensei/app/main.py` |
| Admin router registry | `tubesensei/app/api/admin/__init__.py` |
| Dashboard router (disabled) | `tubesensei/app/api/admin/dashboard.py` |
| Settings router | `tubesensei/app/api/admin/settings.py` |
| YouTube API client | `tubesensei/app/integrations/youtube_api.py` |
| Transcript processor | `tubesensei/app/services/transcript_processor.py` |
| Idea extraction agent | `tubesensei/app/agents/idea_extraction_agent.py` |
| Prompt templates | `tubesensei/app/ai/prompt_templates.py` |
| LLM manager | `tubesensei/app/ai/llm_manager.py` |
| Campaign model | `tubesensei/app/models/topic_campaign.py` |
| Idea model | `tubesensei/app/models/idea.py` |
| Video model | `tubesensei/app/models/video.py` |
| Channel model | `tubesensei/app/models/channel.py` |
| CampaignVideo model | `tubesensei/app/models/campaign_video.py` |
| AgentRun model | `tubesensei/app/models/agent_run.py` |
| Admin templates | `templates/admin/` |
| Settings template | `templates/admin/settings/index.html` |

### Existing Agent Types

The system uses an agent-based architecture with these types defined in `AgentType` enum:

- `COORDINATOR` - Orchestrates entire campaign
- `SEARCH` - YouTube search for initial videos
- `CHANNEL_EXPANSION` - Gets all videos from a channel
- `TOPIC_FILTER` - AI-based relevance filtering
- `TRANSCRIPTION` - Bulk transcription
- `SIMILAR_VIDEOS` - Related videos discovery
- `IDEA_EXTRACTION` - Extract business ideas from transcripts

### Campaign Configuration (JSONB)

```json
{
  "total_video_limit": 3000,
  "per_channel_limit": 5,
  "search_limit": 50,
  "similar_videos_depth": 2,
  "filter_threshold": 0.7,
  "enabled_agents": ["search", "channel_expansion", "topic_filter", "similar_videos"]
}
```
