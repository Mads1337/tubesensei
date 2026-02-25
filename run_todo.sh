#!/bin/bash
# =============================================================================
# TubeSensei Autonomous TODO Runner
# Runs docs/TODO.md in phases, each with fresh context.
# Uses tmux/screen - detach and come back later.
# =============================================================================

set -e

# --- Keep Mac awake until script finishes ---
# caffeinate -i prevents idle sleep, -s prevents system sleep on AC power
# Runs in background and gets killed when this script exits
caffeinate -dims -w $$ &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

echo "Mac sleep disabled for this session (caffeinate PID: $CAFFEINATE_PID)"

# --- Start required services (same as run.sh but without Honcho) ---
echo "Checking Docker is running..."
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "Starting Docker services (Postgres, Redis, etc.)..."
docker-compose up -d

# Wait for PostgreSQL (port 5433)
echo "Waiting for PostgreSQL..."
for i in {1..30}; do
    if PGPASSWORD=tubesensei_dev psql -h localhost -p 5433 -U tubesensei -d tubesensei -c "SELECT 1;" >/dev/null 2>&1; then
        echo "PostgreSQL is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: PostgreSQL failed to start after 60s"
        exit 1
    fi
    sleep 2
done

# Wait for Redis (port 6379)
echo "Waiting for Redis..."
for i in {1..15}; do
    if timeout 2 bash -c "</dev/tcp/localhost/6379" >/dev/null 2>&1; then
        echo "Redis is ready."
        break
    fi
    sleep 1
done

# Run database migrations
echo "Running database migrations..."
cd tubesensei
alembic upgrade head 2>/dev/null || echo "WARNING: Migrations failed or none needed"
cd ..

echo "All services ready."

# --- Configuration ---
MAX_TURNS=150           # Per phase. Increase if phases are timing out.
BRANCH="auto/todo-impl" # Work branch (keeps main safe)
LOG_DIR="logs/auto-run"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Tools to pre-approve (no permission prompts)
ALLOWED_TOOLS="Bash,Read,Write,Edit,Glob,Grep,Task,TaskCreate,TaskUpdate,TaskList,TaskGet,NotebookEdit,mcp__ide__getDiagnostics"

# Phase checkpoint file - tracks which phases are done
CHECKPOINT_FILE="$LOG_DIR/.completed_phases"

# --- Setup ---
mkdir -p "$LOG_DIR"
touch "$CHECKPOINT_FILE"

# Create work branch if not already on it
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo "Creating work branch: $BRANCH"
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
fi

# --- Shared preamble for every phase prompt ---
PREAMBLE="You are implementing tasks from the project TODO. IMPORTANT RULES:

1. PARALLELISM: Use the Task tool with subagent_type='general-purpose' to run independent subtasks in parallel. For example, if a phase has 3 independent tasks, launch 3 Task agents simultaneously.

2. EXPLORATION FIRST: Before writing any code for a task, use a Task agent with subagent_type='Explore' to understand the relevant files and patterns. Feed ONLY the relevant context to each agent - do NOT dump the entire TODO.

3. TESTING: After implementing each task, run tests with 'python -m pytest tests/ -x -q'. If tests fail, fix before moving on.

4. PROGRESS TRACKING: Use TaskCreate/TaskUpdate to track subtasks. Mark each as completed when done.

5. GIT COMMITS - IMPORTANT:
   - NEVER commit from inside a Task subagent. Subagents only write code.
   - After a subagent (or group of parallel subagents) completes, YOU (the orchestrator) should run tests, then commit all their changes together with a descriptive message.
   - Commit after each logical milestone (e.g., after parallel tasks 1+2 finish, commit before starting task 3).
   - Use specific file paths in 'git add' — do NOT use 'git add -A' or 'git add .' to avoid committing unintended files.
   - Example flow: launch parallel subagents -> wait for all -> run tests -> git add specific files -> git commit

6. ERROR RECOVERY: If something fails after 2 attempts, skip it, log what went wrong in a file called 'docs/AUTO_RUN_ISSUES.md', and move to the next task.

7. DO NOT over-engineer. Implement exactly what the acceptance criteria specify, nothing more.

8. When using Task subagents, give them ONLY the context they need - the specific task description, relevant file paths, and acceptance criteria. Do NOT pass the entire TODO.

9. SUBAGENT MODEL SELECTION: When launching Task subagents, choose the model wisely:
   - Use model='sonnet' for Explore agents and routine implementation (CRUD, templates, tests, simple edits, file search)
   - Use model='opus' for subagents doing complex architectural work (designing new systems, complex logic, multi-file coordination)
   - Default to 'sonnet' unless the task clearly requires deeper reasoning."

# --- Phase Definitions ---
# Each phase = one fresh claude invocation = clean context

run_phase() {
    local phase_name="$1"
    local phase_prompt="$2"
    local model="${3:-claude-sonnet-4-6}"  # Default to sonnet 4.6, override with opus where needed
    local log_file="$LOG_DIR/${TIMESTAMP}_${phase_name}.log"

    # Skip if already completed
    if grep -q "^${phase_name}$" "$CHECKPOINT_FILE" 2>/dev/null; then
        echo ""
        echo "  SKIPPING $phase_name (already completed)"
        echo ""
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "  PHASE: $phase_name"
    echo "  Model: $model"
    echo "  Log:   $log_file"
    echo "  Time:  $(date)"
    echo "=============================================="
    echo ""

    # CLAUDECODE="" allows claude to run outside of a parent claude session
    # Output only appears when the phase finishes (claude -p buffers output)
    # To watch progress: tail -f <log_file>
    echo "  Running... (to watch progress: tail -f $log_file)"

    CLAUDECODE="" claude -p "$phase_prompt" \
        --model "$model" \
        --allowedTools "$ALLOWED_TOOLS" \
        --max-turns "$MAX_TURNS" \
        > "$log_file" 2>&1 || true

    echo ""
    echo "  Output (last 20 lines):"
    tail -20 "$log_file"

    # Auto-commit any uncommitted work from this phase
    if [ -n "$(git status --porcelain)" ]; then
        git add -A
        git commit -m "auto: complete phase $phase_name [autonomous run]" --no-verify 2>/dev/null || true
    fi

    # Mark phase as completed
    echo "$phase_name" >> "$CHECKPOINT_FILE"

    echo ""
    echo "Phase $phase_name finished with exit code: $exit_code"
    echo ""

    return 0  # Don't stop the whole script if a phase fails
}

# =============================================================================
# PHASE 1: P0 - Dashboard Fixes
# =============================================================================
run_phase "p0-dashboard-fixes" "$PREAMBLE

YOUR TASK: Implement all P0 Dashboard Fixes. There are 3 tasks:

TASK 1 - Re-enable the Dashboard Router:
- Uncomment dashboard router in tubesensei/app/api/admin/__init__.py (lines 8, 32)
- Fix tubesensei/app/api/admin/dashboard.py so it works
- Build/fix templates/admin/dashboard/index.html with summary stats
- /admin/ should render a dashboard, NOT redirect to /admin/channels/

TASK 2 - Fix Settings Page (Save Functionality):
- Add form inputs and save button to templates/admin/settings/index.html
- Add POST/PUT endpoint in tubesensei/app/api/admin/settings.py
- Support runtime config updates in tubesensei/app/core/config.py
- Editable: feature flags, topic discovery limits, logging level
- Read-only: DB connection, API key status

TASK 3 - Audit All Admin Pages:
- Check every admin page for broken links, non-functional buttons, incorrect redirects
- Fix any HTMX partial-loading issues
- No 404s or 500s during normal navigation

Tasks 1 and 2 are independent - run them in parallel using Task agents. Task 3 should run after both are done." \
"claude-sonnet-4-6"

# =============================================================================
# PHASE 2: P1 - Quick Analysis (Video + Channel)
# =============================================================================
run_phase "p1-quick-analysis" "$PREAMBLE

YOUR TASK: Implement P1 Quick Analysis features. There are 6 tasks in 2 groups:

GROUP A - Single Video URL Analysis (3 tasks):
1. Create URL parser (tubesensei/app/utils/url_parser.py) handling youtube.com/watch?v=, youtu.be/, youtube.com/shorts/
2. Quick Analysis API endpoint: POST /admin/quick-analysis/video - fetches metadata, transcript, extracts ideas
3. Quick Analysis Dashboard UI: URL input + results display on dashboard

GROUP B - Single Channel URL Analysis (3 tasks):
1. Extend URL parser for channel URLs (@handle, /channel/ID, /c/name)
2. Channel Analysis API: POST /admin/quick-analysis/channel with background processing
3. Channel Analysis Dashboard UI: channel input + progress tracking

GROUP A must complete before GROUP B (channel analysis extends video analysis).
Within each group, the URL parser must be done first, then API and UI can be parallel." \
"claude-opus-4-6"

# =============================================================================
# PHASE 3: P1 - Idea Generation Improvements
# =============================================================================
run_phase "p1-idea-improvements" "$PREAMBLE

YOUR TASK: Implement P1 Idea Generation System Improvements. There are 6 tasks:

1. Add Idea Deduplication: Add content_hash/embedding column to Idea model, dedup in idea_extraction_agent.py _process_video()
2. Add Confidence Threshold Filtering: Enforce confidence_score threshold (default 0.5), configurable per campaign
3. Improve Transcript Chunking: Add overlap (200 tokens), prefer sentence boundaries in _extract_ideas_from_text()
4. Improve Extraction Prompts: Add few-shot examples to IDEA_EXTRACTION prompt in prompt_templates.py
5. Wire Up Unused Prompts: Call QUALITY_ASSESSMENT and IDEA_CATEGORIZATION prompts, store results in Idea fields
6. Fix Error Handling: Failed videos should NOT be marked ideas_extracted=True, add retry_count/last_error

Tasks 1-4 are independent - run them in parallel. Task 5 depends on task 4. Task 6 is independent.
Create an alembic migration for any model changes." \
"claude-sonnet-4-6"

# =============================================================================
# PHASE 4: P2 - Custom Investigation Agents
# =============================================================================
run_phase "p2-investigation-agents" "$PREAMBLE

YOUR TASK: Implement P2 Custom Investigation Agents. There are 6 tasks in order:

1. Investigation Agent Model: New model in tubesensei/app/models/investigation_agent.py with fields (id, name, description, system_prompt, user_prompt_template, config JSONB, is_active, timestamps). Create alembic migration.

2. Investigation Run Model: New model in tubesensei/app/models/investigation_run.py (id, agent_id FK, idea_id FK, status enum, result, result_structured JSONB, tokens_used, estimated_cost_usd, timestamps). Create alembic migration.

3. Investigation Agent Runner: New service tubesensei/app/services/investigation_runner.py using llm_manager.py for LLM calls.

4. Pre-built Agent Templates: Seed 5 agents (Financial Analysis, Feasibility Study, Market Research, Technical Complexity, Competitive Analysis).

5. Investigation Agent Management UI: Admin pages to CRUD agents at /admin/investigation-agents/

6. Idea Investigation UI: Add 'Investigate' section to idea detail page with agent dropdown and results display.

Tasks 1+2 are parallel (both are models). Task 3 depends on 1+2. Task 4 depends on 1. Tasks 5+6 depend on 3." \
"claude-opus-4-6"

# =============================================================================
# PHASE 5: P2 - REST API + Export
# =============================================================================
run_phase "p2-api-and-export" "$PREAMBLE

YOUR TASK: Implement P2 REST API Completion and Export features. There are 6 tasks:

REST API (3 tasks):
1. Finalize CRUD Endpoints: Complete CRUD for all resources (campaigns, ideas, videos, channels) in v1 API with pagination
2. API Documentation: Ensure /docs and /redoc are complete with descriptions and examples
3. Rate Limiting: Enforce configurable rate limits, 429 responses with Retry-After header

Export & Integration (3 tasks):
4. IdeaHunter Export: Export ideas as JSON/CSV, filterable by campaign/status/priority/date
5. Webhooks: Webhook model + dispatch for idea.extracted, campaign.completed, campaign.failed events
6. Scheduled Exports: Celery beat task for daily/weekly exports

REST API tasks 1-3 are independent - run in parallel.
Export tasks: 4 is independent, 5 is independent, 6 depends on 4.
Both groups are independent of each other - maximize parallelism." \
"claude-sonnet-4-6"

# =============================================================================
# PHASE 6: P3 - Testing & Production Readiness
# =============================================================================
run_phase "p3-testing-production" "$PREAMBLE

YOUR TASK: Implement P3 Testing & Production Readiness. There are 3 tasks:

1. Expand Test Coverage: Unit tests for all models, services, agents. Integration tests for API endpoints. Mock external APIs. Target 95%+ coverage. Run: python -m pytest tests/ --cov=tubesensei --cov-report=term-missing

2. End-to-End Workflow Tests: Create tests/e2e/ with full flow tests - quick analysis (video+channel), campaign flow, investigation agent flow.

3. Production Deployment: docker-compose.prod.yml, audit config for production defaults, document deployment.

Tasks 1 and 2 are related but can be parallel (unit vs e2e). Task 3 is independent." \
"claude-sonnet-4-6"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=============================================="
echo "  ALL PHASES COMPLETE"
echo "  Time: $(date)"
echo "  Branch: $BRANCH"
echo "  Logs: $LOG_DIR/"
echo "=============================================="
echo ""
echo "Review changes:"
echo "  git log main..$BRANCH --oneline"
echo "  git diff main..$BRANCH --stat"
echo ""
echo "If happy, merge:"
echo "  git checkout main && git merge $BRANCH"
