# TubeSensei - Complete Getting Started Guide

**🚀 Turn YouTube Content Into Business Ideas with AI**

TubeSensei is an intelligent YouTube content analysis platform that automatically discovers videos, extracts transcripts, and uses AI to identify actionable business ideas from video content.

---

## 🎯 What TubeSensei Does

### The Complete Pipeline
1. **🔍 Channel Discovery** - Add YouTube channels and automatically discover all their videos
2. **📝 Transcript Extraction** - Extract and clean transcripts from videos (multiple languages supported)  
3. **🤖 AI Idea Extraction** - Use LLM models (GPT, Claude, Gemini) to identify business ideas from transcripts
4. **📊 Idea Management** - Review, categorize, and export discovered ideas through a web interface
5. **⚡ Scalable Processing** - Handle multiple channels simultaneously with distributed workers

### Key Features
- **Multi-LLM Support**: OpenAI GPT, Anthropic Claude, Google Gemini
- **Admin Dashboard**: Web interface for managing channels and reviewing ideas
- **Real-time Monitoring**: Track processing progress with Flower dashboard
- **Export Capabilities**: Export ideas to JSON, CSV for further analysis
- **Quality Scoring**: AI confidence scores and manual review workflow

---

## 📋 Prerequisites

### Required Software
- **Python 3.11+** 
- **Docker & Docker Compose**
- **Git** for cloning the repository

### Required API Keys
- **YouTube Data API v3 Key** - [Get from Google Cloud Console](https://console.cloud.google.com/apis/credentials)
- **At least one LLM API Key**:
  - OpenAI API Key (recommended)
  - Anthropic Claude API Key
  - Google Gemini API Key

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB+ for database and processing
- **CPU**: 4 cores recommended for optimal performance

---

## ⚙️ Environment Setup

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd TubeSensei

# Install Python dependencies
pip install -r requirements.txt

# Install honcho process manager (included in requirements.txt)
# This enables the single-command startup
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://tubesensei:tubesensei_dev@localhost:5433/tubesensei

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# YouTube API Configuration
YOUTUBE_API_KEY=your_youtube_api_key_here
YOUTUBE_QUOTA_PER_DAY=10000

# LLM API Keys (add at least one)
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-gemini-key-here

# Optional: Additional LLM Providers
DEEPSEEK_API_KEY=your-deepseek-key-here
SERPER_API_KEY=your-serper-key-here

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Monitoring
FLOWER_BASIC_AUTH=admin:admin

# Processing Configuration
WORKER_CONCURRENCY=4
MAX_CONCURRENT_JOBS=10
TRANSCRIPT_QUALITY_THRESHOLD=0.6
```

### 3. Initialize Database

```bash
cd tubesensei

# Run database migrations
alembic upgrade head

# Initialize database with sample data
python init_db.py
```

---

## 🚀 Starting the System

**NEW: Single Command Startup!** 🎉

You now only need **ONE terminal** to run the entire TubeSensei system:

### Single Terminal: All Services

```bash
# Start everything with one command
./run.sh
```

That's it! This will:
1. Start all Docker services (PostgreSQL, Redis, Flower, Prometheus)
2. Wait for services to be healthy
3. Run database migrations
4. Start all Python services (FastAPI server, Celery workers, Admin interface)

**Expected Services:**
- PostgreSQL (port 5433)
- Redis (port 6379) 
- Flower (port 5555)
- Prometheus (port 9090)
- Main API Server (port 8000)
- Admin Interface (port 8001)
- Celery Workers (background processing)

### Stopping Services

```bash
# Stop all Python services (keeps Docker running)
./stop.sh

# Or press Ctrl+C in the terminal running ./run.sh
```

### Alternative: Manual Startup (Old Method)

If you prefer the old 4-terminal setup, you can still use it:

<details>
<summary>Click to expand manual startup instructions</summary>

#### Terminal 1: Infrastructure Services

```bash
# Start PostgreSQL, Redis, Flower monitoring, and Prometheus
docker-compose up -d

# Verify all services are running
docker-compose ps
```

#### Terminal 2: FastAPI Server

```bash
cd tubesensei

# Start the main API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 3: Celery Workers

```bash
cd tubesensei

# Start workers for processing tasks
celery -A app.celery_app worker --loglevel=info --concurrency=4
```

#### Terminal 4: Admin Web Server (Optional)

```bash
cd tubesensei

# Start admin web interface
uvicorn app.main_enhanced:app --host 0.0.0.0 --port 8001 --reload
```

</details>

### 🎉 Verify System Health

Test all components:

```bash
# Basic API health
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/health/database

# Redis connectivity  
curl http://localhost:8000/health/redis

# Worker status
curl http://localhost:8000/health/workers
```

All should return `{"status": "healthy"}`.

---

## 📺 Adding YouTube Channels

### Method 1: API Endpoint (Recommended)

Add channels using the REST API:

```bash
# Add a popular tech channel
curl -X POST "http://localhost:8000/channels/" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_channel_id": "https://www.youtube.com/@MrBeast",
    "processing_config": {
      "extract_ideas": true,
      "languages": ["en", "en-US"]
    }
  }'
```

### Method 2: Admin Web Interface

If you've set up authentication:

1. **Navigate to Admin Panel**: http://localhost:8001/admin/channels
2. **Click "Add Channel"**
3. **Enter Channel URL**: Paste the YouTube channel URL
4. **Configure Settings**: Set processing preferences
5. **Submit**: System will start processing automatically

### Supported URL Formats

TubeSensei accepts various YouTube URL formats:

```bash
# Handle-based URLs (most common)
https://www.youtube.com/@username
https://www.youtube.com/@MrBeast

# Channel ID URLs
https://www.youtube.com/channel/UCChannelId123

# Custom URLs
https://www.youtube.com/c/CustomChannelName

# User URLs (legacy)
https://www.youtube.com/user/OldUsername
```

### Multiple Channels Example

```bash
# Add several channels for comprehensive analysis
channels=(
  "https://www.youtube.com/@MrBeast"
  "https://www.youtube.com/@veritasium" 
  "https://www.youtube.com/@3blue1brown"
  "https://www.youtube.com/@TechCrunch"
)

for channel in "${channels[@]}"; do
  echo "Adding channel: $channel"
  curl -X POST "http://localhost:8000/channels/" \
    -H "Content-Type: application/json" \
    -d "{\"youtube_channel_id\": \"$channel\"}"
  sleep 5  # Avoid rate limiting
done
```

---

## 🤖 Getting Ideas - The AI Pipeline

### How Ideas Are Extracted

1. **Channel Added** → Videos discovered automatically
2. **Transcript Extraction** → Clean transcripts extracted from videos  
3. **AI Processing** → LLM analyzes transcripts for business ideas
4. **Quality Scoring** → Ideas ranked by confidence and relevance
5. **Manual Review** → Review and categorize ideas through admin interface

### Monitoring the Pipeline

#### Option 1: Flower Dashboard (Real-time)
- **URL**: http://localhost:5555
- **Login**: admin / admin
- **Features**: Live task monitoring, queue status, worker health

#### Option 2: API Status Checks

```bash
# Check processing job status
curl http://localhost:8000/jobs/status

# List recent processing jobs
curl "http://localhost:8000/jobs/?limit=10"

# View all channels and their status
curl http://localhost:8000/channels/
```

### Viewing Extracted Ideas

#### Via API Endpoints

```bash
# List all extracted ideas
curl http://localhost:8000/api/v1/ideas/

# Filter ideas by confidence score
curl "http://localhost:8000/api/v1/ideas/?min_confidence=0.8"

# Filter by category
curl "http://localhost:8000/api/v1/ideas/?category=SaaS"

# Search ideas by keyword
curl "http://localhost:8000/api/v1/ideas/?search=automation"

# Get specific idea details
curl http://localhost:8000/api/v1/ideas/{idea_id}
```

#### Via Admin Web Interface

1. **Navigate to Ideas**: http://localhost:8001/admin/ideas
2. **Filter & Sort**: Use filters for status, confidence, category
3. **Review Ideas**: Click on ideas to see full context
4. **Bulk Actions**: Select multiple ideas for bulk operations

### Understanding Idea Data Structure

Each extracted idea includes:

```json
{
  "id": "uuid",
  "title": "AI-Powered Video Summarization Tool",
  "description": "Create a tool that automatically generates summaries...",
  "category": "SaaS",
  "status": "extracted",
  "priority": "high",
  "confidence_score": 0.85,
  "complexity_score": 7,
  "market_size_estimate": "Large",
  "target_audience": "Content creators, Marketers",
  "implementation_time_estimate": "3-6 months",
  "technologies": ["AI", "NLP", "Web Development"],
  "monetization_strategies": ["Subscription", "Usage-based"],
  "competitive_advantage": "First-mover advantage in video AI",
  "potential_challenges": ["Technical complexity", "Data privacy"],
  "video": {
    "title": "Source Video Title",
    "url": "https://youtube.com/watch?v=...",
    "channel": "Channel Name"
  }
}
```

---

## 📊 Managing and Exporting Ideas

### Review Workflow

Ideas go through these statuses:
- **`extracted`** - Newly discovered by AI
- **`reviewed`** - Manually reviewed by user
- **`selected`** - Marked for implementation
- **`rejected`** - Not viable or relevant
- **`in_progress`** - Currently being developed
- **`implemented`** - Successfully executed

### Bulk Operations

```bash
# Mark multiple ideas as reviewed
curl -X PATCH "http://localhost:8000/api/v1/ideas/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "idea_ids": ["id1", "id2", "id3"],
    "action": "review",
    "user_id": "your_user_id"
  }'

# Bulk categorize ideas
curl -X PATCH "http://localhost:8000/api/v1/ideas/bulk" \
  -d '{
    "idea_ids": ["id1", "id2"],
    "action": "update_category",
    "category": "Mobile Apps"
  }'
```

### Export Ideas

#### JSON Export
```bash
# Export selected ideas to JSON
curl -X POST "http://localhost:8000/api/v1/ideas/export" \
  -H "Content-Type: application/json" \
  -d '{
    "idea_ids": ["id1", "id2", "id3"],
    "format": "json"
  }' > ideas.json
```

#### CSV Export  
```bash
# Export to CSV for spreadsheet analysis
curl -X POST "http://localhost:8000/api/v1/ideas/export" \
  -d '{
    "idea_ids": ["id1", "id2", "id3"],
    "format": "csv"
  }' > ideas.csv
```

---

## 🔧 Configuration & Optimization

### LLM Provider Configuration

TubeSensei supports multiple LLM providers with automatic fallback:

#### OpenAI Configuration
```bash
OPENAI_API_KEY=sk-your-key-here
# Uses GPT-4 for idea extraction by default
```

#### Anthropic Claude Configuration  
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
# Uses Claude-3.5-Sonnet for analysis
```

#### Google Gemini Configuration
```bash
GOOGLE_API_KEY=your-gemini-key-here
# Uses Gemini-1.5-Pro for processing
```

### Performance Tuning

#### Worker Scaling
```bash
# Scale workers based on workload
celery -A app.celery_app worker --concurrency=8 --loglevel=info

# Run specialized workers for different tasks
celery -A app.celery_app worker -Q discovery --concurrency=2
celery -A app.celery_app worker -Q transcripts --concurrency=4
celery -A app.celery_app worker -Q ideas --concurrency=6
```

#### Processing Configuration

```bash
# Adjust in .env file
MAX_CONCURRENT_JOBS=20
WORKER_CONCURRENCY=8
TRANSCRIPT_BATCH_SIZE=20
LLM_MAX_RETRIES=3
```

### YouTube API Quota Management

**Daily Quota**: 10,000 units

| Operation | Cost | Notes |
|-----------|------|-------|
| Channel Discovery | ~100 units | Per channel |
| Video Metadata | ~1 unit | Per video |
| Transcript Extraction | **0 units** | Uses youtube-transcript-api |
| Idea Extraction | **0 units** | Uses transcript data |

**Optimization Tips**:
- Process 50-100 channels max per day
- Transcript and idea extraction are unlimited
- Monitor quota in Google Cloud Console

---

## 🎯 Practical Use Cases & Examples

### Use Case 1: Competitor Analysis

```bash
# Add competitor channels in your industry
competitors=(
  "https://www.youtube.com/@competitor1"
  "https://www.youtube.com/@competitor2" 
  "https://www.youtube.com/@competitor3"
)

for channel in "${competitors[@]}"; do
  curl -X POST "http://localhost:8000/channels/" \
    -d "{\"youtube_channel_id\": \"$channel\"}"
done

# After processing, analyze their content strategy
curl "http://localhost:8000/api/v1/ideas/?category=Business&min_confidence=0.8"
```

### Use Case 2: Industry Trend Analysis

```bash
# Add channels from specific industry (e.g., AI/Tech)
tech_channels=(
  "https://www.youtube.com/@3blue1brown"
  "https://www.youtube.com/@TwoMinutePapers"
  "https://www.youtube.com/@lexfridman"
  "https://www.youtube.com/@DeepLearningAI"
)

# Process and analyze emerging trends
for channel in "${tech_channels[@]}"; do
  echo "Processing: $channel"
  curl -X POST "http://localhost:8000/channels/" \
    -d "{\"youtube_channel_id\": \"$channel\"}"
done
```

### Use Case 3: Content Creator Insights

```bash
# Add popular creators to understand content strategies
creators=(
  "https://www.youtube.com/@MrBeast"
  "https://www.youtube.com/@veritasium"
  "https://www.youtube.com/@mkbhd"
)

# Focus on creative and business ideas
curl "http://localhost:8000/api/v1/ideas/?category=Content Creation&status=extracted"
```

---

## 📈 Monitoring & Maintenance

### Health Monitoring Script

Create `health_check.sh`:

```bash
#!/bin/bash
echo "=== TubeSensei Health Check ==="

# Check API
echo "API Status:"
curl -s http://localhost:8000/health | jq '.'

# Check Database
echo -e "\nDatabase Status:"
curl -s http://localhost:8000/health/database | jq '.'

# Check Workers  
echo -e "\nWorker Status:"
curl -s http://localhost:8000/health/workers | jq '.'

# Processing Statistics
echo -e "\nProcessing Stats:"
curl -s http://localhost:8000/jobs/status | jq '.'

# Idea Statistics
echo -e "\nIdea Stats:"
curl -s "http://localhost:8000/api/v1/ideas/?limit=1" | jq '.total'
```

### Daily Maintenance

```bash
# Check processing progress
curl http://localhost:8000/jobs/status

# Review extracted ideas
curl "http://localhost:8000/api/v1/ideas/?status=extracted&limit=10"

# Clean up old jobs (if needed)
curl -X DELETE "http://localhost:8000/jobs/cleanup?days=30"
```

### Backup Your Data

```bash
# Backup PostgreSQL database
pg_dump -h localhost -p 5433 -U tubesensei tubesensei > backup_$(date +%Y%m%d).sql

# Export all ideas to JSON backup
curl "http://localhost:8000/api/v1/ideas/?limit=10000" > ideas_backup_$(date +%Y%m%d).json
```

---

## 🚨 Troubleshooting

### Common Issues

#### Issue: Workers Not Processing
**Symptoms**: Tasks stuck in pending state
**Solution**:
```bash
# Check worker status
celery -A app.celery_app status

# Restart workers
pkill -f "celery worker"
celery -A app.celery_app worker --loglevel=info
```

#### Issue: No Ideas Being Extracted
**Symptoms**: Transcripts processed but no ideas generated
**Check**:
1. LLM API keys are valid: `curl http://localhost:8000/health/llm`
2. Transcript quality: `curl "http://localhost:8000/videos/?has_transcript=true"`
3. Worker logs: Check for LLM errors in worker output

#### Issue: Low Quality Ideas
**Solutions**:
1. Increase confidence threshold in `.env`: `TRANSCRIPT_QUALITY_THRESHOLD=0.8`
2. Use higher-quality LLM: Switch to GPT-4 or Claude-3.5-Sonnet
3. Filter by confidence: Only review ideas with score > 0.7

#### Issue: YouTube API Quota Exceeded
**Symptoms**: 403 errors in logs, channels not discovered
**Solutions**:
```bash
# Check quota usage in Google Cloud Console
# Reduce processing rate
# Wait for daily quota reset
```

### Performance Issues

#### Slow Idea Extraction
```bash
# Scale idea extraction workers
celery -A app.celery_app worker -Q ideas --concurrency=8

# Check LLM response times
curl http://localhost:8000/metrics | grep llm_response_time
```

#### High Memory Usage
```bash
# Monitor system resources
htop

# Workers auto-restart after 1000 tasks
# Or restart manually:
celery -A app.celery_app control pool_restart
```

---

## 📊 Advanced Analytics

### Database Queries for Insights

```sql
-- Connect to database
PGPASSWORD=tubesensei_dev psql -h localhost -p 5433 -U tubesensei -d tubesensei

-- Top idea categories
SELECT category, COUNT(*) as count, AVG(confidence_score) as avg_confidence
FROM ideas 
WHERE confidence_score > 0.7
GROUP BY category 
ORDER BY count DESC;

-- Most productive channels (by ideas generated)
SELECT c.name, COUNT(i.id) as idea_count, AVG(i.confidence_score) as avg_score
FROM channels c
JOIN videos v ON c.id = v.channel_id  
JOIN ideas i ON v.id = i.video_id
GROUP BY c.name
ORDER BY idea_count DESC
LIMIT 10;

-- Processing success rates
SELECT 
  status,
  COUNT(*) as count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM processing_jobs
GROUP BY status;

-- High-value ideas by complexity vs confidence
SELECT title, category, confidence_score, complexity_score,
  ROUND(confidence_score * (10 - complexity_score) / 10, 2) as value_score
FROM ideas
WHERE confidence_score > 0.8
ORDER BY value_score DESC
LIMIT 20;
```

---

## 🚀 Quick Start Checklist

**Setup Phase**:
- [ ] Install Python 3.11+, Docker, Docker Compose
- [ ] Get YouTube API key from Google Cloud Console  
- [ ] Get at least one LLM API key (OpenAI recommended)
- [ ] Clone repository and install dependencies
- [ ] Create and configure `.env` file
- [ ] Run database migrations: `alembic upgrade head`

**Launch Phase**:
- [ ] Start everything with one command: `./run.sh`
- [ ] Verify health: `curl http://localhost:8000/health`
- [ ] Alternative: Use the old 4-terminal setup if preferred

**Processing Phase**:
- [ ] Add your first channel via API
- [ ] Monitor processing in Flower dashboard (http://localhost:5555)
- [ ] Check ideas extraction: `curl http://localhost:8000/api/v1/ideas/`
- [ ] Review ideas in admin interface (http://localhost:8001/admin/ideas)

**Analysis Phase**:
- [ ] Filter high-confidence ideas (> 0.8 score)
- [ ] Categorize and review promising ideas
- [ ] Export selected ideas for further analysis
- [ ] Set up monitoring and backup procedures

---

## 🎯 Next Steps

**Start Small**: Begin with 2-3 channels in your area of interest
**Scale Gradually**: Add more channels as you understand the pipeline  
**Review Regularly**: Set aside time daily to review extracted ideas
**Export & Analyze**: Use the exported data for business planning
**Monitor Performance**: Keep an eye on processing rates and quality

**🎉 You're Ready to Transform YouTube Content into Business Opportunities!**

---

## 📞 Support Resources

- **API Documentation**: http://localhost:8000/docs
- **Admin Interface**: http://localhost:8001/admin  
- **Flower Monitoring**: http://localhost:5555
- **Prometheus Metrics**: http://localhost:9090
- **Project Documentation**: `docs/` directory

---

*TubeSensei - Turn any YouTube channel into a business idea goldmine with AI*