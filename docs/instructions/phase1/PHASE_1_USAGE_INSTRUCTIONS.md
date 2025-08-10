# Phase 1 Usage Instructions
## TubeSensei - YouTube Transcript Processing System

---

## 🎯 What Phase 1 Does

Phase 1 provides a complete YouTube transcript extraction and processing pipeline that:

- **Discovers YouTube Channels** - Add channels and automatically find all their videos
- **Extracts Transcripts** - Gets transcripts from videos (auto-generated or manual)
- **Processes & Stores Data** - Cleans, normalizes, and stores transcripts in PostgreSQL
- **Monitors Processing** - Tracks jobs through Flower dashboard
- **Provides API Access** - Query your data via REST API endpoints

---

## 🚀 Getting Started

### Prerequisites Checklist
- [x] Python 3.11+ installed
- [x] Docker & Docker Compose installed
- [x] YouTube Data API key (get from [Google Cloud Console](https://console.cloud.google.com/apis/credentials))
- [x] All dependencies installed (`pip install -r requirements.txt`)
- [x] Environment configured (`.env` file setup)

### Step 1: Start All Services

You need to run 3 components in separate terminals:

#### Terminal 1: Infrastructure Services
```bash
# Start Redis, Flower, and Prometheus
docker-compose up -d

# Verify services are running
docker-compose ps
```
Expected output: Redis (port 6379), Flower (port 5555), Prometheus (port 9090)

#### Terminal 2: API Server
```bash
cd tubesensei
python -m app.main
```
Expected output: `INFO: Uvicorn running on http://0.0.0.0:8000`

#### Terminal 3: Celery Workers
```bash
cd tubesensei
celery -A app.celery_app worker --loglevel=info
```
Expected output: Worker ready with registered tasks

### Step 2: Verify System Health

Test all components are working:

```bash
# Basic health check
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/health/database

# Redis connectivity  
curl http://localhost:8000/health/redis

# Worker status
curl http://localhost:8000/health/workers
```

All should return `{"status": "healthy", ...}`

---

## 📺 Processing YouTube Channels

### Add Your First Channel

```bash
# Example: Add MrBeast channel
curl -X POST "http://localhost:8000/channels/" \
  -H "Content-Type: application/json" \
  -d '{"channel_url": "https://www.youtube.com/@MrBeast"}'
```

**What happens next:**
1. System parses the channel URL
2. Creates channel record in database
3. Queues discovery task to find all videos
4. Automatically starts extracting transcripts

### Supported Channel URL Formats

```bash
# Handle-based URLs
https://www.youtube.com/@username

# Channel ID URLs
https://www.youtube.com/channel/UCChannelId

# Custom URLs
https://www.youtube.com/c/CustomName

# User URLs
https://www.youtube.com/user/Username
```

### Monitor Processing Progress

#### Option 1: Flower Dashboard (Recommended)
- **URL:** http://localhost:5555
- **Login:** admin:admin
- **Features:** Real-time task monitoring, worker status, queue depths

#### Option 2: API Status Checks
```bash
# Job status summary
curl http://localhost:8000/jobs/status

# List recent jobs
curl http://localhost:8000/jobs/

# List all channels
curl http://localhost:8000/channels/
```

---

## 📊 Viewing Your Data

### Via API Endpoints

#### Channels
```bash
# List all channels
curl http://localhost:8000/channels/

# With pagination
curl "http://localhost:8000/channels/?limit=5&offset=0"
```

#### Videos
```bash
# List all videos
curl http://localhost:8000/videos/

# Videos from specific channel
curl "http://localhost:8000/videos/?channel_id=CHANNEL_UUID"

# With pagination
curl "http://localhost:8000/videos/?limit=20&offset=0"
```

#### Processing Jobs
```bash
# All jobs
curl http://localhost:8000/jobs/

# Filter by status
curl "http://localhost:8000/jobs/?status=completed"

# Job status summary
curl http://localhost:8000/jobs/status
```

### Via Database Access

Connect directly to PostgreSQL:

```bash
# Connect to database
PGPASSWORD=tubesensei_dev psql -h localhost -p 5433 -U tubesensei -d tubesensei
```

#### Useful Queries
```sql
-- View all channels
SELECT id, title, youtube_channel_id, video_count, status 
FROM channels;

-- View videos from a channel
SELECT title, duration, view_count, published_at, has_transcript 
FROM videos 
WHERE channel_id = 'your-channel-uuid'
ORDER BY published_at DESC;

-- View transcripts
SELECT v.title, t.content, t.language, t.quality_score
FROM transcripts t
JOIN videos v ON t.video_id = v.id
WHERE t.quality_score > 0.8;

-- Processing job statistics
SELECT job_type, status, COUNT(*) as count
FROM processing_jobs
GROUP BY job_type, status;
```

---

## 🎯 Practical Use Cases

### Use Case 1: Content Research
```bash
# Add competitor channels
curl -X POST "http://localhost:8000/channels/" \
  -d '{"channel_url": "https://www.youtube.com/@competitor1"}'

curl -X POST "http://localhost:8000/channels/" \
  -d '{"channel_url": "https://www.youtube.com/@competitor2"}'

# After processing, analyze their content
curl http://localhost:8000/videos/ | jq '.videos[] | {title, duration, view_count}'
```

### Use Case 2: Industry Analysis
```bash
# Add multiple channels in your industry
for channel in "@TechChannel1" "@TechChannel2" "@TechChannel3"; do
  curl -X POST "http://localhost:8000/channels/" \
    -d "{\"channel_url\": \"https://www.youtube.com/$channel\"}"
done
```

### Use Case 3: Specific Video Processing
```bash
# Force transcript extraction for specific video
curl -X POST "http://localhost:8000/videos/VIDEO_UUID/transcript"
```

---

## 📈 Understanding the Processing Pipeline

### Phase 1A: Channel Discovery
1. **Input:** YouTube channel URL
2. **Process:** Extract channel metadata, find all videos
3. **Output:** Channel record + video records in database
4. **Duration:** 1-5 minutes depending on channel size

### Phase 1B: Video Metadata
1. **Input:** Video IDs from discovery
2. **Process:** Get video details (title, description, stats)
3. **Output:** Complete video metadata
4. **Duration:** 10-60 seconds per batch

### Phase 1C: Transcript Extraction
1. **Input:** Video IDs with available transcripts
2. **Process:** Extract, clean, and normalize transcripts
3. **Output:** Processed transcript text with quality scores
4. **Duration:** 5-30 seconds per video

### Phase 1D: Quality Control
1. **Input:** Raw transcript data
2. **Process:** Language detection, quality scoring, validation
3. **Output:** High-quality, searchable transcript data
4. **Duration:** Near real-time

---

## 🔧 Configuration & Optimization

### Performance Tuning

#### Worker Scaling
```bash
# Scale workers based on workload
celery -A app.celery_app worker --concurrency=8 --loglevel=info

# Run specialized workers for different queues
celery -A app.celery_app worker -Q discovery --concurrency=2
celery -A app.celery_app worker -Q transcripts --concurrency=6
```

#### Batch Processing
```bash
# Process multiple channels efficiently
curl -X POST "http://localhost:8000/channels/" \
  -d '{"channel_url": "https://www.youtube.com/@channel1"}'

# Wait 10 seconds between requests to avoid rate limits
sleep 10

curl -X POST "http://localhost:8000/channels/" \
  -d '{"channel_url": "https://www.youtube.com/@channel2"}'
```

### YouTube API Quota Management

Your daily quota: **10,000 units**

| Operation | Cost | Notes |
|-----------|------|-------|
| Channel Discovery | ~100 units | Per channel |
| Video Metadata | ~1 unit | Per video |
| Transcript Extraction | **0 units** | Uses youtube-transcript-api |

**Optimization Tips:**
- Process 50-100 channels max per day
- Transcript extraction is unlimited
- Monitor quota in Google Cloud Console

---

## 📊 Monitoring & Maintenance

### Daily Health Checks

```bash
#!/bin/bash
# health_check.sh

echo "=== TubeSensei Health Check ==="

# Check API
echo "API Status:"
curl -s http://localhost:8000/health | jq '.'

# Check Database
echo -e "\nDatabase Status:"
curl -s http://localhost:8000/health/database | jq '.'

# Check Redis
echo -e "\nRedis Status:"
curl -s http://localhost:8000/health/redis | jq '.'

# Check Workers
echo -e "\nWorker Status:"
curl -s http://localhost:8000/health/workers | jq '.'

# Job Statistics
echo -e "\nJob Statistics:"
curl -s http://localhost:8000/jobs/status | jq '.'
```

### Log Monitoring

```bash
# View API logs
tail -f logs/tubesensei.log

# View Docker service logs
docker-compose logs -f

# View worker logs
celery -A app.celery_app events
```

### Backup Your Data

```bash
# Backup PostgreSQL database
pg_dump -h localhost -p 5433 -U tubesensei tubesensei > backup_$(date +%Y%m%d).sql

# Backup Redis data
redis-cli --rdb backup_redis_$(date +%Y%m%d).rdb
```

---

## 🚨 Troubleshooting

### Common Issues

#### Issue: Workers Not Processing Tasks
**Symptoms:** Tasks stuck in "pending" state
**Solution:**
```bash
# Check worker status
celery -A app.celery_app status

# Restart workers
pkill -f "celery worker"
celery -A app.celery_app worker --loglevel=info
```

#### Issue: Database Connection Errors
**Symptoms:** API returns 503 errors
**Solution:**
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Restart PostgreSQL
docker-compose restart postgres
```

#### Issue: Redis Connection Failed
**Symptoms:** Workers can't connect to broker
**Solution:**
```bash
# Check Redis status
docker-compose ps redis
redis-cli ping

# Restart Redis
docker-compose restart redis
```

#### Issue: YouTube API Quota Exceeded
**Symptoms:** 403 errors in logs
**Solution:**
```bash
# Check quota usage in Google Cloud Console
# Wait until quota resets (daily)
# Or reduce processing rate
```

### Performance Issues

#### Slow Processing
```bash
# Check system resources
htop

# Monitor database performance
PGPASSWORD=tubesensei_dev psql -h localhost -p 5433 -U tubesensei -d tubesensei \
  -c "SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;"

# Scale workers
celery -A app.celery_app control pool_grow 2
```

#### High Memory Usage
```bash
# Restart workers periodically
# Workers auto-restart after 1000 tasks (configured)

# Monitor memory
docker stats
```

---

## 📈 Data Export & Analysis

### Export Processed Data

#### Export to JSON
```bash
# Export all channels
curl http://localhost:8000/channels/ | jq '.' > channels.json

# Export all videos
curl http://localhost:8000/videos/ | jq '.' > videos.json
```

#### Export to CSV
```bash
# Export via database query
PGPASSWORD=tubesensei_dev psql -h localhost -p 5433 -U tubesensei -d tubesensei \
  -c "COPY (SELECT title, duration, view_count, published_at FROM videos) TO STDOUT WITH CSV HEADER" > videos.csv
```

### Data Analysis Queries

```sql
-- Top performing videos by views
SELECT v.title, v.view_count, c.title as channel_name
FROM videos v
JOIN channels c ON v.channel_id = c.id
ORDER BY v.view_count DESC
LIMIT 10;

-- Transcript language distribution
SELECT language, COUNT(*) as count
FROM transcripts
GROUP BY language
ORDER BY count DESC;

-- Average video duration by channel
SELECT c.title, AVG(v.duration) as avg_duration_seconds
FROM videos v
JOIN channels c ON v.channel_id = c.id
GROUP BY c.title
ORDER BY avg_duration_seconds DESC;

-- Most active processing days
SELECT DATE(created_at) as date, COUNT(*) as jobs_processed
FROM processing_jobs
WHERE status = 'completed'
GROUP BY DATE(created_at)
ORDER BY jobs_processed DESC;
```

---

## 🎯 Next Steps: Preparing for Phase 2

Once you have processed several channels and have a good dataset:

### Data Quality Check
```bash
# Check transcript coverage
curl http://localhost:8000/videos/ | jq '[.videos[] | select(.has_transcript == true)] | length'

# Check processing success rate
curl http://localhost:8000/jobs/status | jq '.status_counts'
```

### Recommended Dataset Size for Phase 2
- **Minimum:** 100 videos with transcripts
- **Recommended:** 500-1000 videos
- **Optimal:** 2000+ videos across multiple channels

### What You'll Have Ready for Phase 2
- ✅ Clean, normalized transcript data
- ✅ Video metadata and statistics
- ✅ Channel information and relationships
- ✅ Quality scores for filtering
- ✅ Processing history and audit trail

---

## 📞 Support & Resources

### Documentation
- **API Documentation:** http://localhost:8000/docs
- **Phase 1 Completion Report:** `PHASE_1_COMPLETION.md`
- **Technical Specifications:** `tubesensei_spec.md`

### Monitoring Dashboards
- **Flower (Celery):** http://localhost:5555
- **Prometheus (Metrics):** http://localhost:9090

### Useful Commands Reference

```bash
# Start everything
docker-compose up -d && cd tubesensei && python -m app.main &
celery -A app.celery_app worker --loglevel=info

# Stop everything
pkill -f "python -m app.main"
pkill -f "celery worker"
docker-compose down

# Add channel
curl -X POST "http://localhost:8000/channels/" -d '{"channel_url": "CHANNEL_URL"}'

# Check status
curl http://localhost:8000/health && curl http://localhost:8000/jobs/status

# View data
curl http://localhost:8000/channels/ && curl http://localhost:8000/videos/
```

---

**🎉 You're ready to start processing YouTube channels with Phase 1!**

Begin with a few channels you're interested in, monitor the processing through Flower, and build up your transcript database for Phase 2 AI analysis.