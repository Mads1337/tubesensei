# TubeSensei Production Deployment Guide

## Prerequisites

- Docker >= 24.0
- Docker Compose >= 2.20
- A domain name with DNS pointing to your server (for SSL)
- SSL certificate files (e.g. from Let's Encrypt)

---

## Quick Start

### 1. Clone the repository and enter the project directory

```bash
git clone <repository-url>
cd TubeSensei
```

### 2. Create your environment file

```bash
cp .env.production.example .env
```

Open `.env` in your editor and fill in all required values. At a minimum you must set:

- `DATABASE_URL` and `POSTGRES_PASSWORD`
- `REDIS_PASSWORD`
- `SECURITY_SECRET_KEY` — generate one with `openssl rand -hex 32`
- `YOUTUBE_API_KEY`
- At least one LLM key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- `SECURITY_ALLOWED_ORIGINS` and `SECURITY_ALLOWED_HOSTS`

### 3. Configure SSL certificates

Place your SSL certificate and key on the host, then add a volume mount to the `nginx` service in `docker-compose.prod.yml`:

```yaml
nginx:
  volumes:
    - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    - /etc/letsencrypt/live/yourdomain.com:/etc/nginx/ssl:ro
```

Update `nginx.conf` to set the correct `server_name` if needed.

### 4. Start all services

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 5. Run database migrations

```bash
docker-compose -f docker-compose.prod.yml exec app python tubesensei/init_db.py
```

Or, if the project uses Alembic directly:

```bash
docker-compose -f docker-compose.prod.yml exec app bash -c "cd tubesensei && alembic upgrade head"
```

### 6. Create the initial admin user

```bash
docker-compose -f docker-compose.prod.yml exec app python create_admin_user.py
```

Follow the prompts to set the admin username, email, and password.

---

## Service URLs

After a successful deployment the following services are available:

| Service            | URL                                    | Notes                         |
|--------------------|----------------------------------------|-------------------------------|
| Admin Interface    | `https://yourdomain.com/`              | Main application              |
| API Docs (Swagger) | `https://yourdomain.com/docs`          | Interactive API documentation |
| Health Check       | `https://yourdomain.com/health`        | Returns `{"status": "ok"}`    |
| Metrics            | `https://yourdomain.com/metrics`       | Prometheus metrics endpoint   |
| Flower             | `http://yourserver:5555`               | Celery task monitor           |
| Prometheus         | `http://yourserver:9090`               | Metrics database              |

> Flower and Prometheus are not exposed through nginx by default. Restrict access to them via a firewall or reverse proxy authentication.

---

## Environment Variable Reference

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL async connection string |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password (must match `DATABASE_URL`) |
| `REDIS_PASSWORD` | Yes | Redis password |
| `SECURITY_SECRET_KEY` | Yes | JWT/session signing key (`openssl rand -hex 32`) |
| `YOUTUBE_API_KEY` | Yes | YouTube Data API v3 key |
| `OPENAI_API_KEY` | One LLM required | OpenAI API key |
| `ANTHROPIC_API_KEY` | One LLM required | Anthropic API key |
| `SECURITY_ALLOWED_ORIGINS` | Yes | JSON list, e.g. `["https://yourdomain.com"]` |
| `SECURITY_ALLOWED_HOSTS` | Yes | JSON list, e.g. `["yourdomain.com"]` |
| `WORKERS` | No | Gunicorn worker count (default: `4`) |
| `WORKER_CONCURRENCY` | No | Celery worker concurrency (default: `4`) |
| `FLOWER_BASIC_AUTH` | No | Flower UI auth as `user:password` |
| `ENVIRONMENT` | No | Set to `production` (default in compose file) |

See `.env.production.example` for the complete list with descriptions.

---

## Health Checks

The following endpoints should be monitored in production:

| Endpoint | Expected Response |
|---|---|
| `GET /health` | `200 OK` with `{"status": "ok"}` |
| `GET /metrics` | `200 OK` with Prometheus text output |

Docker Compose health checks are configured on `postgres`, `redis`, and `app`. Other services (`worker`, `beat`) wait for those to be healthy before starting.

To check service health manually:

```bash
docker-compose -f docker-compose.prod.yml ps
```

---

## Updating to a New Version

```bash
# Pull latest code
git pull origin main

# Rebuild images with no cache
docker-compose -f docker-compose.prod.yml build --no-cache

# Restart services with zero-downtime (rolling restart)
docker-compose -f docker-compose.prod.yml up -d

# Run any new database migrations
docker-compose -f docker-compose.prod.yml exec app bash -c "cd tubesensei && alembic upgrade head"
```

---

## Viewing Logs

View logs for all services:

```bash
docker-compose -f docker-compose.prod.yml logs -f
```

View logs for a specific service:

```bash
docker-compose -f docker-compose.prod.yml logs -f app
docker-compose -f docker-compose.prod.yml logs -f worker
docker-compose -f docker-compose.prod.yml logs -f beat
docker-compose -f docker-compose.prod.yml logs -f nginx
```

Tail the last 100 lines:

```bash
docker-compose -f docker-compose.prod.yml logs --tail=100 app
```

---

## Stopping Services

```bash
# Stop all containers (data is preserved in volumes)
docker-compose -f docker-compose.prod.yml down

# Stop and remove all volumes (DESTROYS ALL DATA)
docker-compose -f docker-compose.prod.yml down -v
```

---

## Backup

Back up the PostgreSQL database:

```bash
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U tubesensei tubesensei > backup_$(date +%Y%m%d_%H%M%S).sql
```

Restore from a backup:

```bash
docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U tubesensei tubesensei < backup_YYYYMMDD_HHMMSS.sql
```
