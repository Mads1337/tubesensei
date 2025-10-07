web: cd tubesensei && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
worker: cd tubesensei && celery -A app.celery_app worker --loglevel=info --concurrency=4
beat: cd tubesensei && celery -A app.celery_app beat --loglevel=info
admin: cd tubesensei && uvicorn app.main_enhanced:app --host 0.0.0.0 --port 8001 --reload