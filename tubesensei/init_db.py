import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from app.database import init_db, close_db, engine
from app.models import *


async def main():
    try:
        print("Initializing database...")
        await init_db()
        print("Database initialized successfully!")
        
        print("\nTables created:")
        async with engine.begin() as conn:
            from sqlalchemy import inspect
            
            def get_tables(connection):
                inspector = inspect(connection)
                return inspector.get_table_names()
            
            tables = await conn.run_sync(get_tables)
            for table in tables:
                print(f"  - {table}")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())