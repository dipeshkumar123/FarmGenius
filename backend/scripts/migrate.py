import os
import sys
import psycopg2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.config import settings

def run_migration():
    print("Connecting to Supabase Database...")
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()
        
        with open("supabase/migrations/20260701_create_schemes_table.sql", "r") as f:
            sql = f.read()
            
        print("Executing migration...")
        cur.execute(sql)
        print("Migration applied successfully!")
        
        # Notify postgREST to reload schema
        print("Reloading PostgREST schema cache...")
        cur.execute("NOTIFY pgrst, 'reload schema';")
        print("Schema reloaded!")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    run_migration()
