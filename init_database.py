import sqlite3
import os
from datetime import datetime

def get_db_path():
    """Get the absolute path to the database file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'diet_recommendation.db')

def init_database():
    """Initialize the database with required tables for exercise planning"""
    db_path = get_db_path()
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Create workout_plans table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workout_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                u_id INTEGER NOT NULL,
                plan_text TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (u_id) REFERENCES user (u_id)
            )
        """)
        
        # Create exercise_log table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exercise_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                u_id INTEGER NOT NULL,
                exercise_name TEXT NOT NULL,
                sets INTEGER,
                reps INTEGER,
                weight REAL,
                date DATE NOT NULL,
                notes TEXT,
                FOREIGN KEY (u_id) REFERENCES user (u_id)
            )
        """)
        
        # Create user_notes table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                u_id INTEGER NOT NULL,
                note TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (u_id) REFERENCES user (u_id)
            )
        """)
        
        # Create user_feedback table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                u_id INTEGER NOT NULL,
                feedback_text TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (u_id) REFERENCES user (u_id)
            )
        """)
        
        conn.commit()
        print("Database tables initialized successfully!")
        
        # Show existing tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        print("Existing tables:", [table[0] for table in tables])
        
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_database() 