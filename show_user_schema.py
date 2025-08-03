import sqlite3
import os
import hashlib

def pwd_encode(pwd):
    secure_pwd = hashlib.md5(pwd.encode()).hexdigest()
    return secure_pwd

def test_login(email, password):
    try:
        conn = sqlite3.connect('diet_recommendation.db')
        cur = conn.cursor()
        
        # Check if user exists
        cur.execute("SELECT * FROM user WHERE u_email=?", (email,))
        user_info = cur.fetchall()
        
        if not user_info:
            print(f"✗ No user found with email: {email}")
            return False
        
        print(f"✓ Found {len(user_info)} user(s) with email: {email}")
        
        for row in user_info:
            user_id = row[0]
            username = row[1]
            stored_password = row[2]
            encoded_input = pwd_encode(password)
            
            print(f"  User ID: {user_id}")
            print(f"  Username: {username}")
            print(f"  Stored password: {stored_password}")
            print(f"  Input password encoded: {encoded_input}")
            
            if encoded_input == stored_password:
                print("✓ Password match successful!")
                return True
            else:
                print("✗ Password mismatch")
        
        conn.close()
        return False
        
    except Exception as e:
        print(f"✗ Error testing login: {e}")
        return False

db_files = ['diet_recommendation.db']

for db_path in db_files:
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        continue
    print(f"\n=== Tables in {db_path}: ===")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        print("  (No tables found)")
    for table in tables:
        print(f'\nSchema for table: {table}')
        cur.execute(f"PRAGMA table_info({table});")
        columns = cur.fetchall()
        if not columns:
            print("  (No columns found)")
        for col in columns:
            print(f"  {col[0]}: {col[1]} ({col[2]})")
        # If this is the user table, print all rows
        if table == 'user':
            print(f"\nData in table: {table}")
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            # Print header
            col_names = [col[1] for col in columns]
            print(" | ".join(col_names))
            print("-" * 80)
            for row in rows:
                print(" | ".join(str(item) for item in row))
        # If this is the user_notes table, print all rows
        if table == 'user_notes':
            print(f"\nData in table: {table}")
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            col_names = [col[1] for col in columns]
            print(" | ".join(col_names))
            print("-" * 80)
            for row in rows:
                print(" | ".join(str(item) for item in row))
    conn.close()

# Add login testing section
print("\n" + "="*50)
print("LOGIN TESTING")
print("="*50)

# Check if there are any users in the database
conn = sqlite3.connect('diet_recommendation.db')
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM user")
user_count = cur.fetchone()[0]
print(f"Total users in database: {user_count}")

if user_count > 0:
    # Show available users for testing
    cur.execute("SELECT u_id, u_username, u_email FROM user LIMIT 5")
    users = cur.fetchall()
    print("\nAvailable users for testing:")
    for user in users:
        print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")
    
    # Test login with first user
    if users:
        test_email = users[0][2]  # Use first user's email
        print(f"\nTesting login with email: {test_email}")
        print("Enter password for this user:")
        test_password = input("Password: ")
        test_login(test_email, test_password)
else:
    print("No users found in database. You need to register first.")

conn.close() 