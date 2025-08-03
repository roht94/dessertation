import sqlite3
import os

def get_db_path():
    """Get the path to the database file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'diet_recommendation.db')

def list_database_schema():
    """List all tables and their schemas in the database"""
    db_path = get_db_path()
    
    print("🗄️ Database Schema Analysis")
    print("=" * 60)
    print(f"Database: {db_path}")
    print()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"📋 Found {len(tables)} tables:")
        print("-" * 40)
        
        for i, (table_name,) in enumerate(tables, 1):
            print(f"{i}. {table_name}")
        
        print("\n" + "=" * 60)
        
        # Get schema for each table
        for table_name, in tables:
            print(f"\n📊 Table: {table_name}")
            print("-" * 40)
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print("Columns:")
            for col in columns:
                col_id, name, data_type, not_null, default_val, primary_key = col
                constraints = []
                if not_null:
                    constraints.append("NOT NULL")
                if primary_key:
                    constraints.append("PRIMARY KEY")
                if default_val is not None:
                    constraints.append(f"DEFAULT {default_val}")
                
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                print(f"  • {name} ({data_type}){constraint_str}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"Row count: {row_count}")
            
            # Show sample data for tables with data
            if row_count > 0 and row_count <= 10:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                print("Sample data:")
                for row in sample_data:
                    print(f"  {row}")
            elif row_count > 10:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                print("Sample data (first 3 rows):")
                for row in sample_data:
                    print(f"  {row}")
                print(f"  ... and {row_count - 3} more rows")
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ Database schema analysis complete!")
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

def show_table_relationships():
    """Show potential relationships between tables"""
    print("\n🔗 Table Relationships Analysis")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Look for foreign key relationships
        for table_name in tables:
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            
            if foreign_keys:
                print(f"\n📎 {table_name} has foreign keys:")
                for fk in foreign_keys:
                    print(f"  • References {fk[2]}.{fk[4]} -> {fk[3]}")
        
        # Look for common column names that might indicate relationships
        print(f"\n🔍 Potential relationships based on column names:")
        
        # Get all column names from all tables
        all_columns = {}
        for table_name in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            all_columns[table_name] = columns
        
        # Find common column names
        common_columns = {}
        for table1 in tables:
            for table2 in tables:
                if table1 != table2:
                    common = set(all_columns[table1]) & set(all_columns[table2])
                    if common:
                        for col in common:
                            if col not in common_columns:
                                common_columns[col] = []
                            common_columns[col].append((table1, table2))
        
        for col, table_pairs in common_columns.items():
            if len(table_pairs) > 0:
                print(f"  • Column '{col}' appears in:")
                for table1, table2 in table_pairs:
                    print(f"    - {table1} and {table2}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error analyzing relationships: {e}")

if __name__ == "__main__":
    list_database_schema()
    show_table_relationships() 