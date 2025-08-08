import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import os

class DatabaseManager:
    def __init__(self, db_path: str = "medical_records.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                hb REAL,
                totalrbc REAL,
                pcv REAL,
                mcv REAL,
                mch REAL,
                mchc REAL,
                rdw REAL,
                wbc REAL,
                neutrophi REAL,
                lymph REAL,
                eosin REAL,
                mono REAL,
                plt REAL,
                rbcps REAL,
                wbcps REAL,
                pltps REAL,
                paracountp REAL,
                mpv REAL,
                mcvrbc REAL,
                blood_group TEXT,
                prediction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_patient(self, patient_data: Dict) -> int:
        """Create a new patient record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        columns = ', '.join(patient_data.keys())
        placeholders = ', '.join(['?' for _ in patient_data])
        values = list(patient_data.values())
        
        cursor.execute(f'''
            INSERT INTO patients ({columns})
            VALUES ({placeholders})
        ''', values)
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return patient_id
    
    def get_all_patients(self) -> pd.DataFrame:
        """Retrieve all patient records."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY created_at DESC", conn)
        conn.close()
        return df
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict]:
        """Retrieve a specific patient by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [description[0] for description in cursor.description]
            patient = dict(zip(columns, row))
        else:
            patient = None
        
        conn.close()
        return patient
    
    def update_patient(self, patient_id: int, patient_data: Dict) -> bool:
        """Update an existing patient record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add updated_at timestamp
        patient_data['updated_at'] = 'CURRENT_TIMESTAMP'
        
        set_clause = ', '.join([f"{key} = ?" for key in patient_data.keys() if key != 'updated_at'])
        set_clause += ', updated_at = CURRENT_TIMESTAMP'
        
        values = [value for key, value in patient_data.items() if key != 'updated_at']
        values.append(patient_id)
        
        cursor.execute(f'''
            UPDATE patients 
            SET {set_clause}
            WHERE id = ?
        ''', values)
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0
    
    def delete_patient(self, patient_id: int) -> bool:
        """Delete a patient record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        rows_affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return rows_affected > 0
    
    def search_patients(self, search_term: str) -> pd.DataFrame:
        """Search patients by name or ID."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM patients 
            WHERE name LIKE ? OR CAST(id AS TEXT) LIKE ?
            ORDER BY created_at DESC
        '''
        
        search_pattern = f"%{search_term}%"
        df = pd.read_sql_query(query, conn, params=[search_pattern, search_pattern])
        
        conn.close()
        return df
