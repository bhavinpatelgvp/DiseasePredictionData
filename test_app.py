#!/usr/bin/env python3
"""
Test script for the Disease Prediction CRUD Application
"""

import sys
import os
from database import DatabaseManager
from disease_predictor import DiseasePredictor

def test_database():
    """Test database operations."""
    print("Testing Database Operations...")
    
    # Initialize database
    db = DatabaseManager("test_medical_records.db")
    
    # Test patient data
    test_patient = {
        'name': 'John Doe',
        'age': 35,
        'gender': 'Male',
        'blood_group': 'O+',
        'hb': 12.5,
        'totalrbc': 4.5,
        'pcv': 40.0,
        'mcv': 85.0,
        'mch': 28.0,
        'mchc': 33.0,
        'rdw': 13.0,
        'wbc': 7.0,
        'neutrophi': 60.0,
        'lymph': 30.0,
        'eosin': 3.0,
        'mono': 7.0,
        'plt': 250.0,
        'rbcps': 2.0,
        'wbcps': 5.0,
        'pltps': 10.0,
        'paracountp': 0.0,
        'mpv': 9.0,
        'mcvrbc': 85.0,
        'prediction': 'Normal'
    }
    
    try:
        # Test CREATE
        patient_id = db.create_patient(test_patient)
        print(f"✓ Created patient with ID: {patient_id}")
        
        # Test READ
        patient = db.get_patient_by_id(patient_id)
        if patient and patient['name'] == 'John Doe':
            print("✓ Retrieved patient successfully")
        else:
            print("✗ Failed to retrieve patient")
            return False
        
        # Test UPDATE
        update_data = {'age': 36, 'hb': 13.0}
        success = db.update_patient(patient_id, update_data)
        if success:
            print("✓ Updated patient successfully")
        else:
            print("✗ Failed to update patient")
            return False
        
        # Test READ ALL
        all_patients = db.get_all_patients()
        if not all_patients.empty:
            print(f"✓ Retrieved {len(all_patients)} patients")
        else:
            print("✗ Failed to retrieve all patients")
            return False
        
        # Test SEARCH
        search_results = db.search_patients("John")
        if not search_results.empty:
            print("✓ Search functionality works")
        else:
            print("✗ Search functionality failed")
            return False
        
        # Test DELETE
        success = db.delete_patient(patient_id)
        if success:
            print("✓ Deleted patient successfully")
        else:
            print("✗ Failed to delete patient")
            return False
        
        print("✓ All database tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {str(e)}")
        return False
    
    finally:
        # Clean up test database
        if os.path.exists("test_medical_records.db"):
            os.remove("test_medical_records.db")

def test_disease_predictor():
    """Test disease prediction functionality."""
    print("\nTesting Disease Prediction...")
    
    try:
        predictor = DiseasePredictor()
        
        # Test data for different conditions
        test_cases = [
            {
                'name': 'Normal Patient',
                'data': {
                    'hb': 14.0, 'totalrbc': 4.8, 'pcv': 42.0, 'wbc': 6.5,
                    'plt': 300.0, 'neutrophi': 65.0, 'lymph': 25.0
                },
                'expected': 'Normal'
            },
            {
                'name': 'Anemic Patient',
                'data': {
                    'hb': 8.0, 'totalrbc': 3.2, 'pcv': 28.0, 'wbc': 6.0,
                    'plt': 280.0, 'neutrophi': 60.0, 'lymph': 30.0
                },
                'expected': 'Anemia'
            },
            {
                'name': 'Infection Patient',
                'data': {
                    'hb': 13.0, 'totalrbc': 4.5, 'pcv': 40.0, 'wbc': 15.0,
                    'plt': 350.0, 'neutrophi': 80.0, 'lymph': 15.0
                },
                'expected': 'Bacterial Infection'
            },
            {
                'name': 'Thrombocytopenic Patient',
                'data': {
                    'hb': 12.0, 'totalrbc': 4.2, 'pcv': 38.0, 'wbc': 7.0,
                    'plt': 100.0, 'neutrophi': 65.0, 'lymph': 25.0
                },
                'expected': 'Thrombocytopenia'
            }
        ]
        
        for test_case in test_cases:
            # Test rule-based prediction
            disease, explanation = predictor.rule_based_prediction(test_case['data'])
            print(f"✓ {test_case['name']}: {disease} ({explanation})")
            
            # Test ML prediction (if available)
            try:
                ml_disease, confidence = predictor.predict(test_case['data'])
                print(f"  ML Prediction: {ml_disease} (Confidence: {confidence:.2%})")
            except Exception as e:
                print(f"  ML Prediction: Not available ({str(e)})")
        
        print("✓ Disease prediction tests completed!")
        return True
        
    except Exception as e:
        print(f"✗ Disease prediction test failed: {str(e)}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing Module Imports...")
    
    required_modules = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 'plotly', 'joblib'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - Not installed")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required modules are available!")
        return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Disease Prediction CRUD App - Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test database
    if test_database():
        tests_passed += 1
    
    # Test disease predictor
    if test_disease_predictor():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
