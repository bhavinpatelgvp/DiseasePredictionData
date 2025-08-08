import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'hb', 'totalrbc', 'pcv', 'mcv', 'mch', 'mchc', 'rdw', 'wbc',
            'neutrophi', 'lymph', 'eosin', 'mono', 'plt', 'rbcps', 'wbcps',
            'pltps', 'paracountp', 'mpv', 'mcvrbc'
        ]
        self.diseases = ['Healthy', 'Anemia', 'Leukemia', 'Thrombocytopenia', 'Infection']
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a new one with dummy data."""
        model_path = 'disease_model.joblib'
        scaler_path = 'scaler.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model with synthetic data for demonstration."""
        np.random.seed(42)
        
        # Generate synthetic training data
        n_samples = 1000
        X = np.random.randn(n_samples, len(self.feature_names))
        
        # Create realistic ranges for blood parameters
        feature_ranges = {
            'hb': (10, 18),      # Hemoglobin
            'totalrbc': (3.5, 6.0),  # RBC count
            'pcv': (30, 50),     # Packed cell volume
            'mcv': (70, 100),    # Mean corpuscular volume
            'mch': (25, 35),     # Mean corpuscular hemoglobin
            'mchc': (30, 36),    # Mean corpuscular hemoglobin concentration
            'rdw': (10, 16),     # Red cell distribution width
            'wbc': (4, 12),      # White blood cell count
            'neutrophi': (40, 75),   # Neutrophils
            'lymph': (20, 45),   # Lymphocytes
            'eosin': (1, 6),     # Eosinophils
            'mono': (2, 10),     # Monocytes
            'plt': (150, 450),   # Platelets
            'rbcps': (0, 5),     # RBC per square
            'wbcps': (0, 10),    # WBC per square
            'pltps': (0, 20),    # Platelets per square
            'paracountp': (0, 5), # Parasite count
            'mpv': (7, 12),      # Mean platelet volume
            'mcvrbc': (70, 100)  # MCV RBC
        }
        
        # Scale features to realistic ranges
        for i, feature in enumerate(self.feature_names):
            min_val, max_val = feature_ranges[feature]
            X[:, i] = np.random.uniform(min_val, max_val, n_samples)
        
        # Generate labels based on simple rules
        y = self.generate_labels(X)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(self.model, 'disease_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
    
    def generate_labels(self, X: np.ndarray) -> np.ndarray:
        """Generate labels based on simple medical rules."""
        labels = []
        
        for sample in X:
            hb, totalrbc, pcv, mcv, mch, mchc, rdw, wbc, neutrophi, lymph, eosin, mono, plt = sample[:13]
            
            # Simple rule-based classification
            if hb < 12 or totalrbc < 4.0 or pcv < 35:
                labels.append(1)  # Anemia
            elif wbc > 10 or neutrophi > 70:
                labels.append(4)  # Infection
            elif plt < 150:
                labels.append(3)  # Thrombocytopenia
            elif wbc > 15 or lymph > 50:
                labels.append(2)  # Leukemia
            else:
                labels.append(0)  # Healthy
        
        return np.array(labels)
    
    def predict(self, patient_data: Dict) -> Tuple[str, float]:
        """Predict disease based on patient data."""
        if self.model is None:
            return "Model not available", 0.0
        
        # Extract features
        features = []
        for feature_name in self.feature_names:
            value = patient_data.get(feature_name, 0)
            if value is None:
                value = 0
            features.append(float(value))
        
        # Scale features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        disease_name = self.diseases[prediction]
        confidence = max(probability)
        
        return disease_name, confidence
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def rule_based_prediction(self, patient_data: Dict) -> Tuple[str, str]:
        """Simple rule-based prediction with explanation."""
        hb = patient_data.get('hb', 0) or 0
        totalrbc = patient_data.get('totalrbc', 0) or 0
        pcv = patient_data.get('pcv', 0) or 0
        wbc = patient_data.get('wbc', 0) or 0
        plt = patient_data.get('plt', 0) or 0
        neutrophi = patient_data.get('neutrophi', 0) or 0
        lymph = patient_data.get('lymph', 0) or 0
        
        # Simple diagnostic rules
        if hb < 12 and (totalrbc < 4.0 or pcv < 35):
            return "Anemia", "Low hemoglobin, RBC count, or PCV indicates anemia"
        elif wbc > 11 and neutrophi > 75:
            return "Bacterial Infection", "Elevated WBC and neutrophils suggest bacterial infection"
        elif plt < 150:
            return "Thrombocytopenia", "Low platelet count indicates thrombocytopenia"
        elif wbc > 15 or lymph > 50:
            return "Possible Leukemia", "Very high WBC or lymphocyte count may indicate leukemia"
        elif wbc < 4:
            return "Leukopenia", "Low WBC count indicates leukopenia"
        else:
            return "Normal", "Blood parameters appear within normal ranges"
