# Disease Prediction Application

A comprehensive medical records management system built with Streamlit and SQLite that allows healthcare professionals to manage patient data and predict diseases based on blood parameters.

## Features

### 🏥 Core Functionality
- **Create**: Add new patient records with comprehensive blood parameters
- **Read**: View all patients with search functionality
- **Update**: Modify existing patient records and re-predict diseases
- **Delete**: Remove patient records with confirmation

### 🤖 Disease Prediction
- **Rule-based Prediction**: Uses medical knowledge-based rules for disease prediction
- **Machine Learning**: Random Forest classifier trained on synthetic medical data
- **Supported Conditions**: Anemia, Leukemia, Thrombocytopenia, Bacterial Infection, Leukopenia

### 📊 Analytics Dashboard
- Patient demographics analysis
- Blood parameter distribution
- Disease prediction statistics
- Correlation analysis
- Export functionality

### 🩸 Blood Parameters Supported
- Hemoglobin (hb)
- Total RBC count (totalrbc)
- Packed Cell Volume (pcv)
- Mean Corpuscular Volume (mcv)
- Mean Corpuscular Hemoglobin (mch)
- Mean Corpuscular Hemoglobin Concentration (mchc)
- Red Cell Distribution Width (rdw)
- White Blood Cell count (wbc)
- Neutrophils percentage (neutrophi)
- Lymphocytes percentage (lymph)
- Eosinophils percentage (eosin)
- Monocytes percentage (mono)
- Platelet count (plt)
- RBC per square (rbcps)
- WBC per square (wbcps)
- Platelets per square (pltps)
- Parasite count (paracountp)
- Mean Platelet Volume (mpv)
- MCV RBC (mcvrbc)
- Blood group

## Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd AnemiaPrediction-CRUD
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## Usage

### Adding a Patient
1. Navigate to "Add Patient" in the sidebar
2. Fill in the patient's basic information (Name, Age, Gender)
3. Enter available blood parameters
4. Click "Add Patient & Predict Disease"
5. View the disease prediction results

### Viewing Patients
1. Go to "View Patients"
2. Use the search functionality to find specific patients
3. Toggle "Show all columns" to see complete patient data
4. Export data to CSV if needed

### Updating Patient Records
1. Select "Update Patient"
2. Choose the patient from the dropdown
3. Modify the required fields
4. Click "Update Patient & Re-predict Disease"

### Analytics
1. Visit the "Analytics" page
2. View various charts and statistics
3. Analyze blood parameter distributions
4. Export analytics data

## File Structure

```
AnemiaPrediction-CRUD/
├── app.py                 # Main Streamlit application
├── database.py           # Database management class
├── disease_predictor.py  # Disease prediction logic
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── medical_records.db   # SQLite database (created automatically)
├── disease_model.joblib # ML model (created automatically)
└── scaler.joblib        # Feature scaler (created automatically)
```

## Database Schema

The SQLite database contains a `patients` table with the following structure:

- `id`: Primary key (auto-increment)
- `name`: Patient name
- `age`: Patient age
- `gender`: Patient gender
- Blood parameters (19 different parameters)
- `blood_group`: ABO blood group
- `prediction`: Disease prediction result
- `created_at`: Record creation timestamp
- `updated_at`: Last update timestamp

## Disease Prediction Logic

### Rule-based Prediction
The application uses medical knowledge-based rules:
- **Anemia**: Low hemoglobin, RBC count, or PCV
- **Bacterial Infection**: Elevated WBC and neutrophils
- **Thrombocytopenia**: Low platelet count
- **Leukemia**: Very high WBC or lymphocyte count
- **Leukopenia**: Low WBC count

### Machine Learning Model
- Uses Random Forest classifier
- Trained on synthetic medical data
- Features: 19 blood parameters
- Provides confidence scores

## Technical Details

- **Frontend**: Streamlit
- **Database**: SQLite
- **ML Framework**: scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations in your jurisdiction.

## Disclaimer

This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.
