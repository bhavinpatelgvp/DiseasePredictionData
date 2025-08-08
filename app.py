import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import DatabaseManager
from disease_predictor import DiseasePredictor
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Disease Prediction Appliction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and predictor
@st.cache_resource
def init_app():
    db = DatabaseManager()
    predictor = DiseasePredictor()
    return db, None

db, predictor = init_app()

# Sidebar navigation
st.sidebar.title("🏥 Medical Records System")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard", "Add Patient", "View Patients", "Update Patient", "Delete Patient"] # "Analytics"
)

# Helper functions
def get_patient_form_data():
    """Create form for patient data input."""
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Patient Name*", key="name")
        age = st.number_input("Age*", min_value=0, max_value=120, key="age")
        gender = st.selectbox("Gender*", ["Male", "Female", "Other"], key="gender")
        blood_group = st.selectbox("Blood Group", 
                                 ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], 
                                 key="blood_group")
    
    with col2:
        st.subheader("Blood Parameters")
        hb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, step=0.1, key="hb")
        totalrbc = st.number_input("Total RBC (million/μL)", min_value=0.0, max_value=10.0, step=0.1, key="totalrbc")
        pcv = st.number_input("PCV (%)", min_value=0.0, max_value=100.0, step=0.1, key="pcv")
        mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=150.0, step=0.1, key="mcv")
    
    # Additional parameters in expandable sections
    with st.expander("Additional Blood Parameters"):
        col3, col4 = st.columns(2)
        
        with col3:
            mch = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, step=0.1, key="mch")
            mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=50.0, step=0.1, key="mchc")
            rdw = st.number_input("RDW (%)", min_value=0.0, max_value=30.0, step=0.1, key="rdw")
            wbc = st.number_input("WBC (thousand/μL)", min_value=0.0, max_value=50.0, step=0.1, key="wbc")
            neutrophi = st.number_input("Neutrophils (%)", min_value=0.0, max_value=100.0, step=0.1, key="neutrophi")
        
        with col4:
            lymph = st.number_input("Lymphocytes (%)", min_value=0.0, max_value=100.0, step=0.1, key="lymph")
            eosin = st.number_input("Eosinophils (%)", min_value=0.0, max_value=100.0, step=0.1, key="eosin")
            mono = st.number_input("Monocytes (%)", min_value=0.0, max_value=100.0, step=0.1, key="mono")
            plt = st.number_input("Platelets (thousand/μL)", min_value=0.0, max_value=1000.0, step=1.0, key="plt")
            mpv = st.number_input("MPV (fL)", min_value=0.0, max_value=20.0, step=0.1, key="mpv")
    
    with st.expander("Microscopic Parameters"):
        col5, col6 = st.columns(2)
        
        with col5:
            rbcps = st.number_input("RBC per Square", min_value=0.0, max_value=20.0, step=0.1, key="rbcps")
            wbcps = st.number_input("WBC per Square", min_value=0.0, max_value=50.0, step=0.1, key="wbcps")
            pltps = st.number_input("Platelets per Square", min_value=0.0, max_value=100.0, step=0.1, key="pltps")
        
        with col6:
            paracountp = st.number_input("Parasite Count", min_value=0.0, max_value=20.0, step=0.1, key="paracountp")
            mcvrbc = st.number_input("MCV RBC", min_value=0.0, max_value=150.0, step=0.1, key="mcvrbc")
    
    return {
        'name': name,
        'age': age,
        'gender': gender,
        'blood_group': blood_group,
        'hb': hb if hb > 0 else None,
        'totalrbc': totalrbc if totalrbc > 0 else None,
        'pcv': pcv if pcv > 0 else None,
        'mcv': mcv if mcv > 0 else None,
        'mch': mch if mch > 0 else None,
        'mchc': mchc if mchc > 0 else None,
        'rdw': rdw if rdw > 0 else None,
        'wbc': wbc if wbc > 0 else None,
        'neutrophi': neutrophi if neutrophi > 0 else None,
        'lymph': lymph if lymph > 0 else None,
        'eosin': eosin if eosin > 0 else None,
        'mono': mono if mono > 0 else None,
        'plt': plt if plt > 0 else None,
        'rbcps': rbcps if rbcps > 0 else None,
        'wbcps': wbcps if wbcps > 0 else None,
        'pltps': pltps if pltps > 0 else None,
        'paracountp': paracountp if paracountp > 0 else None,
        'mpv': mpv if mpv > 0 else None,
        'mcvrbc': mcvrbc if mcvrbc > 0 else None
    }

def predict_disease(patient_data):
    """Predict disease and return results."""
    if predictor is None:
        # Simple rule-based prediction without the predictor class
        hb = patient_data.get('hb', 0) or 0
        totalrbc = patient_data.get('totalrbc', 0) or 0
        pcv = patient_data.get('pcv', 0) or 0
        wbc = patient_data.get('wbc', 0) or 0
        plt = patient_data.get('plt', 0) or 0
        neutrophi = patient_data.get('neutrophi', 0) or 0
        lymph = patient_data.get('lymph', 0) or 0

        # Simple diagnostic rules
        if hb < 12 and (totalrbc < 4.0 or pcv < 35):
            return "Anemia", "Low hemoglobin, RBC count, or PCV indicates anemia", "N/A", 0.0
        elif wbc > 11 and neutrophi > 75:
            return "Bacterial Infection", "Elevated WBC and neutrophils suggest bacterial infection", "N/A", 0.0
        elif plt < 150:
            return "Thrombocytopenia", "Low platelet count indicates thrombocytopenia", "N/A", 0.0
        elif wbc > 15 or lymph > 50:
            return "Possible Leukemia", "Very high WBC or lymphocyte count may indicate leukemia", "N/A", 0.0
        elif wbc < 4:
            return "Leukopenia", "Low WBC count indicates leukopenia", "N/A", 0.0
        else:
            return "Normal", "Blood parameters appear within normal ranges", "N/A", 0.0
    else:
        # Use rule-based prediction for better interpretability
        disease, explanation = predictor.rule_based_prediction(patient_data)

        # Also get ML prediction if available
        try:
            ml_disease, confidence = predictor.predict(patient_data)
            return disease, explanation, ml_disease, confidence
        except:
            return disease, explanation, "N/A", 0.0

# Main application pages
if page == "Dashboard":
    st.title("🏥 Medical Records Dashboard")
    
    # Get statistics
    patients_df = db.get_all_patients()
    
    if not patients_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(patients_df))
        
        with col2:
            avg_age = patients_df['age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        
        with col3:
            male_count = len(patients_df[patients_df['gender'] == 'Male'])
            st.metric("Male Patients", male_count)
        
        with col4:
            female_count = len(patients_df[patients_df['gender'] == 'Female'])
            st.metric("Female Patients", female_count)
        
        # Recent patients
        st.subheader("Recent Patients")
        recent_patients = patients_df.head(5)[['id', 'name', 'age', 'gender', 'prediction', 'created_at']]
        st.dataframe(recent_patients, use_container_width=True)
        
        # Disease distribution
        if 'prediction' in patients_df.columns and not patients_df['prediction'].isna().all():
            st.subheader("Disease Distribution")
            disease_counts = patients_df['prediction'].value_counts()
            fig = px.pie(values=disease_counts.values, names=disease_counts.index, 
                        title="Distribution of Predicted Diseases")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No patients in the database yet. Add some patients to see the dashboard.")

elif page == "Add Patient":
    st.title("➕ Add New Patient")
    
    with st.form("add_patient_form"):
        patient_data = get_patient_form_data()
        
        submitted = st.form_submit_button("Add Patient & Predict Disease")
        
        if submitted:
            if not patient_data['name'] or not patient_data['age']:
                st.error("Please fill in required fields (Name and Age)")
            else:
                # Predict disease
                disease, explanation, ml_disease, confidence = predict_disease(patient_data)
                patient_data['prediction'] = disease
                
                # Save to database
                try:
                    patient_id = db.create_patient(patient_data)
                    st.success(f"Patient added successfully! ID: {patient_id}")
                    
                    # Display prediction
                    st.subheader("Disease Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Rule-based Prediction:** {disease}")
                        st.write(f"**Explanation:** {explanation}")
                    
                    with col2:
                        if ml_disease != "N/A":
                            st.info(f"**ML Prediction:** {ml_disease}")
                            st.write(f"**Confidence:** {confidence:.2%}")
                        else:
                            st.warning("ML prediction not available")
                    
                except Exception as e:
                    st.error(f"Error adding patient: {str(e)}")

elif page == "View Patients":
    st.title("👥 View All Patients")

    # Search functionality
    search_term = st.text_input("Search by name or ID:", placeholder="Enter patient name or ID")

    if search_term:
        patients_df = db.search_patients(search_term)
        st.subheader(f"Search Results for '{search_term}'")
    else:
        patients_df = db.get_all_patients()
        st.subheader("All Patients")

    if not patients_df.empty:
        # Display options
        col1, col2 = st.columns([3, 1])
        with col2:
            show_all_columns = st.checkbox("Show all columns", value=False)

        if show_all_columns:
            st.dataframe(patients_df, use_container_width=True)
        else:
            # Show essential columns
            essential_cols = ['id', 'name', 'age', 'gender', 'hb', 'wbc', 'plt', 'prediction', 'created_at']
            display_cols = [col for col in essential_cols if col in patients_df.columns]
            st.dataframe(patients_df[display_cols], use_container_width=True)

        # Export functionality
        if st.button("Export to CSV"):
            csv = patients_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"patients_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No patients found.")

elif page == "Update Patient":
    st.title("✏️ Update Patient")

    # Select patient to update
    patients_df = db.get_all_patients()

    if not patients_df.empty:
        patient_options = {f"{row['id']} - {row['name']}": row['id']
                          for _, row in patients_df.iterrows()}

        selected_patient = st.selectbox("Select Patient to Update:",
                                       options=list(patient_options.keys()))

        if selected_patient:
            patient_id = patient_options[selected_patient]
            current_patient = db.get_patient_by_id(patient_id)

            if current_patient:
                st.subheader(f"Updating: {current_patient['name']}")

                with st.form("update_patient_form"):
                    # Pre-populate form with current data
                    for key in st.session_state:
                        if key.startswith(('name', 'age', 'gender', 'blood_group', 'hb', 'totalrbc',
                                          'pcv', 'mcv', 'mch', 'mchc', 'rdw', 'wbc', 'neutrophi',
                                          'lymph', 'eosin', 'mono', 'plt', 'rbcps', 'wbcps',
                                          'pltps', 'paracountp', 'mpv', 'mcvrbc')):
                            del st.session_state[key]

                    # Set current values
                    st.session_state.name = current_patient['name']
                    st.session_state.age = current_patient['age']
                    st.session_state.gender = current_patient['gender']
                    st.session_state.blood_group = current_patient['blood_group'] or "Unknown"

                    # Set blood parameters
                    for param in ['hb', 'totalrbc', 'pcv', 'mcv', 'mch', 'mchc', 'rdw', 'wbc',
                                 'neutrophi', 'lymph', 'eosin', 'mono', 'plt', 'rbcps', 'wbcps',
                                 'pltps', 'paracountp', 'mpv', 'mcvrbc']:
                        if current_patient[param] is not None:
                            st.session_state[param] = float(current_patient[param])

                    updated_data = get_patient_form_data()

                    submitted = st.form_submit_button("Update Patient & Re-predict Disease")

                    if submitted:
                        if not updated_data['name'] or not updated_data['age']:
                            st.error("Please fill in required fields (Name and Age)")
                        else:
                            # Predict disease with updated data
                            disease, explanation, ml_disease, confidence = predict_disease(updated_data)
                            updated_data['prediction'] = disease

                            # Update in database
                            try:
                                success = db.update_patient(patient_id, updated_data)
                                if success:
                                    st.success("Patient updated successfully!")

                                    # Display updated prediction
                                    st.subheader("Updated Disease Prediction")
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.info(f"**Rule-based Prediction:** {disease}")
                                        st.write(f"**Explanation:** {explanation}")

                                    with col2:
                                        if ml_disease != "N/A":
                                            st.info(f"**ML Prediction:** {ml_disease}")
                                            st.write(f"**Confidence:** {confidence:.2%}")
                                        else:
                                            st.warning("ML prediction not available")
                                else:
                                    st.error("Failed to update patient")
                            except Exception as e:
                                st.error(f"Error updating patient: {str(e)}")
    else:
        st.info("No patients available to update.")

elif page == "Delete Patient":
    st.title("🗑️ Delete Patient")

    patients_df = db.get_all_patients()

    if not patients_df.empty:
        patient_options = {f"{row['id']} - {row['name']}": row['id']
                          for _, row in patients_df.iterrows()}

        selected_patient = st.selectbox("Select Patient to Delete:",
                                       options=list(patient_options.keys()))

        if selected_patient:
            patient_id = patient_options[selected_patient]
            current_patient = db.get_patient_by_id(patient_id)

            if current_patient:
                st.subheader("Patient Details")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Name:** {current_patient['name']}")
                    st.write(f"**Age:** {current_patient['age']}")
                    st.write(f"**Gender:** {current_patient['gender']}")

                with col2:
                    st.write(f"**Blood Group:** {current_patient['blood_group']}")
                    st.write(f"**Prediction:** {current_patient['prediction']}")
                    st.write(f"**Created:** {current_patient['created_at']}")

                st.warning("⚠️ This action cannot be undone!")

                if st.button("Delete Patient", type="primary"):
                    try:
                        success = db.delete_patient(patient_id)
                        if success:
                            st.success("Patient deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete patient")
                    except Exception as e:
                        st.error(f"Error deleting patient: {str(e)}")
    else:
        st.info("No patients available to delete.")

elif page == "Analytics":
    st.title("📊 Analytics Dashboard")

    patients_df = db.get_all_patients()

    if not patients_df.empty:
        # Age distribution
        st.subheader("Age Distribution")
        fig_age = px.histogram(patients_df, x='age', nbins=20, title="Patient Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

        # Gender distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gender Distribution")
            gender_counts = patients_df['gender'].value_counts()
            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                               title="Gender Distribution")
            st.plotly_chart(fig_gender, use_container_width=True)

        with col2:
            st.subheader("Blood Group Distribution")
            if 'blood_group' in patients_df.columns:
                blood_group_counts = patients_df['blood_group'].value_counts()
                fig_blood = px.bar(x=blood_group_counts.index, y=blood_group_counts.values,
                                  title="Blood Group Distribution")
                st.plotly_chart(fig_blood, use_container_width=True)

        # Blood parameter analysis
        st.subheader("Blood Parameter Analysis")

        # Select parameters to analyze
        numeric_columns = ['hb', 'totalrbc', 'pcv', 'mcv', 'mch', 'mchc', 'rdw', 'wbc',
                          'neutrophi', 'lymph', 'eosin', 'mono', 'plt', 'mpv']
        available_columns = [col for col in numeric_columns if col in patients_df.columns]

        if available_columns:
            selected_params = st.multiselect("Select parameters to analyze:",
                                           available_columns,
                                           default=available_columns[:4])

            if selected_params:
                # Box plots for selected parameters
                fig_box = go.Figure()

                for param in selected_params:
                    # Remove null values for plotting
                    param_data = patients_df[param].dropna()
                    if not param_data.empty:
                        fig_box.add_trace(go.Box(y=param_data, name=param))

                fig_box.update_layout(title="Blood Parameter Distribution",
                                     yaxis_title="Values")
                st.plotly_chart(fig_box, use_container_width=True)

                # Correlation matrix
                if len(selected_params) > 1:
                    st.subheader("Parameter Correlation Matrix")
                    correlation_data = patients_df[selected_params].corr()

                    fig_corr = px.imshow(correlation_data,
                                        text_auto=True,
                                        aspect="auto",
                                        title="Correlation Matrix of Blood Parameters")
                    st.plotly_chart(fig_corr, use_container_width=True)

        # Disease prediction analysis
        if 'prediction' in patients_df.columns and not patients_df['prediction'].isna().all():
            st.subheader("Disease Prediction Analysis")

            # Disease by age group
            patients_df['age_group'] = pd.cut(patients_df['age'],
                                            bins=[0, 18, 35, 50, 65, 100],
                                            labels=['0-18', '19-35', '36-50', '51-65', '65+'])

            disease_age = pd.crosstab(patients_df['age_group'], patients_df['prediction'])

            fig_disease_age = px.bar(disease_age,
                                   title="Disease Distribution by Age Group",
                                   labels={'value': 'Count', 'index': 'Age Group'})
            st.plotly_chart(fig_disease_age, use_container_width=True)

            # Disease by gender
            disease_gender = pd.crosstab(patients_df['gender'], patients_df['prediction'])

            fig_disease_gender = px.bar(disease_gender,
                                      title="Disease Distribution by Gender",
                                      labels={'value': 'Count', 'index': 'Gender'})
            st.plotly_chart(fig_disease_gender, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")

        if available_columns:
            summary_stats = patients_df[available_columns].describe()
            st.dataframe(summary_stats, use_container_width=True)

        # Export analytics data
        if st.button("Export Analytics Data"):
            analytics_data = {
                'total_patients': len(patients_df),
                'average_age': patients_df['age'].mean(),
                'gender_distribution': patients_df['gender'].value_counts().to_dict(),
                'summary_statistics': summary_stats.to_dict() if available_columns else {}
            }

            import json
            analytics_json = json.dumps(analytics_data, indent=2, default=str)
            st.download_button(
                label="Download Analytics JSON",
                data=analytics_json,
                file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No data available for analytics. Add some patients first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This is a Disease Prediction application built with Streamlit. "
    "It allows you to manage patient records and predict diseases based on blood parameters."
)

st.sidebar.markdown("### Features")
st.sidebar.markdown(
    "- ➕ Add new patients\n"
    "- 👥 View all patients\n"
    "- ✏️ Update patient records\n"
    "- 🗑️ Delete patients\n"
    "- 🔍 Search functionality\n"
    "- 🤖 Disease prediction\n"
    "- 📊 Analytics dashboard\n"
    "- 📁 Export data"
)
