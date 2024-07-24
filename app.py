import os
import sys
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
import logging
from google.cloud import storage
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize GCS client
storage_client = storage.Client()
bucket_name = "ml-webapp-proof-of-concept-bucket"
bucket = storage_client.bucket(bucket_name)

# Function to check if file exists in GCS
def check_file_exists_gcs(blob_name):
    blob = bucket.blob(blob_name)
    return blob.exists()

# Function to upload file to GCS
def upload_to_gcs(file_obj, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_obj)

# Function to download file from GCS
def download_from_gcs(source_blob_name):
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    try:
        if check_file_exists_gcs("local_file.csv"):
            csv_content = download_from_gcs("local_file.csv")
            df = pd.read_csv(io.BytesIO(csv_content))
        else:
            logging.error("Dataset file not found in GCS")
            st.markdown('<div class="error-box">Dataset file not found in GCS.</div>', unsafe_allow_html=True)
            st.stop()
        
        # ... (rest of your preprocessing code)
        
        logging.info("Successfully loaded and preprocessed data")
        return df
    except Exception as e:
        logging.error(f"Error in load_and_preprocess_data: {str(e)}")
        st.error("An error occurred while loading the dataset. Please try again later.")
        st.stop()

# Load the model
@st.cache_resource
def load_model():
    try:
        if check_file_exists_gcs("xgboost_model.json"):
            model_content = download_from_gcs("xgboost_model.json")
            model = xgb.Booster()
            model.load_model(io.BytesIO(model_content))
            logging.info("Successfully loaded model from GCS")
        else:
            logging.error("Model file not found in GCS")
            st.markdown('<div class="error-box">Model file not found in GCS.</div>', unsafe_allow_html=True)
            st.stop()
        return model
    except Exception as e:
        logging.error(f"Error in load_model: {str(e)}")
        st.error("An error occurred while loading the model. Please try again later.")
        st.stop()

# Function to generate and cache SHAP plot
@st.cache_data
def generate_shap_plot(df, model, employee_number):
    try:
        employee_data = df.loc[[employee_number]]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        shap_values_employee = explainer.shap_values(employee_data)

        # Select only non-binary features for SHAP interpretation
        non_binary_columns = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
            'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
            'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
            'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
            'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        shap_values_non_binary = shap_values[:, [df.columns.get_loc(c) for c in non_binary_columns]]
        shap_values_employee_non_binary = shap_values_employee[:, [df.columns.get_loc(c) for c in non_binary_columns]]

        top_5_features_idx = np.argsort(np.abs(shap_values_non_binary).mean(0))[-5:][::-1]
        top_5_features = df[non_binary_columns].columns[top_5_features_idx]

        feature_values_employee = employee_data[top_5_features].values.flatten()
        shap_values_top_5 = shap_values_non_binary[:, top_5_features_idx]
        X_test_top_5 = df[top_5_features]

        colors = ['#1E90FF', '#FF3030']
        n_bins = 100
        custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors, N=n_bins)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_top_5, X_test_top_5, plot_type="dot", color=custom_cmap, show=False)

        for i, feature in enumerate(top_5_features):
            value = shap_values_employee_non_binary[0, top_5_features_idx[i]]
            feature_value = feature_values_employee[i]
            normalized_value = (feature_value - df[feature].min()) / (df[feature].max() - df[feature].min())
            color = custom_cmap(normalized_value)
            ax.scatter(value, i, color=color, s=100, edgecolor='white', linewidth=1.5, zorder=5)

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Upload to GCS
        upload_to_gcs(buf, f'shap_plot_{employee_number}.png')
        
        logging.info(f"Successfully generated and stored SHAP plot for employee {employee_number}")
        return buf
    except Exception as e:
        logging.error(f"Error generating SHAP plot: {str(e)}")
        return None

# Load data and model
df = load_and_preprocess_data()
model = load_model()

# Streamlit interface
st.subheader("Enter Employee Number to Predict Churn Probability")

# Input for employee number
employee_number = st.number_input("Employee Number:", min_value=int(df.index.min()), max_value=int(df.index.max()), step=1)

# Check if employee number exists in the dataset
if employee_number not in df.index:
    st.markdown(f'<div class="warning-box">Employee number {employee_number} not found in the dataset.</div>', unsafe_allow_html=True)
else:
    # Predict the probability of churn for the given employee
    try:
        employee_data = df.loc[[employee_number]]
        dmatrix = xgb.DMatrix(employee_data)
        churn_probability = model.predict(dmatrix)[0]
        
        # Displaying Probability
        st.metric(label="Churn Probability", value=f"{churn_probability:.4f}")
        
        # Generate or retrieve SHAP plot
        shap_plot = generate_shap_plot(df, model, employee_number)
        
        if shap_plot:
            st.image(shap_plot, caption='SHAP Summary Plot', use_column_width=True)
        else:
            # If dynamic generation fails, try to load the default plot
            try:
                default_plot = download_from_gcs('shap_plot_1.png')
                st.image(default_plot, caption='Default SHAP Summary Plot', use_column_width=True)
            except Exception as e:
                logging.error(f"Error loading default SHAP plot: {str(e)}")
                st.error("Unable to display SHAP plot. Please try again later.")
        
        # ... (rest of your code for displaying feature values and interpretation)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        st.markdown(f'<div class="error-box">An error occurred during prediction. Please try again later.</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
---
*Created by Jens Reich*

**Note**: This tool is a prototype and should be used alongside other HR insights and personal knowledge of employees.
""")