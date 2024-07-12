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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        st.error(f"File not found: {file_path}")
        return False
    return True

# Construct the image path dynamically
image_path = os.path.join(os.path.dirname(__file__), "image.png")

# Header Image
if check_file_exists(image_path):
    try:
        st.image(image_path, use_column_width=True)
        logging.info("Successfully loaded header image")
    except Exception as e:
        logging.error(f"Error loading header image: {str(e)}")
        st.error("Unable to load header image. Please refresh the page or try again later.")

# Title and Description
st.title("Employee Churn Prediction")
st.markdown("""
<style>
.big-font {
  font-size:30px !important;
  color: white;
}
.success-box {
  background-color: #cccccc;
  color: #155724;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
  border: 1px solid #C3E6CB;
}
.error-box {
  background-color: #cccccc;
  color: #721C24;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
  border: 1px solid #F5C6CB;
}
.warning-box {
  background-color: #cccccc;
  color: #856404;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
  border: 1px solid #FFEEBA;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Predict the probability of employee churn using a machine learning model.</p>', unsafe_allow_html=True)
st.markdown("""
**Proof of Concept**: This Streamlit web app demonstrates an employee churn model. 
It allows users to input an employee number to predict churn probability and visualize key factors influencing the prediction. 
The model uses the IBM HR Attrition dataset, which includes approximately 1,470 records.
""")

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    try:
        if os.path.exists("local_file.csv"):
            st.markdown('<div class="success-box">Dataset file found.</div>', unsafe_allow_html=True)
            df = pd.read_csv("local_file.csv")
        else:
            logging.error("Dataset file not found")
            st.markdown('<div class="error-box">Dataset file not found.</div>', unsafe_allow_html=True)
            st.stop()
        
        # Fix column types for Arrow compatibility
        for col in df.columns:
            if df[col].dtype == 'object':
                if set(df[col].unique()).issubset({True, False, 'True', 'False'}):
                    df[col] = df[col].map({'True': True, 'False': False}).astype(bool)
                else:
                    df[col] = df[col].astype('category')

        df = pd.get_dummies(df, columns=["Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "Over18", "OverTime"], drop_first=True)
        df.set_index("EmployeeNumber", inplace=True)
        
        # Ensure 'Attrition_Yes' is not in the feature set
        if 'Attrition_Yes' in df.columns:
            df = df.drop(columns=['Attrition_Yes'])
        
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
        if os.path.exists("xgboost_model.json"):
            st.markdown('<div class="success-box">Model file found.</div>', unsafe_allow_html=True)
            model = xgb.Booster()
            model.load_model("xgboost_model.json")
            logging.info("Successfully loaded model")
        else:
            logging.error("Model file not found")
            st.markdown('<div class="error-box">Model file not found. Please ensure "xgboost_model.json" exists.</div>', unsafe_allow_html=True)
            st.stop()
        return model
    except Exception as e:
        logging.error(f"Error in load_model: {str(e)}")
        st.error("An error occurred while loading the model. Please try again later.")
        st.stop()

# Load data and model
df = load_and_preprocess_data()
model = load_model()

st.markdown('<div class="success-box">Dataset preprocessing completed</div>', unsafe_allow_html=True)

# Define non-binary columns
non_binary_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

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
        
        # Explanation
        st.markdown(f"""
        ### Explanation
        The model predicts that there is a **{churn_probability:.2%}** chance that the employee with number **{employee_number}** will leave the company.
        """)

        # Compute SHAP values for the given employee
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)
            shap_values_employee = explainer.shap_values(employee_data)
            logging.info("Successfully computed SHAP values")
        except Exception as e:
            logging.error(f"Error computing SHAP values: {str(e)}")
            st.markdown(f'<div class="error-box">Error computing SHAP values. Please try again later.</div>', unsafe_allow_html=True)
            st.stop()

        # Select only non-binary features for SHAP interpretation
        shap_values_non_binary = shap_values[:, [df.columns.get_loc(c) for c in non_binary_columns]]
        shap_values_employee_non_binary = shap_values_employee[:, [df.columns.get_loc(c) for c in non_binary_columns]]

        # Display SHAP summary plot for the top 5 non-binary features
        st.markdown("### Feature Importance and Impact")
        top_5_features_idx = np.argsort(np.abs(shap_values_non_binary).mean(0))[-5:][::-1]
        top_5_features = df[non_binary_columns].columns[top_5_features_idx]  # Reverse to maintain order

        # Get feature values for the current employee
        feature_values_employee = employee_data[top_5_features].values.flatten()

        # Plot SHAP summary plot for the top 5 features with the current employee highlighted
        shap_values_top_5 = shap_values_non_binary[:, top_5_features_idx]  # Reverse to maintain order
        X_test_top_5 = df[top_5_features]
        
        # Create a custom colormap from blue to red
        colors = ['#1E90FF', '#FF3030']  # Dark blue to bright red
        n_bins = 100  # Number of color gradations
        custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors, N=n_bins)

        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            shap.summary_plot(shap_values_top_5, X_test_top_5, plot_type="dot", color=custom_cmap, show=False)
            
            # Overlay the current employee's SHAP values with corresponding color
            for i, feature in enumerate(top_5_features):
                value = shap_values_employee_non_binary[0, top_5_features_idx[i]]
                feature_value = feature_values_employee[i]
                normalized_value = (feature_value - df[feature].min()) / (df[feature].max() - df[feature].min())
                color = custom_cmap(normalized_value)
                ax.scatter(value, i, color=color, s=100, edgecolor='white', linewidth=1.5, zorder=5)
            
            plt.tight_layout()
            st.pyplot(fig)
            logging.info("Successfully generated and displayed SHAP plot")
        except Exception as e:
            logging.error(f"Error generating SHAP plot: {str(e)}")
            st.markdown(f'<div class="error-box">Error generating SHAP plot. Please try refreshing the page or try again later.</div>', unsafe_allow_html=True)

        # Explanation of the SHAP plot
        st.markdown("""
        ### How to Read This Chart:
        - **Factors**: Listed on the left (e.g., StockOptionLevel, EnvironmentSatisfaction)
        - **Impact**: Dots to the right (positive SHAP values) increase churn risk, dots to the left (negative SHAP values) decrease it
        - **Color**: Blue represents lower feature values, red represents higher feature values
        - **Distribution**: The spread of dots shows how this factor varies across all employees
        - **This Employee**: The larger, circled dot shows where this specific employee stands

        **Key Takeaway**: 
        - Dots to the left of the center line contribute to lower churn risk
        - Dots to the right of the center line contribute to higher churn risk
        - The color (blue to red) shows if this impact comes from a low or high feature value
        - Longer spreads indicate the feature has a more varied impact on predictions across employees
        """)

        # Show the values of the top 5 features for the selected employee
        st.markdown("### Top 5 Feature Values for the Selected Employee")
        feature_values = employee_data[top_5_features].T
        st.dataframe(feature_values)

        # Interpretation of the SHAP plot
        st.markdown("""
        ### Interpretation of SHAP Summary Plot
        - **StockOptionLevel** (Range: 0-3): Higher levels (closer to 3) typically decrease churn likelihood. This represents the employee's level of stock options.
        - **EnvironmentSatisfaction** (Range: 1-4): Higher satisfaction (closer to 4) generally decreases churn likelihood. This measures the employee's satisfaction with their work environment.
        - **JobSatisfaction** (Range: 1-4): Higher satisfaction (closer to 4) tends to decrease churn likelihood. This indicates how satisfied the employee is with their current job.
        - **JobLevel** (Range: 1-5): Higher job levels (closer to 5) often decrease churn likelihood. This represents the employee's level or seniority within the organization.
        - **YearsWithCurrManager** (Range: 0-17): More years with the current manager typically decrease churn likelihood. This shows how long the employee has been working under their current manager.

        Remember, while these factors are significant, each employee's situation is unique. The model provides insights, but should be considered alongside other factors and personal knowledge of the employee's circumstances.
        """)

        # Add a note about the model using more features
        st.markdown("""
        ### Model Background
        The model uses many more features in the background to predict churn, but the top 5 features shown here are chosen for their high interpretability. These features are easy to understand and provide clear insights into the factors affecting employee churn.
        """)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        st.markdown(f'<div class="error-box">An error occurred during prediction. Please try again later.</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
---
*Created by Jens Reich*

**Note**: This tool is a prototype and should be used alongside other HR insights and personal knowledge of employees.
""")