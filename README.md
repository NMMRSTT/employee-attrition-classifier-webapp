# Employee Attrition Classifier Web App

This repository contains a Streamlit web application for predicting employee churn using a machine learning model. The app allows users to input an employee number to predict churn probability and visualize key factors influencing the prediction.

**Main Piece**: The main piece of this project is the Streamlit web app that you will be able to find under [PLACEHOLDER URL].

## Project Structure

```
EMPLOYEE-ATTRITION-CLASSIFIER-WEBAPP/
│
├── .gitignore
├── app.py
├── app.yaml
├── LICENSE
├── local_file.csv
├── README.md
├── requirements.txt
├── Screenshot 2024-07-02 113624.png
├── test_notebook.ipynb
└── xgboost_model.json
```

- **.gitignore**: Specifies files and directories to be ignored by Git.
- **app.py**: The main script for running the Streamlit web application.
- **app.yaml**: Configuration file for deploying the app on platforms like Google App Engine.
- **LICENSE**: License file for the project.
- **local_file.csv**: The dataset used for training and prediction.
- **README.md**: This README file.
- **requirements.txt**: Python dependencies required to run the project.
- **Screenshot 2024-07-02 113624.png**: Header image used in the web app.
- **test_notebook.ipynb**: Jupyter notebook for testing and model training.
- **xgboost_model.json**: Trained XGBoost model saved in JSON format.

## Features

- Predicts the probability of employee churn using a pre-trained XGBoost model.
- Visualizes feature importance and the impact of key factors using SHAP values.
- Provides both SHAP summary plots and traditional feature importance plots.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/EMPLOYEE-ATTRITION-CLASSIFIER-WEBAPP.git
   cd EMPLOYEE-ATTRITION-CLASSIFIER-WEBAPP
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv employee_attrition
   source employee_attrition/bin/activate  # On Windows, use `employee_attrition\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Running the App

To run the Streamlit app locally, use the following command:
```sh
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. Enter the employee number to predict churn probability.
3. View the predicted churn probability and visualize the key factors influencing the prediction.

## Understanding the Visualizations

### Feature Importance and Impact (SHAP Summary Plot)
- **Factors**: Listed on the left (e.g., StockOptionLevel, EnvironmentSatisfaction).
- **Impact**: Dots to the right (red) increase churn risk, dots to the left (blue) decrease it.
- **Color**: Blue means lower values, red means higher values for that factor.
- **This Employee**: The larger, circled dot shows where this specific employee stands.

**Key Takeaway**: Factors with blue dots to the left are helping retain this employee, while red dots to the right suggest areas of concern.

### Traditional Feature Importance
Displays the top 10 most important features used by the model in making predictions.

## Notes

- This tool is a prototype and should be used alongside other HR insights and personal knowledge of employees.
- The model uses many features in the background, but the visualized features are chosen for their high interpretability.

## License

This project is licensed under the terms of the MIT license.

## Author

*Created by Jens Reich*

**Note**: The best model was trained using the `test_notebook.ipynb` file.