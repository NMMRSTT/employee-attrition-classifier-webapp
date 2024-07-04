# Employee Attrition Classifier Web App

This repository contains a Streamlit web application for predicting employee churn using a machine learning model. The app allows users to input an employee number to predict churn probability and visualize key factors influencing the prediction.

**Important**: The primary focus of this project is the Streamlit web app, accessible at [PLACEHOLDER URL]. EDA, feature engineering, and modeling are less important.

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

### Capabilities

- Predicts the probability of employee churn using a pre-trained XGBoost model.
- Visualizes feature importance and the impact of key factors using SHAP values.
- Provides both SHAP summary plots and traditional feature importance plots.

### Data Dictionary

| #  | Column                    | Description                                                   | Dtype  |
|----|---------------------------|---------------------------------------------------------------|--------|
| 0  | Age                       | Employee's age                                                | int64  |
| 1  | Attrition                 | Whether the employee has left the company (Yes/No)            | object |
| 2  | BusinessTravel            | Frequency of business travel (Non-Travel, Travel_Rarely, Travel_Frequently) | object |
| 3  | DailyRate                 | Daily rate of the employee                                    | int64  |
| 4  | Department                | Department the employee belongs to (HR, R&D, Sales)           | object |
| 5  | DistanceFromHome          | Distance from home to workplace                               | int64  |
| 6  | Education                 | Education level (1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor) | int64  |
| 7  | EducationField            | Field of education (Life Sciences, Medical, Marketing, Technical Degree, Other) | object |
| 8  | EmployeeCount             | Count of employees (always 1)                                 | int64  |
| 9  | EmployeeNumber            | Unique employee identifier                                    | int64  |
| 10 | EnvironmentSatisfaction   | Environment satisfaction level (1: Low, 2: Medium, 3: High, 4: Very High) | int64  |
| 11 | Gender                    | Gender of the employee (Male, Female)                         | object |
| 12 | HourlyRate                | Hourly rate of the employee                                   | int64  |
| 13 | JobInvolvement            | Job involvement level (1: Low, 2: Medium, 3: High, 4: Very High) | int64  |
| 14 | JobLevel                  | Job level of the employee                                     | int64  |
| 15 | JobRole                   | Role of the employee (e.g., Sales Executive, Research Scientist) | object |
| 16 | JobSatisfaction           | Job satisfaction level (1: Low, 2: Medium, 3: High, 4: Very High) | int64  |
| 17 | MaritalStatus             | Marital status of the employee (Single, Married, Divorced)    | object |
| 18 | MonthlyIncome             | Monthly income of the employee                                | int64  |
| 19 | MonthlyRate               | Monthly rate of the employee                                  | int64  |
| 20 | NumCompaniesWorked        | Number of companies the employee has worked for               | int64  |
| 21 | Over18                    | Whether the employee is over 18 years old (Yes)               | object |
| 22 | OverTime                  | Whether the employee works overtime (Yes, No)                 | object |
| 23 | PercentSalaryHike         | Percentage increase in salary                                 | int64  |
| 24 | PerformanceRating         | Performance rating (1: Low, 2: Good, 3: Excellent, 4: Outstanding) | int64  |
| 25 | RelationshipSatisfaction  | Relationship satisfaction level (1: Low, 2: Medium, 3: High, 4: Very High) | int64  |
| 26 | StandardHours             | Standard working hours                                        | int64  |
| 27 | StockOptionLevel          | Stock option level (0: None, 1: Low, 2: Medium, 3: High)      | int64  |
| 28 | TotalWorkingYears         | Total number of working years                                 | int64  |
| 29 | TrainingTimesLastYear     | Number of training times last year                            | int64  |
| 30 | WorkLifeBalance           | Work-life balance satisfaction (1: Bad, 2: Good, 3: Better, 4: Best) | int64  |
| 31 | YearsAtCompany            | Number of years at the company                                | int64  |
| 32 | YearsInCurrentRole        | Number of years in the current role                           | int64  |
| 33 | YearsSinceLastPromotion   | Number of years since the last promotion                      | int64  |
| 34 | YearsWithCurrManager      | Number of years with the current manager                      | int64  |

This detailed data dictionary provides a comprehensive overview of the dataset, facilitating better understanding and analysis.

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