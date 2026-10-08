# Credence AI – Credit Card Default Prediction

An end-to-end Machine Learning project that predicts the likelihood of a customer defaulting on their credit card payment. Built using Python, Machine Learning, and Flask, **Credence AI** brings a trained classification model into a simple, interactive web application.

The goal of this project is to explore how machine learning can help identify potential credit risks using customer information, repayment history, billing statements, and previous payment amounts.

## About the Project

Credit card default is an important challenge for banks and financial institutions. Identifying customers who may be at risk of missing future payments can help support better risk assessment and informed decision-making.

In this project, I developed a Flask-based web application that takes customer financial information as input and uses a trained machine learning model to estimate the probability of credit card default.

I also worked on integrating the trained model with a web interface so that users can interact with the prediction system without running the model manually.

## Key Features

- **Machine Learning Prediction:** Estimates the likelihood of credit card payment default.
- **Multiple ML Models:** Includes Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), XGBoost, and CatBoost model artifacts.
- **Model Integration:** Loads the selected trained model and its decision threshold from saved files.
- **Interactive Web Interface:** Provides a user-friendly interface built with Flask, HTML, and CSS.
- **Input Validation:** Checks that the required customer information is entered in a valid numeric format.
- **Data Preprocessing:** Applies preprocessing steps consistent with the training workflow.
- **Prediction Probability:** Displays the estimated probability alongside the predicted risk category.
- **Health Check Endpoint:** Provides a `/health` endpoint to check whether the application and model are ready.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data handling and numerical operations |
| Scikit-learn | Machine learning and model pipelines |
| XGBoost | Gradient boosting classification |
| CatBoost | Gradient boosting classification |
| Joblib | Saving and loading trained models |
| Flask | Backend and web application |
| HTML & CSS | Frontend structure and styling |
| Git & GitHub | Version control and project hosting |

## Dataset

This project uses the **UCI Credit Card Default dataset**, which contains customer demographic information, credit limits, repayment status, bill amounts, payment amounts, and the target variable for default payment.

Dataset source: [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

The application uses 23 input features:

- Credit limit, gender, education, marriage, and age
- Repayment status for the previous six periods
- Bill statement amounts for six periods
- Previous payment amounts for six periods

The `ID` column and `default.payment.next.month` target column are excluded from the prediction inputs.

**Note:** The dataset CSV is expected in the project root directory. It may need to be downloaded separately if it is not included in the repository.

## Project Structure

```text
Credence_AI_Futuristic_Flask_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── saved_models/
│   ├── best_model_name.pkl
│   ├── best_threshold.pkl
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   └── catboost.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## How It Works

1. **Customer Input:** The user enters the required financial and repayment details through the web interface.
2. **Input Validation:** The Flask application checks the submitted values and prepares them in the expected feature format.
3. **Preprocessing:** The application applies the necessary data adjustments before passing the input to the model.
4. **Model Prediction:** The selected trained model estimates the probability of default.
5. **Risk Classification:** The estimated probability is compared with the configured decision threshold.
6. **Result Display:** The application displays the predicted risk category and estimated default probability.

## Installation and Setup

Follow these steps to run the project locally on Windows.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Credence_AI_Futuristic_Flask_Project
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Check the Required Files

Make sure the project contains:

- The `saved_models` directory with the required model artifacts.
- The `templates` directory containing `index.html`.
- The `static` directory containing `style.css`.
- The `UCI_Credit_Card.csv` file in the project root if required by your local workflow.

The Flask application loads the saved model artifacts at startup. The CSV file is not read directly by the current `app.py` prediction route.

### 5. Run the Application

```bash
python app.py
```

### 6. Open the Website

Visit the following address in your browser:

```text
http://127.0.0.1:5000
```

The Flask development server will run the application locally. You do not need VS Code Live Server to serve the Flask website.

### 7. Check Application Health

You can also open:

```text
http://127.0.0.1:5000/health
```

This endpoint returns the model readiness status.

## Machine Learning Approach

The project includes several trained classification models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost
- CatBoost

The application reads the selected model name and decision threshold from saved metadata when those files are available. It then loads the corresponding model and uses its predicted probability to classify the input.

This structure makes it easier to integrate different trained models into the same application without changing the user interface.

## Learning Outcomes

While working on this project, I gained practical experience with:

- Integrating machine learning models into a Flask web application.
- Working with structured financial datasets.
- Preparing input features in the format expected by trained models.
- Loading and managing saved model artifacts using Joblib.
- Connecting a Python backend with HTML and CSS.
- Handling form submissions, validation, and prediction results.
- Organizing a machine learning project for GitHub and portfolio presentation.

## Future Improvements

There are several ways I would like to improve this project further:

- Add model comparison using accuracy, precision, recall, F1-score, and ROC-AUC.
- Improve input validation using appropriate ranges and category restrictions.
- Add visual dashboards for customer risk analysis.
- Include explainability techniques to help users understand the factors influencing predictions.
- Deploy the application to a cloud platform.
- Improve testing, logging, and application security.

## Limitations and Responsible Use

This project is intended for learning, demonstration, and experimentation. The predictions are based on a trained model and historical dataset patterns; they do not guarantee future customer behavior.

The application has not been established as a validated credit decision system. Its predictions should not be used as the sole basis for loan approvals, credit limits, or other consequential financial decisions.

## Author

**Sumit Kumar**

B.Tech in Computer Science and Engineering | Aspiring Data Scientist

Interested in Machine Learning, Data Analytics, Python, and building practical applications that turn data into useful insights.

---

⭐ If you find this project useful, feel free to explore the code and share suggestions for improvement.

