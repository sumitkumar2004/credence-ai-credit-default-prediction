# Credence AI — Credit Card Default Prediction

A Flask web interface for your UCI Credit Card Default Prediction project, with a responsive futuristic UI.

## Run on Windows / VS Code

1. Open this folder in VS Code.
2. In the terminal, create and activate a virtual environment (optional but recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Confirm `saved_models` contains the `.pkl` artifacts and `UCI_Credit_Card.csv` is in the project root.
5. Start the website:

   ```powershell
   python app.py
   ```

6. Open `http://127.0.0.1:5000` in your browser. Flask serves the page; Live Server is not needed.

## Notes

- The notebook saves model artifacts with `joblib.dump()`, so `app.py` uses `joblib.load()` rather than `pickle.load()`.
- The Logistic Regression, KNN, SVM and XGBoost models are pipelines that include their preprocessing. CatBoost was trained on the original feature columns. The app passes the same 23 feature columns to the selected model.
- `ID` and `default.payment.next.month` are intentionally excluded from the form inputs.
- The app reads the selected model name and decision threshold from `saved_models/best_model_name.pkl` and `saved_models/best_threshold.pkl` if available.
- This is a project demonstration, not a validated credit decision system. Do not use its output as the sole basis for lending or other consequential decisions.
