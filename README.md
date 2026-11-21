# Heart Stroke Prediction App

A Streamlit web app that predicts heart disease (heart stroke) risk from 11 clinical inputs,
using a Logistic Regression model (87.1% test accuracy) trained on the Kaggle Heart Failure
Prediction dataset (918 patients).

## Files
- `app.py` – Streamlit app
- `regression_heart.pkl` – trained Logistic Regression model
- `scaler.pkl` – StandardScaler
- `columns.pkl` – feature column order
- `requirements.txt` – Python dependencies

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

*For educational purposes only – not a medical diagnosis tool.*
