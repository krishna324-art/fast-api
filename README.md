# Insurance Premium Predictor

Predicts insurance premiums using Random Forest Regression, served via FastAPI.

Built for learning purposes only. Not for real insurance decisions.

---

## Setup
```bash
git clone https://github.com/yourusername/insurance-premium-predictor.git
cd insurance-premium-predictor

python -m venv insurance_env
insurance_env\Scripts\activate      # Windows
source insurance_env/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Generate the model first — run all cells in `insurance_regression.ipynb`,
then move the generated `insurance_rf_model.pkl` into the project folder.
```bash
uvicorn main:app --reload
# test at http://127.0.0.1:8000/docs
```

---

## Input / Output
```json
// POST /predict
{
  "age": 30,
  "bmi": 27.5,
  "children": 1,
  "sex": "male",
  "smoker": "no",
  "region": "southwest"
}

// Response
{ "predicted_expense": 12450.75 }
```

Valid values:
- age: 18 to 64
- bmi: 15 to 55
- children: 0 to 10
- sex: male, female
- smoker: yes, no
- region: northeast, northwest, southeast, southwest

---

## Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| R2 | 0.87 | explains 87% of variance |
| MAE | $2,435 | average prediction error |
| RMSE | $4,456 | sensitive to large errors |

---

## Where It Won't Be Accurate

- Inputs outside training range (age > 64, BMI > 55)
- Extreme profiles like elderly obese smokers
- Real world factors not in dataset such as medical history and policy type
- Dataset has only 1338 rows — real insurers use millions

---

## Stack

Python, Scikit-learn, FastAPI, Pydantic, Pandas, Joblib

---



