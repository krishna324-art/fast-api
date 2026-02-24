from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import joblib  
import pandas as pd

# Load model
model = joblib.load('insurance_rf_model.pkl')

app = FastAPI()

class UserInput(BaseModel):
    age:      Annotated[int,   Field(..., ge=18, le=120, description="Age of the person must be between 18 and 120")]
    bmi:      Annotated[float, Field(..., ge=15, le=100, description="BMI = weight(kg) / height(m)²",range="15-55")]
    children: Annotated[int,   Field(..., ge=0, le=10,  description="Number of children")]
    sex:      Literal['male', 'female']
    smoker:   Literal['yes', 'no']      
    region:   Literal['northeast', 'northwest', 'southeast', 'southwest']

    
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:   return 'young'
        elif self.age < 45: return 'adult'
        elif self.age < 65: return 'middle_age'
        else:               return 'senior'

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        is_smoker = self.smoker == 'yes'
        if is_smoker and self.bmi > 30:   return 'high'
        elif is_smoker and self.bmi > 25: return 'medium'
        elif is_smoker:                   return 'low_smoker'
        else:                             return 'low'

    @computed_field
    @property
    def bmi_category(self) -> str:
        if self.bmi < 18.5:  return 'underweight'
        elif self.bmi < 25:  return 'normal'
        elif self.bmi < 30:  return 'overweight'
        else:                return 'obese'

    @computed_field
    @property
    def age_bmi(self) -> float:
        return self.age * self.bmi

    @computed_field
    @property
    def smoker_bmi(self) -> float:
        return self.bmi if self.smoker == 'yes' else 0.0

    @computed_field
    @property
    def smoker_age(self) -> float:
        return float(self.age) if self.smoker == 'yes' else 0.0


@app.post('/predict')
def predict(data: UserInput):
    # Build dataframe in exact same column order as training
    input_df = pd.DataFrame([{
        'age':            data.age,
        'bmi':            data.bmi,
        'children':       data.children,
        'sex':            data.sex,
        'smoker':         data.smoker,
        'region':         data.region,
        'age_group':      data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'bmi_category':   data.bmi_category,
        'age_bmi':        data.age_bmi,
        'smoker_bmi':     data.smoker_bmi,
        'smoker_age':     data.smoker_age
    }])

    prediction = model.predict(input_df)[0]
    
    return {
        'predicted_expense': round(float(prediction), 2)
    }
