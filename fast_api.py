import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd

from derma_base import DermaInput, PredictionResult

app = FastAPI()

MODEL_PATH = 'trained_model.pkl'

# Perbaikan: buka file sebelum load pickle
with open(MODEL_PATH, 'rb') as file:
    regressor = pickle.load(file)

@app.get("/")
async def root():
    return {"message": "Productivity Estimator"}

@app.post('/predict', response_model=PredictionResult)
def predict(data: DermaInput):
    input_data_dict = data.dict()
    date = input_data_dict['date']
    quarter = input_data_dict['quarter']
    department = input_data_dict['department']
    day = input_data_dict['day']
    team = input_data_dict['team']
    targeted_productivity = input_data_dict['targeted_productivity']
    smv = input_data_dict['smv']
    wip = input_data_dict['wip']
    over_time = input_data_dict['over_time']
    incentive = input_data_dict['incentive']
    idle_time = input_data_dict['idle_time']
    idle_men = input_data_dict['idle_men']
    no_of_style_change = input_data_dict['no_of_style_change']
    no_of_workers = input_data_dict['no_of_workers']

    input_df = pd.DataFrame([[ 
        date, quarter, department, day, team, targeted_productivity,
        smv, wip, over_time, incentive, idle_time, idle_men,
        no_of_style_change, no_of_workers
    ]], columns=[
        'date', 'quarter', 'department', 'day', 'team', 'targeted_productivity',
        'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
        'no_of_style_change', 'no_of_workers'
    ])

    prediction = regressor.predict(input_df)
    result = float(prediction[0])

    return PredictionResult(
        predicted_actual_productivity=result
    )

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
