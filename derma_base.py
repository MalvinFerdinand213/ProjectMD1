from pydantic import BaseModel

class DermaInput(BaseModel):
    date: str
    quarter: int
    department: str
    day: str
    team: int
    targeted_productivity: float
    smv: float
    wip: int
    over_time: int
    incentive: float
    idle_time: float
    idle_men: int
    no_of_style_change: int
    no_of_workers: int

class PredictionResult(BaseModel):
    predicted_actual_productivity: float