from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("linearmodel.pkl")

class datascience(BaseModel): 
    StudyHours: float

# api end point 
@app.get("/")
def test():
    return {"msg": "Uvicorn is running"}

@app.post("/predict")
def datainsert(ds: datascience):
    marks = model.predict([[ds.StudyHours]])
    return {"predicatedmarks": round(float(marks[0]), 2)}
