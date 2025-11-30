from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rnn_inference import load_model, load_user_data, run_inference

app = FastAPI()

# CORS (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class GoRequest(BaseModel):
    patient: str
    task: str

@app.get("/")
def root():
    return {"message": "Backend running!"}

@app.post("/go")
def go(request: GoRequest):
    print("@@@@@@@@@", request.patient, request.task)
    task_mapping = {'Jar opening': 1, 'Key turning': 2, 'Wall cleaning': 3, 'Backwashing': 4, 'Knife slicing': 5, 'Hammering': 6}
    task = task_mapping[request.task]
    sample = load_user_data(request.patient, task)
    signals = sample[:, 1:]
    result = run_inference(signals, task)
    return result

# @app.post("/overview")
# def analyze(request: GoRequest):
#     print("Received:", request)

#     # TODO: call your ML model or load data
#     sample = load_user_data(request.patient, request.task)
#     signals = sample[:, 1:]
#     result = run_inference(signals, request.task)

#     # return sample data for now
#     return {
#         "movement_issues": [
#             {"type": "error", "title": "Scapular rhythm abnormal", "score": 9.1},
#             {"type": "warning", "title": "Trunk compensations", "score": 6.4},
#             {"type": "success", "title": "Core stability maintained", "score": 8.7}
#         ],
#         "rom": {
#             "elevation": 117,
#             "normal": 165
#         },
#         "counterfactual": {
#             "improved_score": 9.3,
#             "changes": ["Reduce dyskinesis by 15°", "Maintain trunk posture"]
#         }
#     }


# @app.post("/analyze")
# def analyze(req: GoRequest):
#     print("Received:", req)

#     # TODO: call your ML model or load data

#     # return sample data for now
#     return {
#         "movement_issues": [
#             {"type": "error", "title": "Scapular rhythm abnormal", "score": 9.1},
#             {"type": "warning", "title": "Trunk compensations", "score": 6.4},
#             {"type": "success", "title": "Core stability maintained", "score": 8.7}
#         ],
#         "rom": {
#             "elevation": 117,
#             "normal": 165
#         },
#         "counterfactual": {
#             "improved_score": 9.3,
#             "changes": ["Reduce dyskinesis by 15°", "Maintain trunk posture"]
#         }
#     }
