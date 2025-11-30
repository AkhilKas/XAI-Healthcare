from .variables import *

import numpy as np
import pandas as pd
import pickle
import torch

base_patient_path = Path(__file__).resolve().parent.parent.parent / "X-DASH-Data-Analysis" / "data" / "px"
base_control_path = Path(__file__).resolve().parent.parent.parent / "X-DASH-Data-Analysis" / "data" / "fx" 

patient_dirs = [p for p in base_patient_path.glob("PX*") if p.is_dir()]
control_dirs = [c for c in base_control_path.glob("fx*") if c.is_dir()]


def load_csv(file_path, index_col=False, header='infer'):
    if file_path.exists():
        return pd.read_csv(file_path, index_col=index_col, header=header)
    return None

def get_task_time_range(df):
    start_time = df.loc[df["Name"] == f"Task {TASK}", "Timestamp"].values[0]
    var = "Chapter 3" if TASK+1 > 6 else f"Task {TASK+1}"
    end_time = df.loc[df["Name"] == var, "Timestamp"].values[0]
    return start_time, end_time

def get_survey_response(df, end_time):
    response = df.loc[df["Timestamp"] == end_time, "Response"].values
    return response[0] if len(response) > 0 else None

def process_individual(patient_dir):

    df1 = load_csv(patient_dir / "Master.csv")
    start_time, end_time = get_task_time_range(df1)

    df2 = load_csv(patient_dir / "PlayerMovement.csv")
    task_movement = df2[(df2["TimeElapsed"] >= start_time) & (df2["TimeElapsed"] <= end_time)]

    if isinstance(task_movement, list) or isinstance(task_movement, np.ndarray) or isinstance(task_movement, pd.DataFrame):
        task_movement = torch.tensor(task_movement.values, dtype=torch.float32, requires_grad=False).to(DEVICE) # .values in case of pandas df 

    return {
        "patient_id": patient_dir.name,
        f"task{TASK}_movement": task_movement,
    }

def process_all():
    patient_data = list(map(lambda p: process_individual(p), patient_dirs))
    return patient_data

if __name__ == "__main__":
    patient_data = process_all()
    sorted_patient_data = sorted(
        patient_data,
        key=lambda d: int(d['patient_id'].lstrip('PX'))
    )

    sample_patient_data = {sorted_patient_data[10]["patient_id"]: sorted_patient_data[10][f"task{TASK}_movement"]}
    print(TASK)

    with open(f"pickled_datasets/sample_patient_data_task{TASK}.pkl", "wb") as f:
        pickle.dump(sample_patient_data, f)


