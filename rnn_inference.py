from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import base64
import io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

from rnn_train import RNNClassifier, rnn_channel_importance_from_weights
from variables import chan_name, DOFS, DEVICE

app = FastAPI()

class InferenceRequest(BaseModel):
    user: str
    task: int


# MODEL LOADING
def load_model(task):
    path = f"trained_models/rnn_model_task{task}.pt"
    checkpoint = torch.load(path, map_location=DEVICE)

    params = checkpoint["params"]

    model = RNNClassifier(
        input_dim=DOFS,
        rnn_type=params["rnn_type"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        bidirectional=params["bidirectional"],
        dropout_rnn=params["dropout_rnn"],
        dropout_fc=params["dropout_fc"],
        pooling=params["pooling"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, params


# DATA LOADING
def load_user_data(user, task):
    """
    Example: Your real implementation may fetch from database,
    filesystem, S3, etc.
    """
    group, user_id = user.split("_")
    print(group, user_id)  
    with open (f"pickled_datasets/{group}_data_task{task}.pkl", "rb") as file_path:
        user_data = pickle.load(file_path)
        user_data = user_data.get('PX0' + str(user_id)) if group == 'patient' else user_data.get('fx0' + str(user_id))
    return user_data


# ROM / QUALITY / COMPENSATION COMPUTATION
def compute_motion_metrics(sample, importance, alpha_rot = 0.5):
    H = sample[:, 0:3]
    L = sample[:, 6:9]
    R = sample[:, 12:15]

    H_rot = sample[:, 3:6]
    L_rot = sample[:, 9:12]
    R_rot = sample[:, 15:18]

    # ROM

    def rom_pos(seg):
        # 3D peak-to-peak distance
        if isinstance(seg, torch.Tensor):
            seg = seg.detach().cpu().numpy()
        min_vals = seg.min(axis=0)
        max_vals = seg.max(axis=0)
        distance = np.linalg.norm(max_vals - min_vals)
        return distance

    def rom_rot(seg):
        # angular excursion in degrees
        if isinstance(seg, torch.Tensor):
            seg = seg.detach().cpu().numpy()
        min_vals = seg.min(axis=0)
        max_vals = seg.max(axis=0)
        return np.linalg.norm(max_vals - min_vals)
    
    def compute_rom_score(pos_seg, rot_seg, alpha_rot=0.5, max_pos=0.5, max_rot=180):
        # max_pos in meters, max_rot in degrees
        pos_rom = rom_pos(pos_seg) / max_pos
        rot_rom = rom_rot(rot_seg) / max_rot
        score = (1 - alpha_rot) * pos_rom + alpha_rot * rot_rom
        # score = np.clip(score * 100, 0, 100)
        return int(score)

    rom_head_score  = compute_rom_score(H, H_rot)
    rom_left_score  = compute_rom_score(L, L_rot)
    rom_right_score = compute_rom_score(R, R_rot)

    # Movement Quality
    def movement_quality(seg):
        vel = np.diff(seg, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        smoothness = 1 / (1 + np.mean(np.abs(jerk)))
        variability = 1 / (1 + np.var(seg.cpu().numpy()))
        return (smoothness * 0.6 + variability * 0.4) * 100

    mq_head_pos = movement_quality(H)
    mq_left_pos = movement_quality(L)
    mq_right_pos = movement_quality(R)

    mq_head_rot = movement_quality(H_rot)
    mq_left_rot = movement_quality(L_rot)
    mq_right_rot = movement_quality(R_rot)

    # Combined MQ
    mq_head = (1 - alpha_rot) * mq_head_pos + alpha_rot * mq_head_rot
    mq_left = (1 - alpha_rot) * mq_left_pos + alpha_rot * mq_left_rot
    mq_right = (1 - alpha_rot) * mq_right_pos + alpha_rot * mq_right_rot

    # -------- Compensation ----------
    wrist_disp = np.linalg.norm(np.ptp(L, axis=0)) + np.linalg.norm(np.ptp(R, axis=0))
    head_disp = np.linalg.norm(np.ptp(H, axis=0))

    if head_disp > 0.4 * wrist_disp:
        comp_score = min(100, (head_disp / wrist_disp) * 100)
    else:
        comp_score = 0

    # Injury Side Detection
    region_names = ["Head-Shoulder", "Left Wrist", "Right Wrist"]
    region_imp = np.array([
    importance[0:6].sum(),   # Head channels
    importance[6:12].sum(),  # Left wrist channels
    importance[12:18].sum()  # Right wrist channels
    ])

    # Normalize to 0-100 scale
    region_severity = (region_imp / region_imp.max()) * 100  # max region gets 100
    injured_regions = {name: float(score) for name, score in zip(region_names, region_severity)}

    return {
        "rom": {
            "head": rom_head_score,
            "left": rom_left_score,
            "right": rom_right_score
        },
        "movement_quality": {
            "head": mq_head,
            "left": mq_left,
            "right": mq_right
        },
        "compensation": comp_score,
        "injured_region": injured_regions,
    }


# LLM 
def call_llm(summary_inputs):
    """
    Connect your OpenAI / Gemini / Llama model here.
    Currently returns dummy text.
    """
    return {
        "one_sentence_summary": "The movement indicates mild impairment with notable asymmetry.",
        "key_findings": [
            "Asymmetric wrist motion",
            "Compensatory head movement",
            "Reduced left wrist elevation"
        ],
        "detailed_analysis": "The user demonstrates limited ROM in the left wrist with elevated compensatory patterns. A healthier motion would involve stabilizing the head while improving controlled wrist elevation. Expected improvements include smoother trajectory and reduced muscular load."
    }


def generate_plot(data):
    """Plot 3D stick-figure showing head and wrists (hands)"""

    head = data[:, 0:3]
    lh = data[:, 6:9]
    rh = data[:, 12:15]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(head[:,0], head[:,1], head[:,2], c='r', label='Head')
    ax.scatter(lh[:,0], lh[:,1], lh[:,2], c='g', label='Left Wrist')
    ax.scatter(rh[:,0], rh[:,1], rh[:,2], c='b', label='Right Wrist')

    # Draw lines connecting head to hands
    for i in range(len(data)):
        ax.plot([head[i,0], lh[i,0]], [head[i,1], lh[i,1]], [head[i,2], lh[i,2]], c='g', alpha=0.3)
        ax.plot([head[i,0], rh[i,0]], [head[i,1], rh[i,1]], [head[i,2], rh[i,2]], c='b', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Movement: Head and Wrists")
    ax.legend()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# MAIN INFERENCE FUNCTION
def run_inference(sample, task):
    model, params = load_model(task)

    # Ensure batch dimension
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]

    sample_tensor = torch.tensor(sample, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(sample_tensor)
        prob = logits.softmax(dim=-1)[0, 1].item()
        pred = int(prob >= 0.5)

    # Channel importance
    importance = rnn_channel_importance_from_weights(model, kind=params["rnn_type"])
    importance = importance.cpu().numpy()

    top6_idx = np.argsort(importance)[-1:-7:-1]
    feature_imp = {chan_name[idx]: importance[idx] for idx in top6_idx}
    print("Top-6 sensor channels by permutation importance:")
    for k, v in feature_imp.items():
        print(f"{k}: {v:.4f}")

    # Compute biomechanics
    metrics = compute_motion_metrics(sample[0], importance)

    # Prepare LLM prompt and call
    llm_output = call_llm({
        "prediction": pred,
        "importance": importance.tolist(),
        "metrics": metrics
    })

    img_base64 = generate_plot(sample[0])

    print(pred, prob, metrics, llm_output)
    print(type(pred), type(prob), type(metrics), type(llm_output), type(img_base64))

    # Add more TODO:
    return JSONResponse({
        "injured": bool(to_json_safe(pred)),
        "probability_injured": to_json_safe(prob),
        "metrics": to_json_safe(metrics),
        "llm_summary": to_json_safe(llm_output),
        "trajectory_3d": to_json_safe(img_base64)
    })


# FASTAPI SERVER
app = FastAPI()

@app.post("/infer")
def infer(request: InferenceRequest):
    sample = load_user_data(request.user, request.task)
    signals = sample[:, 1:]
    result = run_inference(signals, request.task)
    return result