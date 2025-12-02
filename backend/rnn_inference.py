from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch

from .rnn_train import RNNClassifier, rnn_channel_importance_from_weights
from .variables import chan_name, DOFS, DEVICE
from .helper_functions import lowpass_filter

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
    with open (f"pickled_datasets/{group}_data_task{task}.pkl", "rb") as file_path:
        user_data = pickle.load(file_path)
        user_data = user_data.get('PX0' + str(user_id)) if group == 'patient' else user_data.get('fx0' + str(user_id))
    return user_data

# ---------- ROM ----------
def compute_rom(seg, perc=95):
    """
    Computes ROM based on percentile range to reduce outlier effect.
    """
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    lower = np.percentile(seg, 100 - perc, axis=0)
    upper = np.percentile(seg, perc, axis=0)
    rom = np.linalg.norm(upper - lower)
    return rom

# # ---------- Movement Quality ---------- : V1
# def compute_movement_quality(seg, dt=1/60):
#     """
#     Movement quality based on normalized jerk and variability.
#     """
#     if isinstance(seg, torch.Tensor):
#         seg = seg.detach().cpu().numpy()

#     # Smooth the signal
#     seg = lowpass_filter(seg)

#     vel = np.gradient(seg, dt, axis=0)
#     acc = np.gradient(vel, dt, axis=0)
#     jerk = np.gradient(acc, dt, axis=0)

#     # Normalized jerk metric (higher = smoother)
#     traj_len = np.sum(np.linalg.norm(vel, axis=1)) * dt
#     njm = traj_len**2 / (np.sum(np.linalg.norm(jerk, axis=1)**2) * dt**5 + 1e-6)  # avoid div 0

#     # Variability metric: multidimensional variance
#     variability = 1 / (1 + np.trace(np.cov(seg.T)))  # higher = more consistent

#     # Weighted MQ score
#     mq_score = (0.6 * njm + 0.4 * variability) * 100
#     return np.clip(mq_score, 0, 100)

def compute_movement_quality(seg, dt=1/60):
    """
    Movement quality based on normalized jerk and variability.
    Returns a score from 0-100 where 100 is smoothest.
    """
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()

    # Smooth the signal
    seg = lowpass_filter(seg)

    vel = np.gradient(seg, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    # Trajectory length
    traj_len = np.sum(np.linalg.norm(vel, axis=1)) * dt
    
    # Avoid division by zero for no movement
    if traj_len < 1e-6:
        return 50.0  # Default middle score
    
    # Total jerk
    total_jerk = np.sum(np.linalg.norm(jerk, axis=1)**2) * dt**5
    
    # Normalized jerk metric (lower is smoother)
    njm_raw = total_jerk / (traj_len**2 + 1e-6)
    
    # Convert to 0-100 scale using sigmoid
    # Typical healthy motion has njm_raw between 0-5
    # Use sigmoid to map: low jerk (0-2) → high score (80-100)
    #                     medium jerk (2-5) → medium score (50-80)
    #                     high jerk (>5) → low score (<50)
    njm_score = 100 / (1 + np.exp((njm_raw - 2) * 2))  # Centered at 2, steeper curve
    
    # Variability: lower covariance = more consistent = higher score
    cov_trace = np.trace(np.cov(seg.T))
    # Use inverse exponential to bound the score
    variability_score = 100 * np.exp(-cov_trace / 0.1)  # Adjust 0.1 based on your data scale
    
    # Weighted combination
    mq_score = 0.6 * njm_score + 0.4 * variability_score
    
    return float(np.clip(mq_score, 0, 100))

# ---------- Compensation ----------
def compute_compensation(head, left, right):
    """
    Quantifies compensation as excessive head movement relative to limb movement.
    Also considers asymmetry between left/right limbs.
    """
    if isinstance(head, torch.Tensor):
        head = head.detach().cpu().numpy()
    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()

    wrist_disp = np.linalg.norm(np.ptp(left, axis=0)) + np.linalg.norm(np.ptp(right, axis=0))
    head_disp = np.linalg.norm(np.ptp(head, axis=0))

    compensation_ratio = head_disp / (wrist_disp + 1e-6)
    asymmetry_ratio = np.abs(np.linalg.norm(np.ptp(left, axis=0)) - np.linalg.norm(np.ptp(right, axis=0))) / (wrist_disp + 1e-6)

    # Combine into a score 0-100
    comp_score = np.clip((compensation_ratio + 0.5 * asymmetry_ratio) * 100, 0, 100)
    return comp_score


# # ROM / QUALITY / COMPENSATION COMPUTATION - V1
# def compute_motion_metrics(sample, importance, alpha_rot=0.5, weights={'rom':0.4, 'mq':0.5, 'comp':0.1}):
#     H, L, R = sample[:, 0:3], sample[:, 6:9], sample[:, 12:15]
#     H_rot, L_rot, R_rot = sample[:, 3:6], sample[:, 9:12], sample[:, 15:18]

#     # ---------- ROM ----------
#     rom_head_score  = int((1-alpha_rot) * compute_rom(H) / 0.5 * 100 + alpha_rot * compute_rom(H_rot)/np.radians(180)*100)
#     rom_left_score  = int((1-alpha_rot) * compute_rom(L) / 0.5 * 100 + alpha_rot * compute_rom(L_rot)/np.radians(180)*100)
#     rom_right_score = int((1-alpha_rot) * compute_rom(R) / 0.5 * 100 + alpha_rot * compute_rom(R_rot)/np.radians(180)*100)

#     # Aggregated ROM
#     aggregated_rom = np.mean([rom_head_score, rom_left_score, rom_right_score])
#     aggregated_rom = np.clip(aggregated_rom, 0, 100)

#     # ---------- Movement Quality ----------
#     mq_head = (1-alpha_rot)*compute_movement_quality(H) + alpha_rot*compute_movement_quality(H_rot)
#     mq_left = (1-alpha_rot)*compute_movement_quality(L) + alpha_rot*compute_movement_quality(L_rot)
#     mq_right = (1-alpha_rot)*compute_movement_quality(R) + alpha_rot*compute_movement_quality(R_rot)

#     # Aggregated Movement Quality
#     aggregated_mq = np.mean([mq_head, mq_left, mq_right])
#     aggregated_mq = np.clip(aggregated_mq, 0, 100)

#     # ---------- Compensation ----------
#     comp_score = compute_compensation(H, L, R)

#     # ---------- Aggregated Overall Score ----------
#     aggregated_score = (weights['rom'] * aggregated_rom +
#                         weights['mq'] * aggregated_mq +
#                         weights['comp'] * comp_score)
#     aggregated_score = np.clip(aggregated_score, 0, 100)

#     # ---------- Injury Side Detection ----------
#     region_names = ["head", "left", "right"]
#     region_imp = np.array([
#         importance[0:6].sum(),   # Head channels
#         importance[6:12].sum(),  # Left wrist channels
#         importance[12:18].sum()  # Right wrist channels
#     ])
#     region_severity = (region_imp / region_imp.max()) * 100
#     injured_regions = {name: float(score) for name, score in zip(region_names, region_severity)}

#     return {
#         "rom": {"head": rom_head_score, "left": rom_left_score, "right": rom_right_score},
#         "aggregated_rom": aggregated_rom,
#         "movement_quality": {"head": mq_head, "left": mq_left, "right": mq_right},
#         "aggregated_mq": aggregated_mq,
#         "compensation": comp_score,
#         "aggregated_score": aggregated_score,
#         "injured_region": injured_regions
#     }

def compute_motion_metrics(sample, importance, alpha_rot=0.5, weights={'rom':0.4, 'mq':0.5, 'comp':0.1}):
    H, L, R = sample[:, 0:3], sample[:, 6:9], sample[:, 12:15]
    H_rot, L_rot, R_rot = sample[:, 3:6], sample[:, 9:12], sample[:, 15:18]

    # ---------- ROM ----------
    # Compute raw ROM values
    rom_head_pos = compute_rom(H)
    rom_head_rot = compute_rom(H_rot)
    rom_left_pos = compute_rom(L)
    rom_left_rot = compute_rom(L_rot)
    rom_right_pos = compute_rom(R)
    rom_right_rot = compute_rom(R_rot)
    
    # CRITICAL FIX: Set proper normalization constants based on your actual data
    # For positional data (meters), typical healthy ROM is around 0.3-0.5 meters
    MAX_POS_ROM = 0.5  # Maximum expected positional ROM in meters
    # For rotational data (radians), full range is π radians
    MAX_ROT_ROM = np.pi  # Maximum expected rotational ROM in radians
    
    # Normalize each component separately, then combine
    rom_head_norm = (1-alpha_rot) * min(rom_head_pos / MAX_POS_ROM, 1.0) + \
                    alpha_rot * min(rom_head_rot / MAX_ROT_ROM, 1.0)
    rom_left_norm = (1-alpha_rot) * min(rom_left_pos / MAX_POS_ROM, 1.0) + \
                    alpha_rot * min(rom_left_rot / MAX_ROT_ROM, 1.0)
    rom_right_norm = (1-alpha_rot) * min(rom_right_pos / MAX_POS_ROM, 1.0) + \
                     alpha_rot * min(rom_right_rot / MAX_ROT_ROM, 1.0)
    
    # Convert to 0-100 scale
    rom_head_score = float(np.clip(rom_head_norm * 100, 0, 100))
    rom_left_score = float(np.clip(rom_left_norm * 100, 0, 100))
    rom_right_score = float(np.clip(rom_right_norm * 100, 0, 100))

    # Aggregated ROM
    aggregated_rom = float(np.mean([rom_head_score, rom_left_score, rom_right_score]))

    # ---------- Movement Quality ----------
    mq_head = compute_movement_quality(H) * (1-alpha_rot) + compute_movement_quality(H_rot) * alpha_rot
    mq_left = compute_movement_quality(L) * (1-alpha_rot) + compute_movement_quality(L_rot) * alpha_rot
    mq_right = compute_movement_quality(R) * (1-alpha_rot) + compute_movement_quality(R_rot) * alpha_rot
    
    # Ensure clipping
    mq_head = float(np.clip(mq_head, 0, 100))
    mq_left = float(np.clip(mq_left, 0, 100))
    mq_right = float(np.clip(mq_right, 0, 100))

    # Aggregated Movement Quality
    aggregated_mq = float(np.mean([mq_head, mq_left, mq_right]))

    # ---------- Compensation ----------
    comp_score = float(np.clip(compute_compensation(H, L, R), 0, 100))

    # ---------- Aggregated Overall Score ----------
    aggregated_score = float(np.clip(
        weights['rom'] * aggregated_rom +
        weights['mq'] * aggregated_mq +
        weights['comp'] * comp_score,
        0, 100
    ))

    # ---------- Injury Side Detection ----------
    region_names = ["head", "left", "right"]
    region_imp = np.array([
        importance[0:6].sum(),
        importance[6:12].sum(),
        importance[12:18].sum()
    ])
    # Avoid division by zero
    region_severity = (region_imp / (region_imp.max() + 1e-8)) * 100
    injured_regions = {name: float(score) for name, score in zip(region_names, region_severity)}

    return {
        "rom": {
            "head": rom_head_score,
            "left": rom_left_score,
            "right": rom_right_score
        },
        "aggregated_rom": aggregated_rom,
        "movement_quality": {
            "head": mq_head,
            "left": mq_left,
            "right": mq_right
        },
        "aggregated_mq": aggregated_mq,
        "compensation": comp_score,
        "aggregated_score": aggregated_score,
        "injured_region": injured_regions
    }

# LLM 
def call_llm(summary_inputs):
    """
    Connect your OpenAI / Gemini / Llama model here.
    Currently returns dummy text.
    """
    # https://www.useinvent.com/e/ast_5k8YHb9LcIqNTWBR2MJ0sY
    # https://www.useinvent.com/e/ast_5k8YHb9LcIqNTWBR2MJ0sY
    # key findings - find top 3 always in form of dictionary key:value pairs - add error, warning, success term for each key based on severity
    return {
        "one_sentence_summary": "The movement indicates mild impairment with notable asymmetry.",
        "key_findings": {
            "Error: Asymmetric wrist motion" : "aaaaaaaaaaa",
            "Warning: Compensatory head movement": "bbbbbbbbb",
            "Success: Reduced left wrist elevation": "cccccccccccc"
        },
        "counterfactual_analysis": ["jshbcsdhjbcdshjcbsdhjv"],
        "recommendations": ["sjcbsjhcbdshjehcbejvbdjvdhfbvdjh"],
        "detailed_analysis": ["The user demonstrates limited ROM in the left wrist with elevated compensatory patterns.", \
                              "A healthier motion would involve stabilizing the head while improving controlled wrist elevation.",\
                                  "Expected improvements include smoother trajectory and reduced muscular load."]
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
# app = FastAPI()

# @app.post("/infer")
# def infer(request: InferenceRequest):
#     sample = load_user_data(request.user, request.task)
#     signals = sample[:, 1:]
#     result = run_inference(signals, request.task)
#     return result