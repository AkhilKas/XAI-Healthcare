from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import base64
import io
import os
import numpy as np
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch
import google.generativeai as genai
import json

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')  # Default to flash if not set

# Validate API key is loaded
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

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

# ---------- Movement Quality ---------- :
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


# ROM / QUALITY / COMPENSATION COMPUTATION :
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

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
llm_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config={
        "temperature": float(os.getenv('GEMINI_TEMPERATURE', 0.7)),
        "max_output_tokens": int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', 2048)),
    }
)

# LLM 
def call_llm(summary_inputs):
    """
    Uses Google Gemini to generate clinical insights from motion analysis data.
    
    Args:
        summary_inputs: Dictionary containing:
            - prediction: 0 or 1 (healthy or injured)
            - importance: Array of feature importance values
            - metrics: Dictionary with ROM, movement quality, compensation scores
    
    Returns:
        Dictionary with structured clinical analysis
    """
    
    # Extract data from summary_inputs
    prediction = summary_inputs.get('prediction', 0)
    metrics = summary_inputs.get('metrics', {})
    
    # Prepare the data for the prompt
    rom_data = metrics.get('rom', {})
    mq_data = metrics.get('movement_quality', {})
    injured_region = metrics.get('injured_region', {})
    
    # Build the prompt
    prompt = f"""You are an expert orthopedic physical therapist analyzing motion data from IMU sensors.

**Patient Classification:** {"INJURED" if prediction == 1 else "HEALTHY"}
**Confidence:** {summary_inputs.get('probability', 0)*100:.1f}%

**Motion Metrics:**

Range of Motion (ROM) Scores (0-100, higher = better):
- Head: {rom_data.get('head', 0):.1f}
- Left Wrist: {rom_data.get('left', 0):.1f}
- Right Wrist: {rom_data.get('right', 0):.1f}
- Aggregated ROM: {metrics.get('aggregated_rom', 0):.1f}

Movement Quality Scores (0-100, higher = smoother):
- Head: {mq_data.get('head', 0):.1f}
- Left Wrist: {mq_data.get('left', 0):.1f}
- Right Wrist: {mq_data.get('right', 0):.1f}
- Aggregated Quality: {metrics.get('aggregated_mq', 0):.1f}

Compensation Score: {metrics.get('compensation', 0):.1f} (0-100, lower = better)
Overall Score: {metrics.get('aggregated_score', 0):.1f}

Most Affected Regions (by importance):
- Head: {injured_region.get('head', 0):.1f}%
- Left: {injured_region.get('left', 0):.1f}%
- Right: {injured_region.get('right', 0):.1f}%

**Task:** Provide a clinical analysis in the following JSON format:

{{
  "one_sentence_summary": "A single concise sentence summarizing the patient's condition",
  "key_findings": {{
    "Error: [Issue]": "Brief explanation of critical problem",
    "Warning: [Issue]": "Brief explanation of concerning pattern",
    "Success: [Positive]": "Brief explanation of good movement pattern"
  }},
  "counterfactual_analysis": [
    "If [specific change], then [expected outcome]",
    "Alternative scenario and its impact"
  ],
  "recommendations": [
    "Specific exercise or intervention recommendation",
    "Another actionable recommendation"
  ],
  "detailed_analysis": [
    "Detailed observation about ROM patterns",
    "Analysis of movement quality and compensations",
    "Expected improvements with intervention"
  ]
}}

**Guidelines:**
1. Key findings MUST have exactly 3 items: one Error, one Warning, one Success
2. Base severity on the scores: <40=Error, 40-70=Warning, >70=Success
3. Be specific about body regions (head, left wrist, right wrist)
4. Counterfactual analysis should describe realistic improvements
5. Recommendations should be evidence-based and actionable
6. Return ONLY valid JSON, no markdown formatting or extra text

Analyze the data and respond with the JSON:"""

    try:
        # Call Gemini API
        response = llm_model.generate_content(prompt)
        
        # Extract the text response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith('```'):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        
        response_text = response_text.strip()
        
        # Parse JSON response
        llm_output = json.loads(response_text)
        
        # Validate structure
        required_keys = ['one_sentence_summary', 'key_findings', 'counterfactual_analysis', 
                        'recommendations', 'detailed_analysis']
        
        for key in required_keys:
            if key not in llm_output:
                raise ValueError(f"Missing required key: {key}")
        
        # Ensure key_findings has exactly 3 items with proper prefixes
        if not isinstance(llm_output['key_findings'], dict):
            raise ValueError("key_findings must be a dictionary")
        
        findings = llm_output['key_findings']
        has_error = any(k.startswith('Error:') for k in findings.keys())
        has_warning = any(k.startswith('Warning:') for k in findings.keys())
        has_success = any(k.startswith('Success:') for k in findings.keys())
        
        if not (has_error and has_warning and has_success):
            print("Warning: LLM response missing Error/Warning/Success prefixes, using fallback")
            return get_fallback_response(summary_inputs)
        
        return llm_output
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {response_text[:500]}")  # Print first 500 chars for debugging
        return get_fallback_response(summary_inputs)
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return get_fallback_response(summary_inputs)


def get_fallback_response(summary_inputs):
    """
    Fallback response if LLM fails or returns invalid data.
    Generates a basic analysis from the metrics.
    """
    metrics = summary_inputs.get('metrics', {})
    prediction = summary_inputs.get('prediction', 0)
    
    rom_data = metrics.get('rom', {})
    mq_data = metrics.get('movement_quality', {})
    injured_region = metrics.get('injured_region', {})
    
    # Determine most affected region
    max_region = max(injured_region.items(), key=lambda x: x[1])[0] if injured_region else "unknown"
    
    # Generate findings based on scores
    findings = {}
    
    # Error (lowest score)
    rom_scores = [(k, v) for k, v in rom_data.items()]
    if rom_scores:
        lowest_rom = min(rom_scores, key=lambda x: x[1])
        findings[f"Error: Limited {lowest_rom[0]} ROM"] = \
            f"Range of motion in {lowest_rom[0]} region is {lowest_rom[1]:.1f}/100, indicating significant restriction."
    
    # Warning (compensation)
    comp_score = metrics.get('compensation', 0)
    if comp_score > 30:
        findings["Warning: Compensatory movement patterns"] = \
            f"Compensation score of {comp_score:.1f}/100 suggests reliance on alternative movement strategies."
    else:
        findings["Warning: Movement quality concerns"] = \
            f"Movement quality score of {metrics.get('aggregated_mq', 0):.1f}/100 indicates some irregularities."
    
    # Success (highest score)
    if rom_scores:
        highest_rom = max(rom_scores, key=lambda x: x[1])
        findings[f"Success: Preserved {highest_rom[0]} mobility"] = \
            f"{highest_rom[0].capitalize()} region shows good ROM at {highest_rom[1]:.1f}/100."
    
    status = "impaired movement" if prediction == 1 else "functional movement"
    
    return {
        "one_sentence_summary": f"Analysis indicates {status} with {max_region} region showing highest concern.",
        "key_findings": findings,
        "counterfactual_analysis": [
            f"If {max_region} ROM improved by 15%, overall movement quality would likely increase significantly.",
            "Reducing compensatory patterns through targeted exercises could restore more natural movement."
        ],
        "recommendations": [
            f"Focus on range of motion exercises for {max_region} region",
            "Implement movement pattern retraining to reduce compensation",
            "Progressive strengthening of stabilizing muscles"
        ],
        "detailed_analysis": [
            f"Patient demonstrates {metrics.get('aggregated_rom', 0):.1f}/100 overall ROM with primary limitation in {max_region}.",
            f"Movement quality score of {metrics.get('aggregated_mq', 0):.1f}/100 suggests {'smooth' if metrics.get('aggregated_mq', 0) > 70 else 'irregular'} motion patterns.",
            "Systematic rehabilitation focusing on mobility and motor control is recommended."
        ]
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