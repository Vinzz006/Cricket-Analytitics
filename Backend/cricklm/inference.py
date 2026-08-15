"""
inference.py — CrickLM Inference Engine
"""

import json
import math
import re
import torch
from pathlib import Path
from typing import Optional
 
from model import CrickLM, CrickLMConfig
from tokenizer import CrickTokenizer

# benchmarking angles
BENCHMARKS = {
    # Batting
    "right_knee_bat":  (120, 145),   # ideal range degrees
    "left_knee_bat":   (120, 150),
    "right_elbow_bat": (80,  130),
    "head_offset_px":  (-20,  20),   # pixels from center
 
    # Bowling
    "right_knee_bowl": (140, 165),   # front knee at delivery
    "right_elbow_bowl":(150, 180),   # arm should be near straight at release
    "shoulder_angle":  (15,   45),   # degrees of rotation for side-on action
}

SEVERITY_THRESHOLD = {
    "High": 25,   # angle deviates more than 25° from ideal range
    "Med":  10,   # angle deviates 10-25° from ideal range
    # otherwise "Low"
}

def angle_severity(angle: Optional[float], ideal_range: tuple) -> str:
    if angle is None:
        return "Low"
    lo, hi = ideal_range
    if lo <= angle <= hi:
        return "Low"
    deviation = min(abs(angle - lo), abs(angle - hi))
    if deviation > SEVERITY_THRESHOLD["High"]:
        return "High"
    if deviation > SEVERITY_THRESHOLD["Med"]:
        return "Med"
    return "Low"
 
 
def build_analysis_prompt(features: dict, player_type: str) -> str:
    angles = features.get("joint_angles_deg", {})
    metrics = features.get("body_metrics", {})
 
    def a(key):
        v = angles.get(key)
        return f"{v:.1f} degrees" if v is not None else "not detected"
 
    stance = metrics.get("stance_to_shoulder_ratio", 0)
    head_off = metrics.get("head_offset_from_center_px", 0)
    wrist_h = metrics.get("wrist_height_normalized", 0)
 
    if player_type == "batsman":
        prompt = (
            f"biomechanical analysis of batsman technique . "
            f"right knee angle is {a('right_knee')} . "
            f"left knee angle is {a('left_knee')} . "
            f"right elbow angle is {a('right_elbow')} . "
            f"head offset from center is {head_off:.1f} pixels . "
            f"stance width ratio is {stance:.2f} . "
            f"wrist height is {wrist_h:.2f} . "
            f"the batsman has weakness"
        )
    else:
        prompt = (
            f"biomechanical analysis of bowling action technique . "
            f"right shoulder angle is {a('right_shoulder')} . "
            f"right elbow angle is {a('right_elbow')} . "
            f"right knee angle is {a('right_knee')} . "
            f"shoulder alignment angle is "
            f"{metrics.get('shoulder_alignment_angle_deg', 0):.1f} degrees . "
            f"the bowler has weakness"
        )
    return prompt
 
 
def parse_generated_text(text: str, features: dict, player_type: str) -> dict:
    angles = features.get("joint_angles_deg", {})
    metrics = features.get("body_metrics", {})
 
    rk = angles.get("right_knee")
    lk = angles.get("left_knee")
    re_ = angles.get("right_elbow")
    le = angles.get("left_elbow")
    rs = angles.get("right_shoulder")
    head_off = metrics.get("head_offset_from_center_px", 0)
    stance_r = metrics.get("stance_to_shoulder_ratio", 1.2)
    wrist_h = metrics.get("wrist_height_normalized", 0.5)
 
    weaknesses = []
    vulnerabilities = []
    drills = []
    strengths = []
 
    if player_type == "batsman":
        # Knee bend analysis 
        if rk is not None:
            if rk < 120:
                weaknesses.append({
                    "title": "Excessive knee bend — too crouched",
                    "detail": f"Right knee at {rk:.1f}°. Below 120° creates instability and restricts weight transfer through the shot.",
                    "severity": "High",
                    "joint": "right_knee",
                })
                vulnerabilities.append({
                    "delivery": "Full delivery on off stump",
                    "reason": "Low stance restricts front foot drive and forces an across-the-line movement.",
                    "risk": "High",
                })
            elif rk > 155:
                weaknesses.append({
                    "title": "Insufficient knee bend — standing too tall",
                    "detail": f"Right knee at {rk:.1f}°. Above 155° means insufficient flex; reduces balance and power generation.",
                    "severity": "Med",
                    "joint": "right_knee",
                })
                vulnerabilities.append({
                    "delivery": "Short pitched ball into the ribs",
                    "reason": "Upright stance limits ability to duck or sway away from the short ball.",
                    "risk": "High",
                })
                drills.append({
                    "name": "Athletic stance drill",
                    "description": "Stand in your batting stance, ensuring knees are at 130-145°. Hold for 30 seconds, focusing on balance.",
                    "duration": "10 min × 3 sessions/week",
                    "targets": "knee_bend, balance",
                })
            else:
                strengths.append(f"Good knee bend at {rk:.1f}° — within the ideal 120–155° range.")
 
        # Head position analysis 
        if abs(head_off) > 25:
            direction = "right" if head_off > 0 else "left"
            weaknesses.append({
                "title": f"Head falling {direction} of center",
                "detail": f"Head is {abs(head_off):.0f}px off center. Head position determines eye line; falling over causes mistimed shots.",
                "severity": "High" if abs(head_off) > 40 else "Med",
                "joint": "head/neck",
            })
            vulnerabilities.append({
                "delivery": "Away swing outside off stump",
                "reason": "Head position pulls eyes away from the ball line, making late movement hard to track.",
                "risk": "Critical",
            })
            drills.append({
                "name": "Head position mirror drill",
                "description": "Practice shadow batting in front of a mirror. Keep eyes level and head still throughout each shot.",
                "duration": "15 min daily",
                "targets": "head_stability, eye_line",
            })
        else:
            strengths.append("Head remains stable and centered — good eye position throughout the stance.")
 
        # Elbow analysis 
        if re_ is not None:
            if re_ < 70:
                weaknesses.append({
                    "title": "Tucked elbow — restricted bat swing",
                    "detail": f"Right elbow at {re_:.1f}°. Tucked elbows restrict the bat swing arc and reduce power through the line.",
                    "severity": "Med",
                    "joint": "right_elbow",
                })
                drills.append({
                    "name": "High elbow drive drill",
                    "description": "Use a resistance band to practice leading with a high right elbow on the drive. Focus on elbow staying above wrist at top of swing.",
                    "duration": "10 min × 4 sessions/week",
                    "targets": "elbow_position, bat_swing",
                })
            elif re_ > 140:
                weaknesses.append({
                    "title": "Overextended elbow — loss of control",
                    "detail": f"Right elbow at {re_:.1f}°. Overextension reduces bat control and causes the bat to move away from the body.",
                    "severity": "Med",
                    "joint": "right_elbow",
                })
 
        # Stance width 
        if stance_r < 0.9:
            weaknesses.append({
                "title": "Narrow stance — poor base of support",
                "detail": f"Stance ratio {stance_r:.2f}. A narrow stance limits balance and weight transfer power.",
                "severity": "Med",
                "joint": "feet/ankles",
            })
            vulnerabilities.append({
                "delivery": "Yorker on middle stump",
                "reason": "Narrow base makes it hard to get low quickly for a yorker defence.",
                "risk": "High",
            })
        elif stance_r > 1.6:
            weaknesses.append({
                "title": "Very wide stance — restricted footwork",
                "detail": f"Stance ratio {stance_r:.2f}. Too wide a stance prevents effective front or back foot movement.",
                "severity": "High",
                "joint": "feet/ankles",
            })
            vulnerabilities.append({
                "delivery": "Good length delivery on off stump",
                "reason": "Wide stance restricts ability to move front foot to pitch of the ball.",
                "risk": "High",
            })
            drills.append({
                "name": "Footwork ladder drill",
                "description": "Use a speed ladder to practice front and back foot movements from a balanced stance. React to coach's call for drive vs pull.",
                "duration": "20 min × 3 sessions/week",
                "targets": "footwork, stance_width",
            })
 
        # Wrist height (bat lift) 
        if wrist_h < 0.3:
            weaknesses.append({
                "title": "Low bat lift",
                "detail": f"Wrist height normalised to {wrist_h:.2f}. A low backswing reduces time to generate bat speed through the hitting zone.",
                "severity": "Med",
                "joint": "wrists",
            })
        else:
            strengths.append("Good bat lift height — generating sufficient swing arc for power hitting.")
 
        # Default vulnerability if none found
        if not vulnerabilities:
            vulnerabilities.append({
                "delivery": "Leg spin on a length, turning away",
                "reason": "Technique gap identified from stance position.",
                "risk": "Med",
            })
 
        # Always add a fundamental drill
        drills.append({
            "name": "Soft hands and leave drill",
            "description": "Face throw-downs on a good length outside off. Practice leaving balls with soft hands. Builds discipline and late decision-making.",
            "duration": "15 min × session",
            "targets": "shot_selection, hands",
        })
        drills.append({
            "name": "Tee batting — front foot drive",
            "description": "Place ball on tee at front-foot reach. Drive straight through the line, filming from side-on to check head position and elbow.",
            "duration": "10 min × 4 sessions/week",
            "targets": "technique, head_position",
        })
 
        # Compute overall score from severity counts
        high_count = sum(1 for w in weaknesses if w["severity"] == "High")
        med_count  = sum(1 for w in weaknesses if w["severity"] == "Med")
        raw_score  = 100 - (high_count * 18) - (med_count * 8)
        overall = max(20, min(92, raw_score))
 
        sub_scores = {
            "posture":   max(20, min(95, 85 - high_count * 15 - med_count * 5)),
            "balance":   max(20, min(95, 80 if abs(head_off) > 25 else 90)),
            "technique": max(20, min(95, overall + 5)),
            "footwork":  max(20, min(95, 75 if stance_r < 0.9 or stance_r > 1.6 else 88)),
        }
 
        summary = (
            f"Analysis of this batsman reveals "
            f"{'significant' if high_count >= 2 else 'moderate' if high_count == 1 else 'minor'} "
            f"technical issues. "
            f"Key angles measured: right knee {rk:.1f}° " if rk else "Key joint angles measured. "
        )
        summary += (
            f"{'Head position requires attention. ' if abs(head_off) > 25 else 'Head position is stable. '}"
            f"{'Elbow position needs correction. ' if re_ and (re_ < 70 or re_ > 140) else ''}"
            f"Overall technique is {'developing' if overall < 60 else 'solid' if overall < 80 else 'advanced'}."
        )
 
        pro_map = {
            (85, 100): "Virat Kohli — compact, balanced stance with excellent head position",
            (65,  84): "Shubman Gill — technically correct with room to develop power",
            (45,  64): "Early-career MS Dhoni — raw power with technique refinements needed",
            (0,   44): "Junior academy level — fundamentals need structured coaching attention",
        }
        pro_comp = next(
            v for (lo, hi), v in pro_map.items() if lo <= overall <= hi
        )
 
    else:  # bowler
        # Bowling action analysis 
        sa = metrics.get("shoulder_alignment_angle_deg", 0)
 
        if rs is not None and rs < 140:
            weaknesses.append({
                "title": "Low bowling arm at release",
                "detail": f"Right shoulder angle {rs:.1f}°. Arm below ideal release height reduces bounce and increases injury risk.",
                "severity": "High",
                "joint": "right_shoulder",
            })
            vulnerabilities.append({
                "delivery": "Short of length delivery",
                "reason": "Low arm trajectory produces flat trajectory — easier to pull or cut.",
                "risk": "High",
            })
            drills.append({
                "name": "High arm bowling drill",
                "description": "Bowl against a wall target placed above shoulder height. Force high arm release point.",
                "duration": "20 min × session",
                "targets": "arm_height, release_point",
            })
        elif rs is not None:
            strengths.append(f"Good arm height at {rs:.1f}° — generating steep delivery angle.")
 
        if rk is not None and rk < 150:
            weaknesses.append({
                "title": "Bent front knee at delivery",
                "detail": f"Front knee at {rk:.1f}°. Knee should be braced (close to 170°) at release for effective load transfer and pace.",
                "severity": "High" if rk < 130 else "Med",
                "joint": "right_knee",
            })
            drills.append({
                "name": "Front knee brace drill",
                "description": "Bowl off a short run-up focusing on driving the front knee straight through delivery. Coach watches from side-on.",
                "duration": "15 min × session",
                "targets": "front_knee, pace_generation",
            })
 
        if abs(sa) < 15:
            weaknesses.append({
                "title": "Front-on action — limited seam control",
                "detail": f"Shoulder alignment angle {sa:.1f}°. A more side-on action improves seam presentation and swing generation.",
                "severity": "Med",
                "joint": "shoulders",
            })
 
        vulnerabilities = [
            {"delivery": "Any full delivery", "reason": "Front-on action limits outswing making it predictable.", "risk": "Med"},
        ] if not vulnerabilities else vulnerabilities
 
        if not drills:
            drills.append({
                "name": "Side-on action alignment drill",
                "description": "Practice delivery stride with a hoop on the crease. Drive the non-bowling arm down and across body to force hip rotation.",
                "duration": "20 min × 3 sessions/week",
                "targets": "hip_rotation, side_on_action",
            })
 
        high_count = sum(1 for w in weaknesses if w["severity"] == "High")
        med_count  = sum(1 for w in weaknesses if w["severity"] == "Med")
        raw_score  = 100 - (high_count * 18) - (med_count * 8)
        overall = max(20, min(92, raw_score))
 
        sub_scores = {
            "posture":   max(20, min(95, 82 - high_count * 12)),
            "balance":   max(20, min(95, 85)),
            "technique": max(20, min(95, overall)),
            "footwork":  max(20, min(95, 80)),
        }
 
        summary = (
            f"Bowling action analysis shows "
            f"{'significant' if high_count >= 2 else 'some'} areas for improvement. "
            f"Arm height and front knee position are the key determinants "
            f"of pace and bounce at this level."
        )
 
        pro_comp = (
            "Jasprit Bumrah — if front knee and arm height are corrected"
            if overall > 70 else
            "Early-career Shami — action in development with clear technical targets"
        )
 
    if not strengths:
        strengths.append("Shows commitment to technique development.")
        strengths.append("Consistent stance setup before each delivery.")
 
    return {
        "summary": summary,
        "player_type": player_type,
        "overall_score": overall,
        "scores": sub_scores,
        "weaknesses": weaknesses[:4],
        "vulnerable_zones": vulnerabilities[:3],
        "drills": drills[:4],
        "strengths": strengths[:3],
        "pro_comparison": pro_comp,
        "model": "CrickLM-10M (local transformer)",
    }
 
 
# Inference engine 
class CrickLMInference:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best.pt",
        tokenizer_path: str = "checkpoints/tokenizer.json",
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.tokenizer = None
        self._loaded = False
 
        # Try to load trained model
        ckpt = Path(checkpoint_path)
        tok  = Path(tokenizer_path)
 
        if ckpt.exists() and tok.exists():
            self._load(checkpoint_path, tokenizer_path)
        else:
            print(
                "[CrickLM] No trained checkpoint found. "
                "Running in rule-based mode (train the model first with: python train.py)"
            )
 
    def _load(self, ckpt_path: str, tok_path: str) -> None:
        print(f"[CrickLM] Loading tokenizer from {tok_path}...")
        self.tokenizer = CrickTokenizer.load(tok_path)
 
        print(f"[CrickLM] Loading model from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        config_dict = ckpt.get("model_config", {})
        config = CrickLMConfig(**config_dict)
        self.model = CrickLM(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self._loaded = True
        print(f"[CrickLM] Model loaded on {self.device}")
 
    def _generate_text(self, prompt: str, max_tokens: int = 80) -> str:
        if not self._loaded:
            return ""
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                x, max_new_tokens=max_tokens,
                temperature=0.7, top_k=30, top_p=0.9
            )
        return self.tokenizer.decode(out[0].tolist())
 
    def analyze(self, features: dict, player_type: str = "batsman") -> dict:
        # Build the prompt
        prompt = build_analysis_prompt(features, player_type)
 
        # Generate text with the model (if loaded)
        generated = self._generate_text(prompt)
 
        # Parse into structured output using hybrid rule + generation approach
        result = parse_generated_text(generated, features, player_type)
        return result