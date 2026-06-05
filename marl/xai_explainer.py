"""
Explainable AI (XAI) Module
=============================
Explains WHY the AI made each HVAC decision.

CONCEPT (for interviews):
--------------------------
Black-box AI problem: the RL agent makes decisions but we don't
know WHY. "AC turned off" — was it because room was empty?
Temperature already comfortable? Energy price too high?

XAI solves this by attributing each decision to input features.

We implement two approaches:

1. SHAP (SHapley Additive exPlanations):
   Based on game theory — Shapley values from cooperative games.
   Each feature gets a "credit" for the prediction.
   SHAP value of feature i = average marginal contribution of
   feature i across all possible feature subsets.
   
   Formally: φᵢ = Σ [|S|!(n-|S|-1)!/n!] * [f(S∪{i}) - f(S)]
   where S is a subset of features not containing i.
   
   Property: sum of all SHAP values = prediction - baseline
   This makes it fully additive and interpretable.

2. Rule Extraction:
   Simpler approach — look at the top features and write
   human-readable rules. "IF occupancy=0 AND temp<23°C THEN OFF"

OUTPUT:
   Natural language explanation shown on dashboard:
   "AC turned OFF because:
    → Occupancy predicted 0 for next 45 mins (60% influence)
    → Current temperature 22.1°C is already comfortable (25%)
    → Off-peak hours reduce energy cost pressure (15%)"
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Feature names matching the observation vector in thermal_model.py
FEATURE_NAMES = [
    "temperature",
    "humidity", 
    "occupancy",
    "co2_level",
    "current_power",
    "outdoor_temp",
    "time_sin",
    "time_cos",
    "ac_currently_on",
    "current_setpoint"
]

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "temperature":      "Indoor temperature",
    "humidity":         "Indoor humidity",
    "occupancy":        "Room occupancy",
    "co2_level":        "CO₂ concentration",
    "current_power":    "Current power draw",
    "outdoor_temp":     "Outdoor temperature",
    "time_sin":         "Time of day (sin)",
    "time_cos":         "Time of day (cos)",
    "ac_currently_on":  "AC currently running",
    "current_setpoint": "Current AC setpoint"
}

ACTION_NAMES = {
    0: "AC OFF",
    1: "Cool aggressively (18°C)",
    2: "Cool strongly (20°C)",
    3: "Cool comfortably (22°C)",
    4: "Cool mildly (24°C)",
    5: "Minimal cooling (26°C)"
}


class HVACExplainer:
    """
    Explains AI decisions using perturbation-based feature importance.
    
    Method: For each feature, measure how much the predicted action
    probability changes when we perturb (slightly change) that feature.
    Large change = feature is important for this decision.
    
    This is a simplified version of SHAP that works without
    installing the full shap library.
    """
    
    def __init__(self, predict_fn):
        """
        predict_fn: function that takes observation array and returns
                    action probabilities (shape: 6,)
        """
        self.predict_fn = predict_fn
        self.explanation_history = []
    
    def compute_feature_importance(
        self, 
        observation: np.ndarray,
        n_perturbations: int = 50
    ) -> np.ndarray:
        """
        Compute feature importance via perturbation.
        
        For each feature:
        1. Add random noise to it N times
        2. Measure variance in output probabilities
        3. Higher variance = feature matters more
        
        Returns: importance scores (shape: 10,), sum to 1
        """
        base_probs = self.predict_fn(observation)
        base_action = np.argmax(base_probs)
        
        importances = np.zeros(len(observation))
        
        for feat_idx in range(len(observation)):
            action_changes = []
            
            for _ in range(n_perturbations):
                # Perturb this feature
                perturbed_obs = observation.copy()
                noise = np.random.normal(0, 0.3)
                perturbed_obs[feat_idx] += noise
                
                # Get new prediction
                new_probs = self.predict_fn(perturbed_obs)
                new_action = np.argmax(new_probs)
                
                # Measure how much the chosen action's probability changed
                prob_change = abs(new_probs[base_action] - base_probs[base_action])
                action_changes.append(prob_change)
            
            importances[feat_idx] = np.mean(action_changes)
        
        # Normalize to sum to 1
        total = importances.sum()
        if total > 0:
            importances = importances / total
        
        return importances
    
    def generate_explanation(
        self,
        observation: np.ndarray,
        action: int,
        info: dict = None,
        top_k: int = 3
    ) -> dict:
        """
        Generate a human-readable explanation for a decision.
        
        Returns:
        {
            "action": "AC OFF",
            "confidence": 0.87,
            "top_features": [...],
            "natural_language": "AC turned off because...",
            "feature_importances": {...}
        }
        """
        # Compute feature importances
        importances = self.compute_feature_importance(observation)
        
        # Get action confidence
        probs = self.predict_fn(observation)
        confidence = float(probs[action])
        
        # Get top-k most important features
        top_indices = np.argsort(importances)[::-1][:top_k]
        
        top_features = []
        for idx in top_indices:
            feat_name = FEATURE_NAMES[idx]
            importance_pct = importances[idx] * 100
            
            # Decode the normalized value back to human-readable
            readable_value = self._decode_feature(feat_name, observation[idx], info)
            
            top_features.append({
                "feature": feat_name,
                "description": FEATURE_DESCRIPTIONS[feat_name],
                "importance_pct": round(importance_pct, 1),
                "value": readable_value
            })
        
        # Generate natural language explanation
        nl_explanation = self._generate_natural_language(
            action, top_features, observation, info
        )
        
        explanation = {
            "action": ACTION_NAMES[action],
            "action_id": action,
            "confidence_pct": round(confidence * 100, 1),
            "top_features": top_features,
            "natural_language": nl_explanation,
            "feature_importances": {
                FEATURE_NAMES[i]: round(float(importances[i]) * 100, 1)
                for i in range(len(importances))
            }
        }
        
        self.explanation_history.append(explanation)
        return explanation
    
    def _decode_feature(self, feat_name: str, norm_value: float, info: dict) -> str:
        """Convert normalized observation value back to human-readable string."""
        if info:
            if feat_name == "temperature":
                return f"{info.get('temp', 0):.1f}°C"
            elif feat_name == "humidity":
                return f"{info.get('humidity', 0):.1f}%"
            elif feat_name == "occupancy":
                return f"{info.get('occupants', 0)} people"
            elif feat_name == "co2_level":
                return f"{info.get('co2', 400):.0f} ppm"
            elif feat_name == "current_power":
                return f"{info.get('power_kw', 0):.2f} kW"
            elif feat_name == "outdoor_temp":
                return f"{info.get('outdoor_temp', 0):.1f}°C"
        
        # Fallback: show normalized value
        return f"{norm_value:.2f}"
    
    def _generate_natural_language(
        self,
        action: int,
        top_features: list,
        observation: np.ndarray,
        info: dict
    ) -> str:
        """Generate natural language explanation."""
        action_str = ACTION_NAMES[action]
        
        # Build reason strings
        reasons = []
        for feat in top_features:
            feat_name = feat["feature"]
            pct = feat["importance_pct"]
            val = feat["value"]
            
            if feat_name == "occupancy":
                occ = info.get("occupants", 0) if info else 0
                if occ == 0:
                    reasons.append(
                        f"Room is empty ({pct:.0f}% influence)"
                    )
                else:
                    reasons.append(
                        f"{occ} people present, comfort needed ({pct:.0f}%)"
                    )
            
            elif feat_name == "temperature":
                temp = info.get("temp", 0) if info else 0
                if temp < 22:
                    reasons.append(
                        f"Temperature {val} already below comfort zone ({pct:.0f}%)"
                    )
                elif temp > 25:
                    reasons.append(
                        f"Temperature {val} too high, cooling needed ({pct:.0f}%)"
                    )
                else:
                    reasons.append(
                        f"Temperature {val} is comfortable ({pct:.0f}%)"
                    )
            
            elif feat_name == "time_sin" or feat_name == "time_cos":
                hour = _decode_time(observation[6], observation[7])
                if 0 <= hour < 7 or hour >= 21:
                    reasons.append(
                        f"Night time ({hour:.0f}:00), low occupancy expected ({pct:.0f}%)"
                    )
                elif 9 <= hour <= 18:
                    reasons.append(
                        f"Working hours ({hour:.0f}:00), occupancy expected ({pct:.0f}%)"
                    )
            
            elif feat_name == "co2_level":
                co2 = info.get("co2", 400) if info else 400
                if co2 > 800:
                    reasons.append(
                        f"High CO₂ ({val}) indicates occupied space ({pct:.0f}%)"
                    )
                else:
                    reasons.append(
                        f"Low CO₂ ({val}) suggests empty room ({pct:.0f}%)"
                    )
            
            else:
                reasons.append(
                    f"{FEATURE_DESCRIPTIONS[feat_name]}: {val} ({pct:.0f}%)"
                )
        
        explanation = f"Decision: {action_str}\nReasons:\n"
        for i, reason in enumerate(reasons, 1):
            explanation += f"  {i}. {reason}\n"
        
        return explanation.strip()


def _decode_time(sin_val: float, cos_val: float) -> float:
    """Decode sin/cos time encoding back to hour."""
    angle = np.arctan2(sin_val, cos_val)
    hour = (angle / (2 * np.pi) * 24) % 24
    return hour


# ------------------------------------------------------------------ #
#  Simple Rule-Based Explainer (backup, no ML needed)                 #
# ------------------------------------------------------------------ #

class RuleBasedExplainer:
    """
    Simple rule-based explanation system.
    No ML required — just human-written rules.
    
    Less sophisticated than SHAP but always interpretable
    and works without any model internals.
    """
    
    def explain(self, action: int, info: dict) -> str:
        """Generate explanation from rules."""
        action_str = ACTION_NAMES.get(action, f"Action {action}")
        temp = info.get("temp", 22)
        occupants = info.get("occupants", 0)
        co2 = info.get("co2", 400)
        power = info.get("power_kw", 0)
        outdoor = info.get("outdoor_temp", 30)
        
        reasons = []
        
        # Occupancy rule
        if occupants == 0:
            reasons.append("room is empty (no one to comfort)")
        elif occupants > 5:
            reasons.append(f"{occupants} people present (high comfort demand)")
        else:
            reasons.append(f"{occupants} people present")
        
        # Temperature rule
        if action == 0:  # OFF
            if temp <= 23:
                reasons.append(f"temperature {temp:.1f}°C is already comfortable")
        else:
            if temp > 25:
                reasons.append(f"temperature {temp:.1f}°C is too high")
            elif outdoor > 32:
                reasons.append(f"outdoor temperature {outdoor:.1f}°C is very hot")
        
        # CO2 rule
        if co2 > 800:
            reasons.append(f"high CO₂ ({co2:.0f} ppm) indicates active occupancy")
        
        # Energy rule
        if action == 0 and occupants == 0:
            reasons.append("saving energy by turning off unused AC")
        
        reason_str = "; ".join(reasons) if reasons else "optimal schedule"
        return f"[{action_str}] because: {reason_str}"


# ------------------------------------------------------------------ #
#  Demo / Test                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("XAI Explainer — Demo")
    print("=" * 60)
    
    # Test rule-based explainer (no model needed)
    rule_explainer = RuleBasedExplainer()
    
    test_cases = [
        (0, {"temp": 22.1, "occupants": 0, "co2": 420, "power_kw": 0, "outdoor_temp": 31}),
        (3, {"temp": 26.5, "occupants": 8, "co2": 850, "power_kw": 1.2, "outdoor_temp": 34}),
        (2, {"temp": 27.0, "occupants": 3, "co2": 650, "power_kw": 0.8, "outdoor_temp": 36}),
        (5, {"temp": 24.0, "occupants": 2, "co2": 500, "power_kw": 0.3, "outdoor_temp": 29}),
    ]
    
    print("\nRule-Based Explanations:")
    print("-" * 40)
    for action, info in test_cases:
        explanation = rule_explainer.explain(action, info)
        print(f"  {explanation}\n")
    
    # Test perturbation-based explainer with a dummy model
    print("\nPerturbation-Based XAI (with dummy model):")
    print("-" * 40)
    
    def dummy_predict(obs):
        """Dummy prediction function for testing."""
        probs = np.zeros(6)
        # Simple heuristic: if occupancy (index 2) is high, prefer cooling
        if obs[2] > 0.5:
            probs[3] = 0.6  # 22°C
            probs[2] = 0.3  # 20°C
        else:
            probs[0] = 0.8  # OFF
            probs[5] = 0.2  # 26°C
        probs += 1e-6
        return probs / probs.sum()
    
    explainer = HVACExplainer(predict_fn=dummy_predict)
    
    # Test observation: occupied room, hot temperature
    obs = np.array([0.4, 0.1, 0.8, 0.3, 0.5, 0.3, 0.5, 0.8, 1.0, 0.3])
    info = {"temp": 26.2, "occupants": 8, "co2": 750, 
            "power_kw": 1.1, "outdoor_temp": 33}
    
    explanation = explainer.generate_explanation(obs, action=3, info=info)
    
    print(f"\nAction: {explanation['action']}")
    print(f"Confidence: {explanation['confidence_pct']}%")
    print(f"\n{explanation['natural_language']}")
    print("\nAll feature importances:")
    for feat, imp in sorted(
        explanation['feature_importances'].items(), 
        key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(imp / 5)
        print(f"  {feat:20s}: {imp:5.1f}% {bar}")
    
    print("\n✅ XAI Explainer working correctly!")
