"""
XAI Explainer.
Takes the reasoning dict from marl_agents.joint_decision_verbose() (or a
QUBO-optimized schedule from qubo_scheduler.py) and turns it into a
plain-English sentence describing WHY the AC did what it did.
"""


def explain_decision(action, reasoning):
    """Explain a single ON/OFF decision from joint_decision_verbose()."""
    temp = reasoning["temp"]
    occupancy = reasoning["occupancy"]
    votes = reasoning["votes"]
    override = reasoning["override"]

    state = "ON" if action == 1 else "OFF"
    reasons = []

    if override == "too_hot":
        reasons.append(f"room temperature {temp:.1f}°C exceeded the safety limit "
                        f"({reasoning['comfort_high']}°C), so AC was forced ON")
    elif override == "too_cold":
        reasons.append(f"room temperature {temp:.1f}°C dropped below the safety limit "
                        f"({reasoning['comfort_low']}°C), so AC was forced OFF")
    else:
        agreeing = [obj for obj, v in votes.items() if v == action]
        disagreeing = [obj for obj, v in votes.items() if v != action]

        if agreeing:
            reasons.append(f"{', '.join(agreeing)} agent(s) voted {state}")
        if disagreeing:
            reasons.append(f"{', '.join(disagreeing)} agent(s) voted the opposite way "
                            f"but were outweighed")

        if not occupancy:
            reasons.append("room is currently empty")
        else:
            reasons.append(f"room temperature is {temp:.1f}°C")

    return f"AC turned {state} because: " + "; ".join(reasons) + "."


def explain_schedule(schedule, preferred_actions):
    """Explain a 24h QUBO-optimized schedule against what the agents
    originally preferred hour-by-hour."""
    lines = []
    for t, (qubo_action, agent_pref) in enumerate(zip(schedule, preferred_actions)):
        state = "ON" if qubo_action else "OFF"
        if qubo_action == agent_pref:
            lines.append(f"Hour {t:02d}: AC {state} — matches what the agents preferred.")
        else:
            pref_state = "ON" if agent_pref else "OFF"
            lines.append(
                f"Hour {t:02d}: AC {state} — agents preferred {pref_state}, but QUBO "
                f"overrode this to reduce unnecessary switching across the 24h schedule."
            )
    return "\n".join(lines)


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from marl_agents import joint_decision_verbose

    agents = {}
    for objective in ["energy", "comfort", "carbon"]:
        from marl_agents import robust_load 
        agents[objective] = robust_load(objective) 

    scenarios = [
        {"temp": 30.0, "outdoor_temp": 35.0, "occupancy": 1, "t": 0},
        {"temp": 22.1, "outdoor_temp": 28.0, "occupancy": 0, "t": 6},
        {"temp": 28.5, "outdoor_temp": 33.0, "occupancy": 1, "t": 12},
        {"temp": 20.0, "outdoor_temp": 24.0, "occupancy": 1, "t": 18},
    ]

    print("=== Single-decision explanations ===")
    for s in scenarios:
        action, reasoning = joint_decision_verbose(agents=agents, **s)
        print(explain_decision(action, reasoning))

    print("\n=== QUBO 24h schedule explanation ===")
    try:
        from qubo_scheduler import load_federated_agents, get_preferred_actions, build_qubo, solve_schedule
        agents2 = load_federated_agents()
        preferred = get_preferred_actions(agents2)
        bqm = build_qubo(preferred)
        schedule = solve_schedule(bqm)
        print(explain_schedule(schedule, preferred))
    except ImportError as e:
        print(f"Skipping schedule demo — missing dependency: {e}")