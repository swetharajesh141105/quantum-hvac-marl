import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tou_env import TariffHVACEnv, evaluate_schedule, TARIFF, HOURS
from day_ahead_planner import build_qubo, plan

def test_obs_is_7_dim():
    obs, _ = TariffHVACEnv().reset(seed=0)
    assert obs.shape == (7,) and obs[5] == TARIFF[0]

def test_always_off_is_not_free():
    m = evaluate_schedule([0] * HOURS, seed=0)
    assert m["violation_hours"] > 0          # collapse is penalised

def test_planner_returns_binary_24h():
    s = plan(np.array([0]*9 + [1]*9 + [0]*6))
    assert len(s) == HOURS and set(np.unique(s)) <= {0, 1}
