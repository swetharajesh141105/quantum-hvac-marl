import pytest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hvac_env import HVACEnv

def test_reset():
    env = HVACEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)

def test_step():
    env = HVACEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)

def test_episode_ends():
    env = HVACEnv()
    env.reset()
    for _ in range(env.max_steps):
        env.step(0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated == True

def test_action_space():
    env = HVACEnv()
    env.reset()
    env.step(0)
    env.step(1)
    with pytest.raises(AssertionError):
        env.step(2)
