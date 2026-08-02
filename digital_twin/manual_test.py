from hvac_env import HVACEnv
import numpy as np

env = HVACEnv()
obs, info = env.reset()

print("Step | Action | Room Temp | Outdoor Temp | Reward")
print("-------------------------------------------------")

for i in range(20):
    action = np.random.randint(0, 2)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"{i+1:4d} |   {action}    |   {obs[0]:.2f}°C   |    {obs[1]:.2f}°C    |  {reward:.2f}")
    if terminated:
        print("Day finished!")
        break
