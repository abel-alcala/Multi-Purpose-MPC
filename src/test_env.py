from RL_Env import TrackingEnv

env = TrackingEnv(simMode='Sim_Track', renderMode='human')
obs, info = env.reset()

attempt = 1
for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        reason = "lap complete" if info['lap_complete'] else "off track"
        print(f"Attempt {attempt} ended at step {step} ({reason})")
        obs, _ = env.reset()
        attempt += 1

env.close()