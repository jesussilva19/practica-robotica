from robobo_env import RoboboEnv

env = RoboboEnv()
obs, info = env.reset()

for step in range(200):  # Forzar máximo 200 pasos
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: Term={terminated}, Trunc={truncated}")
    
    if terminated or truncated:
        print(f"✅ Episodio terminó en el paso {step}")
        break
else:
    print("❌ El episodio NUNCA terminó en 200 pasos!")