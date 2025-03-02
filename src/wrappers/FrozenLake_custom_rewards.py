import gymnasium as gym
from gymnasium.wrappers import RewardWrapper

class FrozenLakeCustomRewards(RewardWrapper):
    def __init__(self, env, hole_penalty=-1.0):
        super().__init__(env)
        self.hole_penalty = hole_penalty  # Penalización por caer en un agujero
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Si el episodio termina y la recompensa es 0, significa que caímos en un agujero
        if terminated and reward == 0:
            reward = self.hole_penalty  # Penalización negativa

        return next_state, reward, terminated, truncated, info
