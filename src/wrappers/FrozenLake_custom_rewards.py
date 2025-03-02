import gymnasium as gym

class FrozenLakeCustomRewards(gym.Wrapper):
    def __init__(self, env, hole_penalty=-1.0):
        super().__init__(env)
        self.hole_penalty = hole_penalty  # Penalización por caer en un agujero
        self.goal_state = self.env.observation_space.n - 1
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Si el episodio termina y la recompensa es 0, significa que caímos en un agujero
        if terminated and next_state != self.goal_state:
            reward = self.hole_penalty  # Penalización por fallar

        return next_state, reward, terminated, truncated, info
