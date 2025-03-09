import gymnasium as gym

class FrozenLakeCustomRewards(gym.Wrapper):
    def __init__(self, env, hole_penalty=-100, step_penalty=-0.01, goal_reward=100.0):
        super().__init__(env)
        self.hole_penalty = hole_penalty  # Penalización por caer en un agujero
        self.step_penalty = step_penalty  # Penalización por cada paso
        self.goal_reward = goal_reward  # Recompensa por llegar a la meta
        self.goal_state = self.env.observation_space.n - 1  # Estado final (meta)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Penalización por cada paso, excepto si se llega a la meta
        if next_state != self.goal_state:
            reward += self.step_penalty
        
        # Recompensa por llegar a la meta
        if terminated and next_state == self.goal_state:
            reward = self.goal_reward  
        
        # Penalización por caer en un agujero
        if terminated and next_state != self.goal_state:
            reward = self.hole_penalty  

        return next_state, reward, terminated, truncated, info


