import gymnasium as gym
from gymnasium import Wrapper

class CliffWalkingCustomRewards(Wrapper):
    def __init__(self, env, fall_penalty=-100, step_penalty=-1, goal_reward=100):
        super().__init__(env)
        self.fall_penalty = fall_penalty  # Penalización por caer al acantilado
        self.step_penalty = step_penalty  # Penalización por cada paso
        self.goal_reward = goal_reward    # Recompensa al llegar a la meta
        self.cliff_states = set(range(37, 47))  # Estados del acantilado (según la matriz 4x12)
        self.goal_state = 47  # Último estado es la meta

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Modificar recompensas según la situación
        if next_state in self.cliff_states:  
            reward = self.fall_penalty  # Penalización por caer
        elif next_state == self.goal_state:  
            reward = self.goal_reward  # Recompensa por llegar a la meta
        else:
            reward = self.step_penalty  # Penalización por cada paso normal

        return next_state, reward, terminated, truncated, info
