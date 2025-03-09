import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MountainCarCustomRewards(gym.Wrapper):
    def __init__(self, env, max_steps=500, negative_reward=-1, goal_reward=100):
        """
        Wrapper para el entorno MountainCar para modificar los pasos y recompensas.
        
        :param env: El entorno original (MountainCar).
        :param max_steps: Número máximo de pasos por episodio.
        :param negative_reward: Recompensa por cada paso dado (por defecto -1).
        :param goal_reward: Recompensa al llegar a la cima de la montaña (por defecto 100).
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.negative_reward = negative_reward
        self.goal_reward = goal_reward
        self.steps = 0

    def reset(self, **kwargs):
        """
        Resetea el entorno. Reinicia el contador de pasos.
        """
        self.steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        """
        Modifica la recompensa y el límite de pasos en el entorno.
        """
        # Realiza un paso en el entorno original
        observation, reward, done, truncated, info = super().step(action)
        
        # Aumenta el contador de pasos
        self.steps += 1
        
        # Modifica la recompensa por movimiento
        reward = self.negative_reward
        
        # Si el agente ha alcanzado la cima (posición >= 0.5), asigna la recompensa del objetivo
        if observation[0] >= 0.5:
            reward = self.goal_reward
            done = True  # El episodio termina cuando el agente llega a la cima

        # Si el número de pasos alcanza el límite, termina el episodio
        if self.steps >= self.max_steps:
            done = True  # Forzar el término del episodio si se alcanza el límite de pasos

        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)
