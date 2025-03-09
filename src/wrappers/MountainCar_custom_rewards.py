import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MountainCarCustomRewards(gym.Wrapper):
    def __init__(self, env, negative_reward=-1, goal_reward=100, velocity_factor=10, progress_factor=5):
        """
        Wrapper para modificar las recompensas del entorno MountainCar.

        :param env: Entorno original (MountainCar).
        :param negative_reward: Penalización por cada paso.
        :param goal_reward: Recompensa al alcanzar la cima.
        :param velocity_factor: Factor de recompensa por velocidad.
        :param progress_factor: Factor de recompensa por progreso en la posición.
        """
        super().__init__(env)
        self.negative_reward = negative_reward
        self.goal_reward = goal_reward
        self.velocity_factor = velocity_factor
        self.progress_factor = progress_factor
        self.last_position = None

    def reset(self, **kwargs):
        """ Reinicia el entorno y guarda la posición inicial. """
        obs, info = self.env.reset(**kwargs)
        self.last_position = obs[0]  # Guardar la posición inicial
        return obs, info

    def step(self, action):
        """
        Modifica la recompensa del entorno.
        """
        observation, reward, done, truncated, info = super().step(action)
        position, velocity = observation  # Extraer la posición y la velocidad
        
        # Penalización por cada paso
        reward = self.negative_reward

        # Incentivar la velocidad
        reward += self.velocity_factor * abs(velocity)  

        # Incentivar el progreso en la posición (evita restar None)
        if self.last_position is not None:
            reward += self.progress_factor * (position - self.last_position)

        # Actualizar la última posición
        self.last_position = position

        # Si llega a la cima, darle una gran recompensa y terminar el episodio
        if position >= 0.5:
            reward += self.goal_reward
            done = True  
        
        return observation, reward, done, truncated, info



