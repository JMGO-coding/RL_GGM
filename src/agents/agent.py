#######################################
from abc import ABC, abstractmethod
import numpy as np
#######################################


class Agent():
    def __init__(self, env: gym.Env):
        """
        Inicializa el agente de decisión..
        """
        # Environment del agente
        self.env: gym.Env = env

    @abstractmethod
    def get_action(self):
        """
        Selecciona una acción en base a un estado de partida y una política de decisión
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")
      
    @abstractmethod
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Actualiza el agente 
        :param obs: Estado del environment.
        :param action: Acción que se ha tomado desde el estado.
        :param next_obs: Estado al que se transiciona tras la acción.
        :param reward: Recompensa de la acción.
        :param terminated: Refleja si el estado es terminal.
        :param truncated: Refleja si el estado finaliza el episodio sin ser terminal.
        :param info: Información sobre el entorno.
        """

    @abstractmethod
    def reset(self):
        """
        Reinicia el estado del algoritmo (opcional).
        """
