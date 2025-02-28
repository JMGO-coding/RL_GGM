#######################################
from abc import ABC, abstractmethod
import numpy as np
from .agent import Agent
from policies.epsilon_soft import EpsilonSoftPolicy
from policies.greedy_from_Q import GreedyFromQPolicy
#######################################


class On_P_MC_AllVisits(Agent):
    def __init__(self, env: gym.Env, epsilon: float, decay: bool, discount_factor: float):
        """
        Inicializa el agente de decisión..
        """

        assert 0 <= eps <= 1
        assert 0 <= discount_factor <= 1
        
        # Environment del agente
        self.env: gym.Env = env
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, nA])
        self.epsilon_soft_policy = EpsilonSoft(epsilon=epsilon, nA=nA)
        self.greedy_policy = None

    def reset(self):
        """
        Reinicia el agente
        """
        self.Q = np.zeros([env.observation_space.n, nA])
        self.greedy_policy = None

    
    def get_action(self, state):
        """
        Selecciona una acción en base a un estado de partida y una política de decisión
        """
        return self.epsilon_soft_policy.get_action(self.Q, state)
        

    def full_episode(Q, obs):
        states = []
        actions = []
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        return states, actions
        
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
        
        action = self.epsilon_soft_policy.get_action(reward)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))  # Almacena estado, acción y recompensa
        G += factor * reward
        factor *= discount_factor
        state = new_state

        # Al final del episodio, actualizamos Q utilizando todas las visitas
        for (state, action, reward) in episode:
            n_visits[state, action] += 1  # Contamos cuántas veces hemos visitado (s, a)
            returns[state, action] += G  # Acumulamos los retornos
            # Usamos el promedio de los retornos observados
            Q[state, action] = returns[state, action] / n_visits[state, action]

        # Guardamos datos sobre la evolución
        stats += G
        list_stats.append(stats/(t+1))

        
        

