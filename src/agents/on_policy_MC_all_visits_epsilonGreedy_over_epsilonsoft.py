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

        assert 0 <= epsilon <= 1
        assert 0 <= discount_factor <= 1
        
        # Environment del agente
        self.env: gym.Env = env
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.epsilon_soft_policy = EpsilonSoft(epsilon=epsilon, nA=self.nA)
        self.greedy_policy = None
        self.discount_factor = discount_factor
        self.factor = 1
        self.n_visits = np.zeros([env.observation_space.n, self.nA])
        self.returns = np.zeros([env.observation_space.n, self.nA])
        self.G = 0
        self.stats = 0.0
        self.list_stats = []
        self.t = 0

    def reset(self):
        """
        Reinicia el agente
        """
        self.Q = np.zeros([self.env.observation_space.n, nA])
        self.greedy_policy = None
        self.factor = 1
        self.G = 0
        self.stats = 0.0
        self.list_stats = []
        self.t = 0
    
    def get_soft_action(self, state):
        """
        Selecciona una acción en base a un estado de partida y una política epsilon-soft
        """
        return self.epsilon_soft_policy.get_action(self.Q, state)
    
    def get_greedy_action(self, state):
        """
        Selecciona una acción en base a un estado de partida y una política greedy
        """
        return self.greedy_policy.get_action(state)

    def full_episode(self, Q, seed):
        state, info = self.env.reset(seed=seed)
        states = []
        actions = []
        episode = []
        done = False
        # play one episode
        while not done:
            action = self.get_soft_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # update the agent
            agent.update(self, state, action, next_state, reward, terminated, truncated, info)
            # update if the environment is done and the current state
            done = terminated or truncated
            episode.append((state, action, reward))  # Almacena estado, acción y recompensa

            states.append(state)
            actions.append(action)
            state = next_state

        # Al final del episodio, actualizamos Q utilizando todas las visitas
        for (state, action, reward) in episode:
            self.n_visits[state, action] += 1  # Contamos cuántas veces hemos visitado (s, a)
            self.returns[state, action] += self.G  # Acumulamos los retornos
            # Usamos el promedio de los retornos observados
            self.Q[state, action] = self.returns[state, action] / self.n_visits[state, action]

        self.greedy_policy = GreedyFromQPolicy(self.env, self.Q)

        # Guardamos datos sobre la evolución
        self.stats += self.G
        self.list_stats.append(self.stats/(self.t+1))
        self.t += 1
        
        return states, actions
        
    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Actualiza el agente 
        :param state: Estado del environment.
        :param action: Acción que se ha tomado desde el estado.
        :param next_state: Estado al que se transiciona tras la acción.
        :param reward: Recompensa de la acción.
        :param terminated: Refleja si el estado es terminal.
        :param truncated: Refleja si el estado finaliza el episodio sin ser terminal.
        :param info: Información sobre el entorno.
        """
        
        self.G += self.factor * reward
        self.factor *= self.discount_factor

        

        
        

