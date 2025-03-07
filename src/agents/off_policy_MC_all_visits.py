#######################################
from abc import abstractmethod
import numpy as np
from .agent import Agent
from policies.epsilon_soft import EpsilonSoftPolicy
from policies.greedy_from_Q import GreedyFromQPolicy
import gymnasium as gym
from tqdm import tqdm
#######################################


class AgentMCOffPolicyAllVisits(Agent):
    def __init__(self, env: gym.Env, epsilon: float, decay: bool, discount_factor: float):
        """
        Inicializa el agente de decisión
        """

        assert 0 <= epsilon <= 1
        assert 0 <= discount_factor <= 1
        
        # Environment del agente
        self.env: gym.Env = env
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.C = np.zeros([env.observation_space.n, self.nA])
        self.epsilon_soft_policy = EpsilonSoftPolicy(epsilon=self.epsilon, nA=self.nA)  # Política de comportamiento (b)
        self.greedy_policy = GreedyFromQPolicy(env=self.env, Q=self.Q)    # Política objetivo (pi)
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []    # Lista para almacenar las longitudes de los episodios

    def reset(self):
        """
        Reinicia el agente
        """
        self.epsilon = self.initial_epsilon
        self.Q = np.zeros([self.env.observation_space.n, self.nA])
        self.C = np.zeros([self.env.observation_space.n, self.nA])
        self.greedy_policy = GreedyFromQPolicy(env=self.env, Q=self.Q)
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
    
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

    def full_episode(self, seed):
        """
        Genera un episodio completo siguiendo la política epsilon-soft
        """
        state, info = self.env.reset(seed=seed)
        done = False
        episode = []

        # Generar un episodio siguiendo la política epsilon-soft
        while not done:    
            action = self.get_soft_action(state)
            new_state, reward, terminated, truncated, info = self.env.step(action)
            
            done = terminated or truncated
            episode.append((state, action, reward))  # Almacenar estado, acción y recompensa
            state = new_state

        return episode
        
    def update(self, episode):
        """
        Actualiza Q y W al final del episodio utilizando todas las visitas
        """
        G = 0  # Retorno
        W = 1  # Peso
        
        # Recorrer el episodio en orden inverso
        for (state, action, reward) in reversed(episode):
            # Calcular el retorno acumulado
            G = self.discount_factor * G + reward

            # Actualizar C(s, a)
            self.C[state,action] += W

            # Actualizar Q(s, a)
            self.Q[state, action] += (W / self.C[state,action]) * (G - self.Q[state,action])

            # Actualizar política objetivo (pi)
            self.greedy_policy.Q = self.Q  # Actualizar la matriz Q para la política greedy
            self.greedy_policy.pi = self.greedy_policy.compute_policy_matrix()  # Recalcular la política óptima

            # Si la acción tomada no es la acción según la política objetivo (pi), salir del bucle
            if action != self.get_greedy_action(state):
                break

            # Obtenemos las probabilidades de acción de la política de comportamiento (b) 
            action_probabilities_b = self.epsilon_soft_policy.get_action_probabilities(self.Q, state)

            # Actualizar el peso W
            W = W * 1 / action_probabilities_b[action]
            
        # Guardamos datos sobre la evolución
        self.stats += G
        
    def train(self, num_episodes):
        step_display = num_episodes / 10
        for t in tqdm(range(num_episodes)):
            if self.decay:
                self.epsilon = min(1.0, 1000.0/(t+1))
                
            episode = self.full_episode(seed = t)  # Generar episodio
            self.update(episode)  # Actualizar Q

            self.list_stats.append(self.stats/(t+1))
            self.episode_lengths.append(len(episode))

            # Para mostrar la evolución
            if t % step_display == 0 and t != 0:
                print(f"success: {self.stats/t}, epsilon: {self.epsilon}")

    def get_stats(self):
        """
        Retorna los resultados estadísticos, incluyendo la evolución de
        la recompensa acumulada por episodio y la longitud de los episodios
        """

        return self.list_stats, self.episode_lengths
