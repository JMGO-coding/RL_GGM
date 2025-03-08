#######################################
from abc import abstractmethod
import numpy as np
from .agent import Agent
from policies.epsilon_greedy_DQN import EpsilonGreedyPolicyDQN
import gymnasium as gym
from tqdm import tqdm
from networks.DQN_Network impor DQN_Network
import torch
import torch.optim as optim
from memories.ReplayMemory import ReplayMemory
#######################################


class AgentDeepQLearning(Agent):
    def __init__(self, env: gym.Env, epsilon: float, decay: bool, discount_factor: float, alpha: float, memory_capacity: int, batch_size: int, learning_rate: float):
        """
        Inicializa el agente de  Deep Q-Learning.

        Parámetros:
        - env (gym.Env): El entorno en el que el agente interactúa.
        - epsilon (float): Probabilidad de exploración para la política epsilon-greedy.
        - decay (bool): Si la epsilon debe decrecer durante el entrenamiento.
        - discount_factor (float): Factor de descuento gamma.
        - alpha (float): Tasa de aprendizaje para el optimizador.
        - memory_capacity (int): Tamaño máximo de la memoria de repetición.
        - batch_size (int): Tamaño del minibatch para el entrenamiento.
        - learning_rate (float): Tasa de aprendizaje del optimizador.
        """
        assert 0 <= epsilon <= 1
        assert 0 <= discount_factor <= 1
        assert 0 < alpha <= 1
        
        # Environment del agente
        self.env: gym.Env = env
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.decay = decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Definir el espacio de acciones
        self.nA = env.action_space.n

        # Red neuronal para Q(s, a)
        self.dqn_network = DQN_Network(input_dim=env.observation_space.shape[0], num_actions=self.nA)
        self.optimizer = optim.Adam(self.dqn_network.parameters(), lr=self.learning_rate)

        # Memoria de repetición
        self.memory = ReplayMemory(memory_capacity)

        # Estadísticas para realizar seguimiento del rendimiento
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []    # Lista para almacenar las longitudes de los episodios

    def reset(self):
        """
        Reinicia el agente
        """
        self.epsilon = self.initial_epsilon
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
    
    def get_action(self, state):
        """
        Selecciona una acción en base a un estado de partida y una política epsilon-greedy
        """
        return self.epsilon_greedy_policy.get_action(self.Q, state)

    def run_episode(self, seed):
        """
        Ejecuta un episodio utilizando Q-Learning y actualiza Q en cada paso
        """
        # Inicializar S
        state, info = self.env.reset(seed=seed)
        
        done = False
        total_reward = 0
        steps = 0

        # Recorremos cada paso del episodio
        while not done: 
            # Elegir A a partir de S usando política epsilon-greedy
            action = self.get_action(state) 
        
            # Tomar la acción A, observar R, S'
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Actualización de Q(S, A) con Q-Learning
            self.update(state, action, reward, next_state)

            # Avanzar al siguiente estado
            state = next_state

        # Guardar estadísticas
        self.stats += total_reward  # Acumular la recompensa total del episodio
        self.list_stats.append(self.stats / (len(self.list_stats) + 1))  # Promedio acumulado
        self.episode_lengths.append(steps)

    def update(self, state, action, reward, next_state):
        """
        Actualiza Q en base a la ecuación de actualización de Q-Learning
        """
        # Actualización de Q(S, A) con la fórmula de Q-Learning
        self.Q[state, action] += self.alpha * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state, action])
        
    def train(self, num_episodes):
        """
        Entrena al agente Q-Learning durante un número de episodios
        """
        step_display = num_episodes / 10
        for t in tqdm(range(num_episodes)):
            if self.decay:
                self.epsilon = min(1.0, 1000.0/(t+1))

                # Actualizar la política con el nuevo valor de epsilon
                self.epsilon_greedy_policy.epsilon = self.epsilon

            # Ejecutar un episodio de Q-Learning
            self.run_episode(seed=t)  

            # Para mostrar la evolución
            if t % step_display == 0 and t != 0:
                print(f"success: {self.stats/t}, epsilon: {self.epsilon}")

    def get_stats(self):
        """
        Retorna los resultados estadísticos, incluyendo la evolución de
        la recompensa acumulada por episodio y la longitud de los episodios
        """
        return self.list_stats, self.episode_lengths
