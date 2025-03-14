#######################################
from abc import abstractmethod
import numpy as np
from .agent import Agent
from policies.epsilon_soft import EpsilonSoftPolicy
import gymnasium as gym
from tqdm import tqdm
#######################################


class AgentSARSA(Agent):
    def __init__(self, env: gym.Env, epsilon: float, decay: bool, discount_factor: float, alpha: float):
        """
        Inicializa el agente de decisión
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
        self.nA = env.action_space.n

        # Inicialización de Q(s, a) con valores arbitrarios (a excepción de terminales)
        self.Q = np.zeros([env.observation_space.n, self.nA])

        # Política basada en epsilon-greedy
        self.epsilon_greedy_policy = EpsilonSoftPolicy(epsilon=self.epsilon, nA=self.nA)

        # Estadísticas para realizar seguimiento del rendimiento
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []    # Lista para almacenar las longitudes de los episodios

    def reset(self):
        """
        Reinicia el agente
        """
        self.epsilon = self.initial_epsilon
        self.epsilon_greedy_policy = EpsilonSoftPolicy(epsilon=self.initial_epsilon, nA=self.nA)
        self.Q = np.zeros([self.env.observation_space.n, self.nA])
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
        Ejecuta un episodio utilizando SARSA y actualiza Q en cada paso
        """
        # Inicializar S
        state, info = self.env.reset(seed=seed)

        # Elegir A a partir de S usando política epsilon-greedy
        action = self.get_action(state) 
        
        done = False
        total_reward = 0
        steps = 0

        # Recorremos cada paso del episodio
        while not done: 
            # Tomar la acción A, observar R, S'
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Elegir A' a partir de S' usando política epsilon-greedy
            if not done:
                next_action = self.get_action(next_state)
            else:
                next_action = None  # No hay acción en el estado terminal

            # Actualización de Q(S, A) con SARSA
            self.update(state, action, reward, next_state, next_action)

            # Avanzar al siguiente estado y acción
            state = next_state
            action = next_action

        # Guardar estadísticas
        self.stats += total_reward  # Acumular la recompensa total del episodio
        self.list_stats.append(self.stats / (len(self.list_stats) + 1))  # Promedio acumulado
        self.episode_lengths.append(steps)

    def update(self, state, action, reward, next_state, next_action):
        """
        Actualiza Q en base a la ecuación de actualización de SARSA
        """
        target = reward
        if next_action is not None:  # Si no es estado terminal
            target += self.discount_factor * self.Q[next_state, next_action]
    
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
    def train(self, num_episodes):
        """
        Entrena al agente SARSA durante un número de episodios
        """
        step_display = num_episodes / 10
        for t in tqdm(range(num_episodes)):
            if self.decay:
                self.epsilon = min(1.0, 1000.0/(t+1))

                # Actualizar la política con el nuevo valor de epsilon
                self.epsilon_greedy_policy.epsilon = self.epsilon

            # Ejecutar un episodio de SARSA
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
