#######################################
from abc import abstractmethod
import numpy as np
from .agent import Agent
from policies.epsilon_greedy_continuous import EpsilonGreedyPolicyContinuous
import gymnasium as gym
from tqdm import tqdm
from wrappers.TileCoding import TileCodingEnv
#######################################


class AgentSemiGradientSARSA(Agent):
    def __init__(self, tcenv: gym.Env, epsilon: float, decay: bool, discount_factor: float, alpha: float):
        """
        Inicializa el agente de decisión.
        """

        assert epsilon > 0
        assert 0 <= discount_factor <= 1
        assert alpha > 0
        
        # Environment del agente
        self.tcenv: gym.Env = tcenv
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.decay = decay
        self.num_actions = env.action_space.n

        # Inicialización de los pesos w
        self.w = np.zeros([env.observation_space.n, self.num_actions])

        # Política basada en epsilon-greedy
        self.epsilon_greedy_policy = EpsilonGreedyPolicyContinuous(epsilon=self.epsilon, num_actions=self.num_actions)

        # Estadísticas para realizar seguimiento del rendimiento
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []    # Lista para almacenar las longitudes de los episodios
        self.returns = []  # Para guardar la recompensa total de cada episodio

    def reset(self):
        """
        Reinicia el agente.
        """
        self.epsilon = self.initial_epsilon
        self.w = np.zeros([self.env.observation_space.n, self.num_actions])
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.returns = []  # Para guardar la recompensa total de cada episodio
    
    def get_action(self, active_features):
        """
        Selecciona una acción en base a una política epsilon-greedy.
        """
        return self.epsilon_greedy_policy.get_action(active_features, self.w)

    def q_value(active_features, a, weights):
        """
        Calcula q(s,a) como la suma de los pesos para los índices activos.
    
        Parámetros:
          - active_features: lista de índices de features activas para el estado s.
          - a: acción seleccionada.
          - weights: matriz de pesos de dimensiones [n_features, n_actions].
    
        Retorna:
          - q: valor aproximado de Q(s,a).
        """
        return weights[active_features, a].sum()

    def run_episode(self, seed):
        """
        Ejecuta un episodio utilizando SARSA semi-gradiente y actualiza Q en cada paso.
        """
        # Resetear el entorno (Gymnasium devuelve (obs, info))
        obs, info = self.tcenv.reset(seed=seed)

        # El método observation() del wrapper actualiza internamente tcenv.last_active_features.
        active_features = self.tcenv.last_active_features  

        # Seleccionar acción inicial usando epsilon-greedy
        a = self.get_action(active_features)
        
        done = False
        total_reward = 0
        steps = 0

        # Recorremos cada paso del episodio
        while not done: 
            # Tomar la acción A, observar R, S'
            new_state, reward, terminated, truncated, info = self.env.step(action)
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
        Actualiza los pesos w utilizando el método SARSA semi-gradiente.
        """
        # Calcular Q(s,a) para el estado actual y la acción tomada
        q_sa = q_value(active_features, a, w)
        # Si no es estado terminal, calcular Q(s',a')
        if not (done or truncated):
            q_sap = q_value(active_features_next, a_next, w)
            delta = reward + gamma * q_sap - q_sa
        else:
            delta = reward - q_sa

        # Actualizar los pesos solo en las features activas para la acción 'a'
        for i in active_features:
            w[i, a] += alpha * delta
        
    def train(self, num_episodes):
        """
        Entrena al agente SARSA semi-gradiente durante un número de episodios
        """    
        step_display = num_episodes / 10
        for t in tqdm(range(num_episodes)):
            if self.decay:
                self.epsilon = min(1.0, 1000.0/(t+1))

                # Actualizar la política con el nuevo valor de epsilon
                self.epsilon_greedy_policy.epsilon = self.epsilon

            # Ejecutar un episodio de SARSA semi-gradiente
            self.run_episode(seed=t)  

            # Para mostrar la evolución
            if t % step_display == 0 and t != 0:
                print(f"success: {self.stats/t}, epsilon: {self.epsilon}")

        # Después de entrenar, evaluar la política
        avg_return = np.mean(returns)
        print(f"Average return over {num_episodes} episodes: {avg_return}")

    def get_stats(self):
        """
        Retorna los resultados estadísticos, incluyendo la evolución de
        la recompensa acumulada por episodio y la longitud de los episodios.
        """
        return self.list_stats, self.episode_lengths
