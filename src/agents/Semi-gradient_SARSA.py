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

        # Número de acciones en el entorno original
        self.num_actions = self.tcenv.action_space.n

        # Número total de características en el aproximador lineal:
        self.total_features = self.tcenv.n_tilings * np.prod(self.tcenv.bins)

        # Inicialización de los pesos w
        self.w = np.zeros([self.total_features, self.num_actions])

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
        self.w = np.zeros([self.total_features, self.num_actions])
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.returns = []  # Para guardar la recompensa total de cada episodio
    
    def get_action(self, active_features):
        """
        Selecciona una acción en base a una política epsilon-greedy.
        """
        return self.epsilon_greedy_policy.get_action(active_features, self.w)

    def q_value(self, active_features, a):
        """
        Calcula q(s,a) como la suma de los pesos para los índices activos.
    
        Parámetros:
          - active_features: lista de índices de features activas para el estado s.
          - a: acción seleccionada.
    
        Retorna:
          - q: valor aproximado de Q(s,a).
        """
        return self.w[active_features, a].sum()

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
            # Ejecutar la acción 'a' y obtener la siguiente observación
            obs_next, reward, terminated, truncated, info = self.tcenv.step(a)
            done = terminated or truncated
            total_reward += reward
            steps += 1

             # Después de step, tcenv.last_active_features se actualiza para el nuevo estado s'
            active_features_next = self.tcenv.last_active_features

           # Seleccionar la siguiente acción a' (si el episodio continúa)
            if not done:
                a_next = self.get_action(active_features_next)
            else:
                a_next = None  # No se usa si es terminal

            # Calcular Q(s,a) para el estado actual y la acción tomada
            self.update(active_features, a, active_features_next, a_next, reward, done)

            if done:
                break

            # Actualiza estado y acción para el siguiente paso
            active_features = active_features_next
            a = a_next

        # Guardar estadísticas
        self.returns.append(total_reward)
        self.stats += total_reward  # Acumular la recompensa total del episodio
        self.list_stats.append(self.stats / (len(self.list_stats) + 1))  # Promedio acumulado
        self.episode_lengths.append(steps)

    def update(self, active_features, a, active_features_next, a_next, reward, done):
        """
        Actualiza los pesos w utilizando el método SARSA semi-gradiente.
        """
        # Calcular Q(s,a) para el estado actual y la acción tomada
        q_sa = self.q_value(active_features, a)
        # Si no es estado terminal, calcular Q(s',a')
        if not done:
            q_sap = self.q_value(active_features_next, a_next)
            delta = reward + self.discount_factor * q_sap - q_sa
        else:
            delta = reward - q_sa

        # Actualizar los pesos solo en las features activas para la acción 'a'
        for i in active_features:
            self.w[i, a] += self.alpha * delta
        
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
                total_reward = self.returns[-1]
                print(f"Episode {t+1}/{num_episodes}, total reward: {total_reward}")

        # Después de entrenar, evaluar la política
        avg_return = np.mean(self.returns)
        print(f"Average return over {num_episodes} episodes: {avg_return}")

    def get_stats(self):
        """
        Retorna los resultados estadísticos, incluyendo la evolución de
        la recompensa acumulada por episodio y la longitud de los episodios.
        """
        return self.list_stats, self.episode_lengths
