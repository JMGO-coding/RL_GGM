#######################################
from abc import abstractmethod
import numpy as np
from .agent import Agent
from policies.epsilon_soft import EpsilonSoftPolicy
from policies.greedy_from_Q import GreedyFromQPolicy
#######################################


class AgentMCOnPolicyAllVisits(Agent):
    def __init__(self, env: gym.Env, epsilon: float, decay: bool, discount_factor: float):
        """
        Inicializa el agente de decisión
        """

        assert 0 <= epsilon <= 1
        assert 0 <= discount_factor <= 1
        
        # Environment del agente
        self.env: gym.Env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.n_visits = np.zeros([env.observation_space.n, self.nA])
        self.returns = np.zeros([env.observation_space.n, self.nA])
        self.epsilon_soft_policy = EpsilonSoft(epsilon=self.epsilon, nA=self.nA)
        self.stats = 0.0
        self.list_stats = []

    def reset(self):
        """
        Reinicia el agente
        """
        self.Q = np.zeros([self.env.observation_space.n, self.nA])
        self.n_visits = np.zeros([self.env.observation_space.n, self.nA])
        self.returns = np.zeros([self.env.observation_space.n, self.nA])
        self.stats = 0.0
        self.list_stats = []
    
    def get_soft_action(self, state):
        """
        Selecciona una acción en base a un estado de partida y una política epsilon-soft
        """
        return self.epsilon_soft_policy.get_action(self.Q, state)

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
            new_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode.append((state, action, reward))  # Almacenar estado, acción y recompensa
            state = new_state

        return episode
        
    def update(self, episode):
        """
        Actualiza Q al final del episodio utilizando todas las visitas
        """
        
        G = 0.0  # Retorno
        # Recorrer el episodio en orden inverso
        for (state, action, reward) in reversed(episode):
            G = self.discount_factor * G + reward
            
            self.n_visits[state, action] += 1  # Contamos cuántas veces hemos visitado (state, action)
            self.returns[state, action] += G  # Acumulamos los retornos

            # Usamos el promedio de los retornos observados
            self.Q[state, action] = self.returns[state, action] / self.n_visits[state, action]
        
    def train(self, num_episodes):
        for t in tqdm(range(num_episodes)):
            if self.decay:
                self.epsilon = min(1.0, 1000.0/(t+1))
                
            episode = self.full_episode(seed = t)  # Generar episodio
            self.update(episode)  # Actualizar Q

            # Guardamos datos sobre la evolución
            self.stats += G
            self.list_stats.append(self.stats/(t+1))

            # Para mostrar la evolución
            if t % step_display == 0 and t != 0:
                print(f"success: {stats/t}, epsilon: {epsilon}")

    def stats(self):
        """
        Retorna los resultados estadísticos, incluyendo el promedio de recompensas por episodio
        y la evolución de la recompensa acumulada por episodio
        """
        # Retorna el promedio acumulado de la recompensa por episodio
        avg_stats = self.stats / len(self.list_stats) if len(self.list_stats) > 0 else 0

        return avg_stats, self.list_stats
