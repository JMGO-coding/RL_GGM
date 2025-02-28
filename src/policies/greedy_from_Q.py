class GreedyFromQPolicy:

    def __init__(self, env, Q):
        """
        Inicializa la política greedy a partir de la matriz de valores Q.

        Parámetros:
        - env: el entorno de OpenAI Gym.
        - Q: la matriz de valores Q de la política óptima.
        """
        self.env = env
        self.Q = Q
        self.pi = self.compute_policy_matrix()

    def compute_policy_matrix(self):
        """
        Calcula la matriz de política óptima π* a partir de Q.

        Retorna:
        - Una matriz de dimensiones (n_estados, n_acciones) con la política óptima.
        """
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        pi_star = np.zeros((n_states, n_actions))

        for state in range(n_states):
            best_action = np.argmax(self.Q[state, :])  # Selecciona la mejor acción según Q
            pi_star[state, best_action] = 1  # Política determinista (acción óptima)

        return pi_star

    def new_episode(self, initial_state=None):
        """
        Genera una secuencia de acciones siguiendo la política óptima desde un estado inicial.

        Parámetros:
        - initial_state: Estado inicial del episodio. Si es None, se obtiene desde env.reset().

        Retorna:
        - Una cadena con la secuencia de estados y otra con la secuencia de acciones tomadas.
        """
        state, info = self.env.reset() if initial_state is None else (initial_state, None)
        done = False
        states = [state]
        actions = []

        while not done:
            action = np.argmax(self.Q[state, :])  # Selecciona la mejor acción según Q
            actions.append(action)
            state, reward, terminated, truncated, info = self.env.step(action)
            states.append(state)
            done = terminated or truncated
        
        return states, actions  # Devuelve la secuencia de acciones tomadas

    def get_action(self, state):
        """
        Selecciona una acción según una política greedy.
        :param state: Estado actual.
        :return: Política seleccionada.
        """

        return np.argmax(self.Q[state, :])
