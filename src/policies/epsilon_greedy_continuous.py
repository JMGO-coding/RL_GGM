import numpy as np

class EpsilonGreedyPolicyContinuous:

    def __init__(self, epsilon: float, num_actions: int):
        """
        Inicializa la política epsilon-greedy.

        :param epsilon: Probabilidad de exploración.
        :param num_actions: Número de acciones posibles.
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        self.epsilon = epsilon
        self.num_actions = num_actions

    def get_action(self, active_features, w):
        """
        Selecciona una acción según una política epsilon-greedy.
        :param active_features: Características activas del estado actual.
        :param w: Matriz de pesos de dimensión [n_features, n_actions].
        :return: Política seleccionada.
        """
        # Con probabilidad epsilon, seleccionamos una acción aleatoria
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            # Con probabilidad 1-epsilon, seleccionamos la acción que maximiza Q(s, a)
            q_vals = self.compute_q_values(active_features, self.num_actions, w)
            action = np.argmax(q_vals)

        return action

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

    def compute_q_values(active_features, num_actions, w):
        """
        Calcula Q(s,a) para todas las acciones a, dado un vector de features activas.
    
        Parámetros:
        - w: array 2D de dimensiones [n_features, num_actions]
        - active_features: lista/array de índices de features activas
        - num_actions: número de acciones
    
        Retorna:
        - q_vals: array 1D de longitud num_actions, donde q_vals[a] = Q(s,a)
        """
        q_vals = np.zeros(num_actions)
        for a in range(num_actions):
            q_vals[a] = q_value(active_features, a, w) # Invoca a la función anterior
        return q_vals
