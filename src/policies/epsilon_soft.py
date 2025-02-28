class EpsilonSoftPolicy:

    def __init__(self, epsilon: float, nA: int):
        """
        Inicializa la política epsilon-soft.

        :param epsilon: Probabilidad de exploración.
        :param nA: Número de acciones posibles.
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        self.epsilon = epsilon
        self.nA = nA

    def get_action_probabilities(self, Q, state):
        """
        Calcula las probabilidades de cada acción según la política epsilon-soft a partir de un estado. Se usa para el entrenamiento.
        :param Q: Matriz de valores Q.
        :param state: Estado actual.
        :return: Ndarray con las probabilidades de cada acción.
        """

        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A
