import torch
import numpy as np

class EpsilonGreedyPolicyDQN:
    def __init__(self, epsilon: float, num_actions: int, dqn_network: torch.nn.Module):
        """
        Inicializa la política epsilon-greedy.

        :param epsilon: Probabilidad de exploración.
        :param num_actions: Número de acciones posibles.
        :param dqn_network: Red neuronal que predice los valores Q.
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."
        
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.dqn_network = dqn_network

    def get_action(self, state):
        """
        Selecciona una acción según una política epsilon-greedy.
        
        :param state: El estado actual del entorno.
        :return: La acción seleccionada.
        """
        # Con probabilidad epsilon, seleccionamos una acción aleatoria
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            # Con probabilidad 1-epsilon, seleccionamos la acción que maximiza Q(s, a)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convertir a tensor
            q_values = self.dqn_network(state_tensor)  # Obtener los valores Q desde la red
            action = torch.argmax(q_values).item()  # Seleccionar la acción con el valor Q máximo

        return action
