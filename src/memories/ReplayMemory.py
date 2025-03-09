import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        """
        Inicializa la memoria de repeticiones.
        
        Parámetros:
        - capacity (int): Número máximo de experiencias que se pueden almacenar.
        """
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """
        Almacena una nueva experiencia en la memoria.
        
        Parámetros:
        - state (torch.Tensor): Estado actual.
        - action (int): Acción tomada.
        - reward (float): Recompensa obtenida.
        - next_state (torch.Tensor): Siguiente estado.
        - done (bool): Indica si el episodio terminó.
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # Elimina el elemento más antiguo si la capacidad se excede
        
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Obtiene un minibatch aleatorio de experiencias.

        Parámetros:
        - batch_size (int): Número de muestras a extraer.

        Retorna:
        - Un batch de experiencias como listas separadas (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertimos las listas en tensores de PyTorch
        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states])
        dones = torch.tensor(dones, dtype=torch.bool)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Retorna la cantidad de elementos almacenados en la memoria.
        """
        return len(self.memory)
