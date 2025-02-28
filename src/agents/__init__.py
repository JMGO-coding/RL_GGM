# Importación de módulos o clases
from .agent1 import Agent
from .epsilon_greedy import EpsilonGreedy
from .ucb1 import UCB1
from .ucb2 import UCB2
from .softmax import Softmax
from .gradiente_preferencias import GradienteDePreferencias

# Lista de módulos o clases públicas
__all__ = ['Agent', 'EpsilonGreedy', 'UCB1', 'UCB2', 'Softmax', 'GradienteDePreferencias']
