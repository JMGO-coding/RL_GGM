# RL_GGM
Segunda entrega de la asignatura de Aprendizaje por Refuerzo (RL)

## Información
- **Alumnos:** Gil, Ana; García, José María; Malest, Levi.
- **Asignatura:** Extensiones de Machine Learning.
- **Curso:** 2024/2025
- **Grupo:** GGM
 
## Descripción
Este repositorio ha sido creado para realizar un estudio comparativo entre distintos algoritmos clásicos del aprendizaje por refuerzo, sobre el problema del aprendizaje en entornos complejos.
Contiene un informe del experimento en formato ``.pdf`` y otros ficheros ``.py`` y ``.ipynb`` que dan respaldo a los contenidos que aparecen en el informe. 

## Estructura
- /docs/ $\hspace{0.1 cm}$ Contiene los ficheros ``.pdf``
- /src/   $\hspace{0.42 cm}$ Contiene los scripts de python ``.py``, organizados en las siguientes subcarpetas:
  
  - ./agents/ $\hspace{0.2 cm}$ --> $\hspace{0.15 cm}$ Aquí encontramos las diferentes clases elaboradas para modelizar los agentes
  - ./memories/ $\hspace{1.2 cm}$ --> $\hspace{0.15 cm}$ Aquí encontramos una clase elaborada para modelizar una memoria para Deep Q-Learning
  - ./networks/ $\hspace{0.65 cm}$ --> $\hspace{0.15 cm}$ Aquí encontramos un script con la red neuronal que se usa en Deep Q-Learning
  - ./policies/ $\hspace{0.65 cm}$ --> $\hspace{0.15 cm}$ Aquí encontramos las políticas de decisión que usan los agentes
  - ./wrappers/ $\hspace{0.65 cm}$ --> $\hspace{0.15 cm}$ Aquí encontramos la implementación de diversos wrappers para los entornos utilizados

## Instalación y Uso
A través de ``main.ipynb`` se podrá navegar entre los distintos notebooks que contiene este repositorio, sin más que clicar los enlaces que allí aparecen (véase sección **[2]**).
Todas estas notebooks están listas para ejecutar sobre Google Colab sin necesidad de ningún paso de instalación previo.

## Tecnologías Utilizadas
- Entorno: Jupyter Notebooks (Google Colab)
- Lenguajes de programación: Python (100%)
