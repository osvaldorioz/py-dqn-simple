Funcionamiento:

    El código implementa un modelo DQN (Deep Q-Network) en C++ con una arquitectura hexagonal:
        Dominio: State, Action, DQNModel (red neuronal simplificada).
        Puertos: ModelRepository, DQNTrainer, DQNPredictor.
        Capa de aplicación: DQNService para entrenamiento y predicción.
        Adaptadores: InMemoryModelRepository y SimpleEnvironment.
    La interfaz pybind11 permite invocar las funcionalidades desde Python (train, predict).
    El script Python (dqn_test.py) entrena el modelo durante n episodios en un entorno simulado y realiza una predicción.

Salida:
text
Episodio 1 completado
Episodio 2 completado
...
Episodio 10 completado
Acción predicha para estado [0.8049722647217376, 0.5211069496694883, 0.5689997817030652, 0.17938255486769614]: 1

    Indica que el agente completó el entrenamiento y predijo la acción 1 para un estado dado.

Entorno:

    Se utilizó un entorno virtual (myenv) con Python 3.12 en Linux y g++ con soporte para C++17.
    Las dependencias (pybind11, numpy) se instalaron correctamente.
