import dqn_module
import numpy as np

def main():
    # Configurar entorno y servicio
    state_size = 4
    action_size = 2
    env = dqn_module.SimpleEnvironment(state_size, action_size)
    dqn = dqn_module.DQNService(state_size, action_size)

    # Entrenamiento
    episodes = 25
    max_steps = 100
    gamma = 0.99
    learning_rate = 0.1

    for episode in range(episodes):
        state = env.reset()
        states = []
        actions = []
        rewards = []

        for _ in range(max_steps):
            action = dqn.predict(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        dqn.train(states, actions, rewards, gamma, learning_rate)
        print(f"Episodio {episode + 1} completado")

    # Predicción
    state = env.reset()
    action = dqn.predict(state)
    print(f"Acción predicha para estado {state.features}: {action.id}")

if __name__ == "__main__":
    main()