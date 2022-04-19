import gym
from catboost import CatBoostClassifier


env = gym.make('CartPole-v0')
target = 500  # maximum number of iterations

model = CatBoostClassifier(iterations=5,
                           learning_rate=0.01,
                           depth=2)

max_result_total = 0
train_data = []
train_labels = []

# Train model during 1000 iterations, first 100 iterations with randomly selected actions
for epoch in range(1000):
    actions = []
    states = []

    state = env.reset()
    env._max_episode_steps = target
    done = False

    steps_total = 0
    while not done:
        env.render()

        if epoch < 100:
            action = env.action_space.sample()  # choose random action
        else:
            action = model.predict(state)   # choose action based on the current state

        states.append(state)
        actions.append(action)

        state, reward, done, info = env.step(action)
        steps_total += 1

    max_result_total = max(max_result_total, steps_total)

    # Teach classifier (all steps are good, but last steps are bad):
    wrong_steps = 0 if steps_total == target else 4
    if steps_total < max_result_total:
        wrong_steps = max(4, steps_total//2)

    # Teach on positive results:
    for i, state in enumerate(states[:-wrong_steps]):
        train_data.append([str(r) for r in state.tolist()])
        train_labels.append(actions[i])

    # Teach on negative results:
    for i, state in enumerate(states[-wrong_steps:]):
        train_data.append([str(r) for r in state.tolist()])
        train_labels.append(abs(actions[-wrong_steps + i] - 1))  # choose opposite action

    if len(set(train_labels)) > 1:
        model.fit(train_data, train_labels)

    print(epoch, steps_total, max_result_total)
