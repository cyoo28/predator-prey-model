from environment import PredatorPreyEnv
from DeepQNetwork import *


# Function to run Deep Q Learning on the environment
def run_dqn(env, local_state_size, action_size, args):
    dqn_reward = []
    dqn_avg_rewards = []
    dqn_successes = 0
    dqn_success_history = []
    steps = 200
    for episode in range(10000):
        episode_loss = 0
        episode_reward = 0
        state = env.reset()

        # Create initial number of DQN agents corresponding to number of designated predators
        active_agents = args["active_agents"]
        DQN_predators = []
        for predator in range(active_agents):
            DQN_predators.append(DQNAgent(local_state_size, action_size, args))

        for s in range(steps):
            # For each step, determine the action of each agent
            # This depends on local state of each agent (predator location and nearest prey location)
            choices = []
            for index, active_predators in enumerate(env.predator_active_list):
                local_state = active_predators.get_local_state(state["prey"])
                Q = DQN_predators[index].get_Q(np.array(local_state).ravel())
                choice = DQN_predators[index].choose_action(env, Q)
                choices.append(choice)
            # Based on the chosen actions of each agent, find the next state and reward
            # Also check if the episode is done (done flag) and if the predators have successfully hunted all prey (terminal flag)
            next_state, reward, done, terminal = env.step(choices)
            # Update the Q networks for each agent
            for index, active_predators in enumerate(env.predator_active_list):
                next_local_state = active_predators.get_local_state(next_state["prey"])
                loss, reward = DQN_predators[index].update_network(Q, np.array(next_local_state).ravel(), choices[index], reward)

            episode_loss += loss
            episode_reward += reward
            if done or s == steps-1:
                dqn_reward.append(episode_reward)
                if (episode + 1) % 100 == 0:
                    dqn_avg_rewards.append(np.mean(dqn_reward))
                    print('Episode: ', episode + 1, '\nAverage Reward over 100 Episodes: ', np.mean(dqn_reward),
                          "\nNumber of Successes over 100 Episodes:", dqn_successes)
                    print('\n')
                    dqn_successes = 0
                    dqn_reward = []
                if terminal:
                    # Update epsilon and scheduler
                    for active_predators in DQN_predators:
                        active_predators.update_epsilon()
                        active_predators.update_scheduler()
                    # Update successes
                    dqn_successes += 1
                dqn_success_history.append(dqn_successes/(episode+1))
                break
            else:
                state = next_state
    return dqn_avg_rewards, dqn_success_history


# Function to run without any learning
def run_fixed(env):
    fixed_reward = []
    fixed_avg_rewards = []
    fixed_successes = 0
    fixed_success_history = []
    steps = 200
    for episode in range(10000):
        episode_reward = 0
        state = env.reset()

        for s in range(steps):
            # For each step, determine the action of each agent
            # This depends on local state of each agent (predator location and nearest prey location)
            prey_positions = []
            for prey in env.prey_active_list:
                prey_positions.append(prey.get_position())
            choices = []
            for index, active_predators in enumerate(env.predator_active_list):
                choices.append(active_predators.fixed_move(prey_positions))
            # Based on the chosen actions of each agent, find the next state and reward
            # Also check if the episode is done (done flag) and if the predators have successfully hunted all prey (terminal flag)
            next_state, reward, done, terminal = env.step(choices)
            episode_reward += reward
            if done or s == steps-1:
                fixed_reward.append(episode_reward)
                if (episode + 1) % 100 == 0:
                    fixed_avg_rewards.append(np.mean(fixed_reward))
                    print('Episode: ', episode + 1, '\nAverage Reward over 100 Episodes: ', np.mean(fixed_reward),
                          "\nNumber of Successes over 100 Episodes:", fixed_successes)
                    print('\n')
                    fixed_successes = 0
                    fixed_reward = []
                if terminal:
                    # Update successes
                    fixed_successes += 1
                fixed_success_history.append(fixed_successes/(episode+1))
                break
            else:
                state = next_state
    return fixed_avg_rewards, fixed_success_history



if __name__ == '__main__':
    # Define the relevant parameters for the environment
    env_args = {"gridsize": 7,
                "predator_count": 4,
                "prey_count": 4}

    # Create the environment
    env = PredatorPreyEnv(env_args)

    # Define the relevant parameters for DQN
    dqn_args = {"epsilon": 0.3,
                "max_episodes": 1000,
                "learning_rate": 0.001,     # the learning rate for optimizer
                "step_size": 1,             # the step size for scheduler
                "gamma": 0.9,               # the gamma for scheduler
                "decay_rate": 0.95,         # decay rate for epsilon
                "active_agents": 4}

    # State and action state sizes for an individual predator
    # State includes predator position and nearest prey
    # Action includes the 9 defined predator actions defined in the problem
    state_size = 4
    action_size = 9

    # Parameter for how you would like to run the environment
    # fixed for non-learning predators
    # DQN for learning predators with DQN
    run_mode = "fixed"

    if run_mode == "DQN":
        # Run environment using DQN predators
        avg_rewards, success = run_dqn(env, state_size, action_size, dqn_args)
    elif run_mode == "fixed":
        # Run environment using fixed policy non learning predators
        avg_rewards, success = run_fixed(env)
