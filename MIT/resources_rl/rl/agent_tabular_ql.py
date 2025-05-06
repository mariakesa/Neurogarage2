"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    benchmark=np.random.uniform()
    if benchmark>epsilon:
        best_action=q_func[state_1,state_2,0,0]
        action_index=0
        object_index=0
        for a in range(NUM_ACTIONS):
            for o in range(NUM_OBJECTS):
                action_value=q_func[state_1,state_2,a,o]
                if action_value>=best_action:
                    action_index=a
                    object_index=o
                    best_action=action_value
    else:
        action_index=np.random.randint(0,NUM_ACTIONS)
        object_index=np.random.randint(0,NUM_OBJECTS)
    return (action_index, object_index)


# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    if not terminal:
        next_q=np.max(q_func[next_state_1,next_state_2,:,:].flatten())
        q_func[current_state_1, current_state_2, action_index,
            object_index] = (1-ALPHA)*q_func[current_state_1, current_state_2, action_index,
            object_index]+ALPHA*(reward+GAMMA*next_q)
    else:
        q_func[current_state_1, current_state_2, action_index,
            object_index]=(1-ALPHA)*q_func[current_state_1, current_state_2, action_index,
            object_index]+ALPHA*(reward)

    return None  # This function shouldn't return anything


# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0
    # initialize for each episode
    # TODO Your code here
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    #state_1,state_2=framework.rooms_desc_map[current_room_desc],framework.rooms_desc_map[current_quest_desc]
    state_1,state_2=dict_room_desc[current_room_desc],dict_quest_desc[current_quest_desc]
    counter=0
    while not terminal:
        # Choose next action and execute
        # TODO Your code here
        action_index, object_index = epsilon_greedy(state_1, state_2, q_func, epsilon)
        next_room_desc, next_quest_desc, reward, terminal=framework.step_game(current_room_desc,current_quest_desc,action_index,object_index)
        current_state_1, current_state_2=state_1, state_2
        next_state_1,next_state_2=dict_room_desc[next_room_desc],dict_quest_desc[next_quest_desc]
        #next_state_1, next_state_2 = framework.rooms_desc_map[next_room_desc],framework.rooms_desc_map[next_quest_desc]
        if for_training:
            # update Q-function.
            # TODO Your code here
            tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal)

        if not for_training:
            # update reward
            epi_reward+=(GAMMA**counter)*(reward)
            counter+=1

        state_1,state_2=next_state_1,next_state_2

    if not for_training:
        return epi_reward


# pragma: coderesponse end


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))

'''
def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test

def run():
    """Returns array of test reward per epoch for one run and reports convergence epoch and avg reward after convergence"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=100)

    ewma_rewards = []
    convergence_epoch = None
    stable_epochs_required = 5  # How many stable epochs we require to say it converged
    delta = 0.01  # Allowed fluctuation in EWMA to consider "stable"

    for epoch in pbar:
        test_reward = run_epoch()
        single_run_epoch_rewards_test.append(test_reward)

        ewma = utils.ewma(single_run_epoch_rewards_test)
        ewma_rewards.append(ewma)

        pbar.set_description(
            f"Epoch {epoch} | Avg reward: {np.mean(single_run_epoch_rewards_test):.4f} | EWMA: {ewma:.4f}")

        # Check convergence
        if epoch >= stable_epochs_required:
            recent_ewmas = ewma_rewards[-stable_epochs_required:]
            if max(recent_ewmas) - min(recent_ewmas) < delta and convergence_epoch is None:
                convergence_epoch = epoch - stable_epochs_required + 1  # First epoch of stable range

    # Average reward after convergence
    if convergence_epoch is not None:
        avg_post_convergence_reward = np.mean(single_run_epoch_rewards_test[convergence_epoch:])
        print(f"\nConverged at epoch {convergence_epoch}")
        print(f"Average episodic reward after convergence: {avg_post_convergence_reward:.4f}")
    else:
        print("\nDid not converge within given epochs.")

    return single_run_epoch_rewards_test



def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    for _ in range(NUM_EPOCHS):
        single_run_epoch_rewards_test.append(run_epoch())
    return single_run_epoch_rewards_test


def evaluate_average_post_convergence(num_runs=10, convergence_epoch=14):
    all_runs = []
    for i in range(num_runs):
        print(f"Running training iteration {i+1}/{num_runs}")
        rewards = run()
        all_runs.append(rewards)

    all_runs = np.array(all_runs)  # Shape: (num_runs, NUM_EPOCHS)

    # Take only epochs from convergence onward
    post_convergence_rewards = all_runs[:, convergence_epoch:]
    avg_post_convergence = np.mean(post_convergence_rewards)

    print(f"\nConvergence epoch: {convergence_epoch}")
    print(f"Average test reward after convergence (epoch ≥ {convergence_epoch}): {avg_post_convergence:.4f}")
    return avg_post_convergence

'''

import matplotlib.pyplot as plt
import numpy as np

def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    for _ in range(NUM_EPOCHS):
        single_run_epoch_rewards_test.append(run_epoch())
    return single_run_epoch_rewards_test


def detect_convergence_and_plot(rewards, start=100, end=200):
    rewards = np.array(rewards)
    reference_window = rewards[start:end+1]
    reward_mean = np.mean(reference_window)
    reward_min = np.min(reference_window)
    reward_max = np.max(reference_window)

    # Find first epoch where reward enters [min, max] and stays there
    for i in range(len(rewards)):
        if reward_min <= rewards[i] <= reward_max:
            if np.all((reward_min <= rewards[i:i+10]) & (rewards[i:i+10] <= reward_max)):
                convergence_epoch = i
                break
    else:
        convergence_epoch = None

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Test Reward", color="blue")
    plt.hlines(reward_mean, 0, len(rewards)-1, colors='green', linestyles='dashed', label='Mean (100–200)')
    plt.hlines([reward_min, reward_max], 0, len(rewards)-1, colors='red', linestyles='dotted', label='Min/Max (100–200)')
    if convergence_epoch is not None:
        plt.axvline(convergence_epoch, color='purple', linestyle='--', label=f'Converged at {convergence_epoch}')
    plt.xlabel("Epoch")
    plt.ylabel("Test Reward")
    plt.legend()
    plt.title("Convergence Detection via Reward Stability")
    plt.grid(True)
    plt.show()

    if convergence_epoch is not None:
        print(f"\n✅ Converged at epoch: {convergence_epoch}")
        avg_after_convergence = np.mean(rewards[convergence_epoch:])
        print(f"Average reward after convergence: {avg_after_convergence:.4f}")
        return convergence_epoch, avg_after_convergence
    else:
        print("⚠️ Did not detect convergence.")
        return None, None


if __name__ == '__main__':
    
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS
    '''
    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()'''
    rewards = run()
    detect_convergence_and_plot(rewards)
    #evaluate_average_post_convergence(num_runs=10, convergence_epoch=14)