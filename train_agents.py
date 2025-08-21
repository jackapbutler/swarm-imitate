import functools
import random
import argparse
import torch
from swarmi import game, net, rl

GAMMA = 0.95
MAX_TURNS = 50  # 50
MIN_TURNS = MAX_TURNS  # 10
MAX_POOLERS = 20
MIN_POOLERS = 10
N_POOLS = 1
N_AGENTS = 100
AGENT_FUNC = lambda: N_AGENTS
INITIALISATION = functools.partial(
    game.agent_init,
    belief_init_func=lambda: random.uniform(0, 1),
    evidence_init_func=lambda: random.uniform(0, 0.5),
    noise_init_func=lambda: random.uniform(0.5, 1),
    pooling_init_func=lambda: random.uniform(0, 1),
    communication_init_func=lambda: random.uniform(0, 0.1),
    zealot_init_func=lambda: random.uniform(0, 0.5),
)
NN_HIDDEN_SIZE = int(
    torch.ceil(torch.exp(torch.tensor(min(MAX_POOLERS ** (0.5), 6)))).item()
)
REWARD_FUNC = functools.partial(game.get_distance_reward, exp=1)
N_TRUTH_FLIPS = 0
NUM_CHECKPOINTS = 10

if __name__ == '__main__':
    # Config
    parser = argparse.ArgumentParser(description='A CLI for training binary opinion pooling operators.')
    parser.add_argument('--name', type=str, help='Print a greeting message.', required=True)
    parser.add_argument('--imitation', type=str, help='The path to the imitation neural network.', required=False)
    parser.add_argument('--n_episodes', type=int, help='The number of episodes to train for.', required=True)
    parser.add_argument('--batch_size', type=int, help='The batch size to use.', required=False, default=1)
    parser.add_argument('--n_trials', type=int, help='The number of trials to run.', required=False, default=2)
    parser.add_argument('--lr', type=float, help='The learning rate to use.', required=False, default=3e-6)
    args = parser.parse_args()
    
    RETRAIN_FROM_IMITATE = False
    if args.imitation:
        IMITATED_MODEL_PATH = args.imitation
        RETRAIN_FROM_IMITATE = True
    NAME = args.name
    BATCH_SIZE = args.batch_size
    N_BATCHES = int(round(args.n_episodes / BATCH_SIZE))
    N_TRIALS = args.n_trials
    LR = args.lr

    # Run
    trial_rewards = []
    trial_successes = []
    for n_trial in range(N_TRIALS):
        print(f"trial {n_trial+1}/{N_TRIALS}")
        trial_name = f"{NAME}_{n_trial}"
        global_rl_policy = (
            torch.load(IMITATED_MODEL_PATH)
            if RETRAIN_FROM_IMITATE
            else net.Net(NN_HIDDEN_SIZE, 1, agg_fn=torch.mean)
        )
        optimiser = torch.optim.Adam(global_rl_policy.parameters(), lr=LR)

        final_rewards, fraction_success_end = rl.reinforce(
            global_rl_policy,
            optimiser,
            GAMMA,
            BATCH_SIZE,
            AGENT_FUNC,
            MAX_TURNS,
            MIN_TURNS,
            MIN_POOLERS,
            MAX_POOLERS,
            N_POOLS,
            REWARD_FUNC,
            INITIALISATION,
            N_TRUTH_FLIPS,
            update_func=None,
            preprocess_func=None,
            using_nn_pooling=True,
            n_batches=N_BATCHES,
            num_checkpoints=NUM_CHECKPOINTS,
            trial_name=trial_name,
        )

        trial_rewards.append(final_rewards)
        trial_successes.append(fraction_success_end)

        torch.save(global_rl_policy, f"./models/{trial_name}_done.pt")
    
    rl.plot_rewards(
        trial_rewards,
        trial_successes,
        name=NAME + "_100k",
        interval_size=100_000,
        save=True,
        save_raw=True
    )
