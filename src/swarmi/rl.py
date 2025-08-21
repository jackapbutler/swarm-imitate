import collections
import random

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from swarmi import net
from swarmi.game import (
    agent_init,
    generate_truth_flipping,
    get_converged_reward,
    play_game_turn,
)


def reinforce(
    initial_policy,
    optimiser,
    gamma,
    batch_size,
    num_agent_func,
    max_turns,
    min_turns,
    min_poolers,
    max_poolers,
    num_pools,
    reward_func,
    pool_initialiser=agent_init,
    n_truth_flips: int = 0,
    update_func=None,
    preprocess_func=None,
    using_nn_pooling=False,
    n_batches=1,
    num_checkpoints=1,
    trial_name='test'
):
    final_rewards = []
    fraction_success_end = []
    fixed_std = torch.tensor([0.01])

    def temp_rl_policy(x_values):
        input_tens = torch.tensor(x_values).unsqueeze(0)
        mean_out = initial_policy(input_tens)
        dist = torch.distributions.Normal(mean_out, fixed_std)
        output = dist.sample()
        return output.item(), dist.log_prob(output)

    for _step in tqdm(range(n_batches), desc="Batches"):
        losses = []

        for _ in range(batch_size):
            n_agents = num_agent_func()
            possible_truths = [0, 1]
            truth = random.choice(possible_truths)
            belief_pool_aftermerge = pool_initialiser(n_agents)
            all_agents = list(belief_pool_aftermerge.keys())

            if n_truth_flips == 0:
                truth_values = [truth] * max_turns
            else:
                truth_values = generate_truth_flipping(
                    truth, n_truth_flips, max_turns
                )

            beliefs = [belief_pool_aftermerge]
            rewards = []
            log_probs = []
            for turn_num, truth in enumerate(truth_values):
                (
                    belief_pool_afterevidence,
                    belief_pool_aftermerge,
                    turn_log_probs,
                    turn_rewards,
                ) = play_game_turn(
                    belief_pool_aftermerge,
                    temp_rl_policy,
                    min_poolers,
                    max_poolers,
                    num_pools,
                    possible_truths,
                    truth,
                    reward_func,
                    return_rl_actions=True,
                    update_func=update_func,
                    preprocess_func=preprocess_func,
                    using_nn_pooling=using_nn_pooling,
                )

                rewards.extend(turn_rewards)
                log_probs.extend(turn_log_probs)
                beliefs.extend(
                    [belief_pool_afterevidence, belief_pool_aftermerge]
                )

                fraction_success = get_converged_reward(
                    belief_pool_aftermerge, all_agents, possible_truths, truth
                )
                if 0.99 < fraction_success and min_turns < turn_num:
                    break

            ## Compute the discounted returns at each timestep, as
            ## the sum of the gamma-discounted return at time t (G_t) + the reward at time t
            n_steps = len(rewards)
            returns = collections.deque(maxlen=n_steps)
            for t in range(n_steps)[::-1]:
                disc_return_t = returns[0] if returns else 0
                returns.appendleft(gamma * disc_return_t + rewards[t])

            # Loss calculation considers the prob(choosing that action)
            ## so we prefer outputing actions that are correct
            policy_losses = []
            for log_prob, disc_return in zip(log_probs, returns):
                policy_losses.append(-log_prob * disc_return)

            losses.append(torch.stack(policy_losses).sum())

        reduced_batch_loss = torch.mean(torch.stack(losses))

        # Gradient descent
        optimiser.zero_grad()
        reduced_batch_loss.backward()
        optimiser.step()
        
        if _step % (n_batches // num_checkpoints) == 0:
            torch.save(initial_policy, f"./models/{trial_name}_{_step}.pt")

        # Metrics
        final_rewards.append(sum(rewards) / len(rewards))
        fraction_success_end.append(
            get_converged_reward(
                belief_pool_aftermerge, all_agents, possible_truths, truth
            )
        )

    return final_rewards, fraction_success_end


def plot_rewards(
    trial_final_rewards,
    trial_fraction_success_end,
    interval_size=100,
    name=" ",
    plot_converged=False,
    save=False,
    save_raw=False
):
    plt.figure(figsize=(18, 6))
    for i, final_rewards in enumerate(trial_final_rewards):
        grouped_rewards = []
        for x in range(0, len(final_rewards), interval_size):
            grouped_rewards.append(
                sum(final_rewards[x : x + interval_size]) / interval_size
            )
        plt.plot(grouped_rewards, label=f"Trial {i}")
        if save_raw:
            with open(f"./plots/{name}_rewards_{i}.txt", "a+") as f:
                f.write(str(grouped_rewards))
    plt.title(
        f"Avg.Reward over {interval_size}-sized intervals of episodes - {name}"
    )
    plt.ylabel("Avg.Reward")
    plt.xlabel("Interval Index")
    plt.legend()
    if save:
        plt.savefig(f"./plots/{name}_rewards.png")
    else:
        plt.show()

    if plot_converged:
        plt.figure(figsize=(18, 6))
        for i, fraction_success_end in enumerate(trial_fraction_success_end):
            grouped_successes = []
            for x in range(0, len(fraction_success_end), interval_size):
                grouped_successes.append(
                    sum(fraction_success_end[x : x + interval_size])
                    / interval_size
                )
            if save_raw:
                with open(f"./plots/{name}_rewards_{i}.txt", "a+") as f:
                    f.write(str(grouped_rewards))
        plt.plot(grouped_successes)
        plt.title(
            f"Avg.Wins over {interval_size}-sized intervals of episodes - {name}"
        )
        plt.ylabel("Avg.Fraction Success")
        plt.xlabel("Interval Index")
        plt.legend()
        if save:
            plt.savefig(f"./plots/{name}_converged.png")
        else:
            plt.show()
