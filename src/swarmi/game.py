import random

import numpy as np

from swarmi import net

randu = lambda: random.uniform(0, 1)
small_randu = lambda: random.uniform(0, 0.2)


def bayes_update(x, E, alpha):
    delta_E = 1 - alpha if E == 1 else alpha
    denom = (delta_E * x + (1 - delta_E) * (1 - x))
    if denom == 0:
        return 0.50
    else:
        return (delta_E * x) / denom


def agent_init(
    n_agents: int,
    belief_init_func=lambda: 0.50,
    evidence_init_func=randu,
    noise_init_func=randu,
    pooling_init_func=randu,
    communication_init_func=small_randu,
    zealot_init_func=small_randu,
):
    state_fns = [
        belief_init_func,
        evidence_init_func,
        noise_init_func,
        pooling_init_func,
        communication_init_func,
        zealot_init_func,
    ]
    return {
        a: tuple(round(f(), 4) for f in state_fns) for a in range(n_agents)
    }


def get_distance_reward(
    beliefs, agents, possible_truths, truth, sign=-1, exp=2
):
    """
    Rewarded by how close these agents are to the truth
        - continuous MSE to truth of each agent
    """
    distances = [sign * (abs(beliefs[a][0] - truth) ** exp) for a in agents]
    return sum(distances) / len(agents)


def get_converged_reward(
    beliefs, agents, possible_truths, truth, closeness_threshold=0.5
):
    """
    Rewarded by how close these agents are to the truth
        - binary or fraction of agents which round to truth
    """
    # find distance to all beliefs
    distances = [
        [abs(beliefs[a][0] - belief) for belief in possible_truths]
        for a in agents
    ]
    # get index of actual truth
    truth_index = possible_truths.index(truth)
    # show that closest one is to the truth and within threshold
    chosen_beliefs = [
        a
        for a in distances
        if min(a) == a[truth_index] and min(a) < closeness_threshold
    ]
    return len(chosen_beliefs) / len(agents)

def receive_evidence(agent, truth_value) -> int | None:
    evidence_v = None
    if randu() < agent[1]:
        evidence_v = truth_value
        # get correct if not noise < P(correct evidence)
        if not randu() < agent[2]:
            evidence_v = 1 if truth_value == 0 else 0
    new_belief = None
    if evidence_v is not None:
        # NOTE hardcoded for now (could be learned)
        new_belief = bayes_update(agent[0], evidence_v, agent[5])
    return new_belief

def get_pooling_groups(frozen_pool: dict, all_agents: list[int], min_poolers: int, max_poolers: int, num_pools: int) -> list[list[int]]:
    weightings = np.array([frozen_pool[a][3] for a in frozen_pool])
    normed_weightings = weightings / np.sum(weightings)
    num_poolers = random.randint(min_poolers, max_poolers)
    all_mergers = np.random.choice(all_agents, num_poolers * num_pools, p=normed_weightings, replace=False)
    np.random.shuffle(all_mergers)
    merger_groups = [all_mergers[i : i + num_poolers] for i in range(0, len(all_mergers), num_poolers)]
    return merger_groups

def represent_poolers(beliefs_to_merge, preprocess_func, using_nn_pooling) -> list:
    if using_nn_pooling:
        merged_representations = [
            [add_communication_noise(b[0], b[4]), *b[1:]] for b in beliefs_to_merge
        ]
        merged_representations = [
            [net.normalise(x) for x in b] for b in merged_representations
        ]
    else:
        # standard policies cannot see all 5D features
        mergers_reliability = [
            preprocess_func(b[1:]) if preprocess_func else b[2]
            for b in beliefs_to_merge
        ]
        merged_representations = [
            [add_communication_noise(b[0], b[4]), w]
            for b, w in zip(beliefs_to_merge, mergers_reliability)
        ]
    return merged_representations

def add_communication_noise(belief: float, noise: float) -> float:
    sampled_noise = random.uniform(-1 * noise, noise)
    noisy_belief = belief + sampled_noise
    return 0 if noisy_belief < 0 else 1 if noisy_belief > 1 else noisy_belief

def update_beliefs(belief_pool_aftermerge, mergers, merged_belief, update_func) -> dict:
    for index in mergers:
        # update
        agent = belief_pool_aftermerge[index]
        if update_func:
            merged_belief = update_func(agent[0], merged_belief)
        belief_pool_aftermerge[index] = (merged_belief, *agent[1:])
    return belief_pool_aftermerge

def evidence_pooling(agents: dict[str, tuple], truth_value: int) -> dict[str, tuple]:
    # merge with sporadic evidence
    for index, agent in agents.items():
        # get evidence if location < P(getting evidence)
        new_belief = receive_evidence(agent, truth_value)
        if new_belief is not None:
            agents[index] = (new_belief, *agent[1:])
    return agents

def play_game_turn(
    belief_pool: dict,
    policy_strategy,
    min_poolers: int,
    max_poolers: int,
    num_pools: int,
    possible_truths: list,
    truth_value: int,
    reward_func,
    return_rl_actions: bool = False,
    update_func=None,
    preprocess_func=None,
    using_nn_pooling: bool = False,
):
    # merge with sporadic evidence
    belief_pool = evidence_pooling(belief_pool, truth_value)
    
    # pool k agents together, n times
    turn_rewards = []
    turn_log_probs = []
    all_agents = list(belief_pool.keys())
    merger_groups = get_pooling_groups(belief_pool, all_agents, min_poolers, max_poolers, num_pools)
    belief_pool_aftermerge = belief_pool.copy()
    for mergers in merger_groups:
        # represent poolers for operators (normalisation, collapsing, etc)
        beliefs_to_merge = [belief_pool_aftermerge[a] for a in mergers]
        merged_representations = represent_poolers(beliefs_to_merge, preprocess_func, using_nn_pooling)

        # act
        log_probs_merge = None
        merged_belief = policy_strategy(merged_representations)
        if hasattr(merged_belief, "__len__"):
            merged_belief, log_probs_merge = merged_belief
        belief_pool_aftermerge = update_beliefs(
            belief_pool_aftermerge, mergers, merged_belief, update_func
        )

        # reward
        turn_reward = reward_func(
            belief_pool_aftermerge, all_agents, possible_truths, truth_value
        )
        turn_log_probs.append(log_probs_merge)
        turn_rewards.append(turn_reward)

    if return_rl_actions:
        return (
            belief_pool,
            belief_pool_aftermerge,
            turn_log_probs,
            turn_rewards,
        )
    else:
        return belief_pool, belief_pool_aftermerge


def generate_truth_flipping(
    current_truth: int, n_truth_flips: int, steps: int
):
    truth_values = []
    n_truth_segments = n_truth_flips + 1
    change_increment = steps / n_truth_segments
    for step in range(1, steps + 1):
        if change_increment < step:
            current_truth = 1 if current_truth == 0 else 0
            change_increment += steps / n_truth_segments
        truth_values.append(current_truth)
    return truth_values


def evaluate(
    policies,
    n_repeats,
    num_agent_func,
    max_iters,
    min_poolers,
    max_poolers,
    num_pools,
    pool_initialiser=agent_init,
    evaluation_func=get_converged_reward,
    n_truth_flips=0,
    update_func=None,
    preprocess_func=None,
    using_nn_pooling=False,
):
    scores = {p: [] for p in policies}
    belief_history = {p: {} for p in policies}

    for i in range(n_repeats):
        n_agents = num_agent_func()
        possible_truths = [0, 1]
        truth = random.choice(possible_truths)
        shared_initial_belief_pool = pool_initialiser(n_agents)
        all_agents = list(shared_initial_belief_pool.keys())

        if n_truth_flips == 0:
            truth_values = [truth] * max_iters
        else:
            truth_values = generate_truth_flipping(
                truth, n_truth_flips, max_iters
            )

        for p_name, policy_strategy in policies.items():
            # ensures comparability between policies
            belief_pool = shared_initial_belief_pool.copy()
            belief_history[p_name][i] = []
            timestep_eval = []

            for iteration, truth in enumerate(truth_values):
                if iteration == 0:
                    belief_history[p_name][i].append(
                        list(belief_pool.values())
                    )
                belief_pool = play_game_turn(
                    belief_pool,
                    policy_strategy,
                    min_poolers,
                    max_poolers,
                    num_pools,
                    possible_truths,
                    truth,
                    evaluation_func,
                    update_func=update_func,
                    preprocess_func=preprocess_func,
                    using_nn_pooling=using_nn_pooling,
                )

                # Unpacking for two modes
                ## reinforcement learning requires other outputs
                ## evaluation requires both belief pool states
                if len(belief_pool) == 4:
                    (
                        belief_pool_afterevidence,
                        belief_pool_aftermerge,
                        _,
                        _,
                    ) = belief_pool
                else:
                    (
                        belief_pool_afterevidence,
                        belief_pool_aftermerge,
                    ) = belief_pool

                for pooling_stage in [
                    belief_pool_afterevidence,
                    belief_pool_aftermerge,
                ]:
                    belief_history[p_name][i].append(
                        list(pooling_stage.values())
                    )

                belief_pool = belief_pool_aftermerge
                converged_agents_now = evaluation_func(
                    belief_pool,
                    all_agents,
                    possible_truths,
                    truth,
                )
                timestep_eval.append(converged_agents_now)

            scores[p_name].append(timestep_eval)

    return scores, belief_history
