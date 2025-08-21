import random
import functools
import torch
import numpy as np


def average(x_values) -> float | np.ndarray:
    beliefs = [x[0] for x in x_values]
    return sum(beliefs) / len(beliefs)

def weighted_avg(x_values) -> float | np.ndarray:
    sum_w = sum([x[1] for x in x_values])
    return sum([(x[0] * x[1]) / sum_w for x in x_values])

def prodop(x_values, normalise_w = False)  -> float | np.ndarray:
    outs = []
    if normalise_w:
        sum_w = sum([x[1] for x in x_values])
        x_values = [(x[0], x[1] / sum_w if sum_w > 0 else 0) for x in x_values]
    for index in range(len(x_values[0][0])):
        single_x_vals = [(float(x[index]), w) for x, w in x_values]
        outs.append(single_prodop(single_x_vals))
    return outs
    
def single_prodop(x_values) -> float:
    first_val = x_values[0]
    prod_val = first_val[0] ** first_val[1]
    neg_prod_val = (1 - first_val[0]) ** first_val[1]
    for x, w in x_values[1:]:
        prod_val *= x**w
        minus_x = 1 - x
        neg_prod_val *= minus_x ** w
    denom = (prod_val + neg_prod_val)
    if denom == 0:
        return 0.5
    else:
        return prod_val / (prod_val + neg_prod_val)

def make_belief_array(n_hypotheses: int, n_agents: int) -> list[list[np.ndarray | float]]:
    outs = [[np.random.random(size=(n_hypotheses,)) for _ in range(n_agents)]]
    outs[0] = [np.float32(x/sum(x)) for x in outs[0]]
    return outs

def average_sampler_11D(max_pooling_agents, min_pooling_agents=2, n_hypotheses=2, vary_num_agents=False, add_normalised=True):
    num_agents = random.randint(min_pooling_agents,max_pooling_agents) if vary_num_agents else max_pooling_agents
    outs = make_belief_array(n_hypotheses, num_agents)
    for _ in range(5):
        vals = [random.uniform(0, 1) for _ in range(num_agents)]
        outs.append(vals)
        if add_normalised:
            normalised_vals = [v / sum(vals) for v in vals]
            outs.append(normalised_vals)
    outs.append(average([[x] for x in outs[0]]))
    return outs

def normmeanprodop_sampler_11D(max_pooling_agents, min_pooling_agents=2, n_hypotheses=2, vary_num_agents=False, add_normalised=True):
    num_agents = random.randint(min_pooling_agents,max_pooling_agents) if vary_num_agents else max_pooling_agents
    outs = make_belief_array(n_hypotheses, num_agents)
    for _ in range(5):
        vals = [random.uniform(0, 1) for _ in range(num_agents)]
        outs.append(vals)
        if add_normalised:
            normalised_vals = [v / sum(vals) for v in vals]
            outs.append(normalised_vals)
    mergers_reliability = [outs[2][i] for i in range(num_agents)]
    product_out = prodop([[x, w] for x, w in zip(outs[0], mergers_reliability)], normalise_w=True)
    outs.append([x / sum(product_out) for x in product_out])
    return outs

def lambda_update(old_belief, new_belief, _lambda_new=0.90):
    return new_belief * _lambda_new + old_belief * (1 - _lambda_new)


class NeuralNetOp:
    def __init__(self, network) -> None:
        if isinstance(network, str):
            self.net = torch.load(network)
            if not hasattr(self.net, "aggregator"):
                self.net.aggregator = functools.partial(torch.sum, dim=0)
        elif isinstance(network, torch.nn.Module):
            self.net = network
        else:
            raise ValueError

    def __call__(self, x_values, dtype=torch.float32):
        with torch.no_grad():
            input_tens = torch.tensor(x_values, dtype=dtype).unsqueeze(0)
            output = self.net(input_tens)
            return output.item()
