import numpy as np

CONST = np.sqrt(2 * np.pi)


def cdf_func(x, sum_val):
    return 0.5 + (sum_val / CONST) * np.exp(-1 * (x * x) / 2)


def CDF(normalized, max_i=100):
    cfd_vals = []
    sum_vals = []
    for x in normalized:
        sum_val = x
        value = x
        for i in np.arange(max_i) + 1:
            value = value * x * x / (2.0 * i + 1)
            sum_val += value
        cfd_vals.append(cdf_func(x, sum_val))
        sum_vals.append(sum_val)
    return cfd_vals, sum_vals


def inverse_cdf(transformed, sum_val):
    inverted = np.sqrt(-2 * np.log((transformed - 0.5) * CONST / sum_val))
    if transformed < 0.5:
        inverted *= -1
    return inverted


def unnormalise(transformed, mu, sig):
    return transformed * sig + mu
