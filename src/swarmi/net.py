import functools
import torch


def normalise(value, min_value=0, max_value=1):
    return 2 * (value - min_value) / (max_value - min_value) - 1


class MultiNet(torch.nn.Module):
    def __init__(self, Nhid, Nout, agg_fn = torch.mean):
        super(MultiNet, self).__init__()
        self.lin1 = initialise_encoder(Nhid)
        self.lin2 = initialise_decoder(Nhid)
        self.out = torch.nn.LazyLinear(Nout)
        self.aggregator = functools.partial(agg_fn, dim=0)

    def forward(self, inputs):
        outputs = []

        for sample in inputs:
            projected = self.lin1(sample)
            collapsed_projected = self.aggregator(projected)
            decoded = self.lin2(collapsed_projected)
            final_processed = self.out(decoded)
            final_out = torch.softmax(final_processed, dim=0)
            outputs.append(final_out)

        return torch.stack(outputs)

class Net(torch.nn.Module):
    def __init__(self, Nhid, Nout, agg_fn = torch.mean):
        super(Net, self).__init__()
        self.lin1 = initialise_encoder(Nhid)
        self.lin2 = initialise_decoder(Nhid)
        self.out = torch.nn.LazyLinear(Nout)
        self.aggregator = functools.partial(agg_fn, dim=0)

    def forward(self, inputs):
        outputs = []

        for sample in inputs:
            projected = self.lin1(sample)
            collapsed_projected = self.aggregator(projected)
            decoded = self.lin2(collapsed_projected)
            final_processed = self.out(decoded)
            final_out = torch.sigmoid(final_processed)
            outputs.append(final_out)

        return torch.stack(outputs)


def initialise_encoder(n_nodes):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(n_nodes),
        torch.nn.ReLU(),
        torch.nn.LazyLinear(n_nodes),
        torch.nn.ReLU(),
        torch.nn.LazyLinear(n_nodes),
    )


def initialise_decoder(n_nodes):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(n_nodes),
        torch.nn.ReLU(),
        torch.nn.LazyLinear(n_nodes),
        torch.nn.ReLU(),
    )
