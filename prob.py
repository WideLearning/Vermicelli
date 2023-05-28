import re
from copy import deepcopy

import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from torch import tensor
from tqdm import tqdm

torch.random.manual_seed(0)

PERP_LEFT = tensor([[0, -1], [1, 0]], dtype=torch.float)
MIRROR_X = tensor([[-1, 0], [0, 1]], dtype=torch.float)
MIRROR_Y = tensor([[1, 0], [0, -1]], dtype=torch.float)
ES = tensor([0.1, 0.1], dtype=torch.float)


def model():
    pass  # model


def check(z):
    conditioned = pyro.poutine.condition(model, data={"o1": z})
    trace = pyro.poutine.trace(conditioned).get_trace()
    return trace.log_prob_sum()


trace = pyro.poutine.trace(model).get_trace()
vars = {
    name: node["value"].requires_grad_(True)
    for name, node in trace.nodes.items()
    if "is_observed" in node and not node["is_observed"]
}
opt = torch.optim.Adam(vars.values(), lr=1.0)

step = 1e-9
plt.figure(figsize=(10, 10))
for i in tqdm(range(10000)):
    conditioned = pyro.poutine.condition(model, data=vars)
    trace = pyro.poutine.trace(conditioned).get_trace()
    loss = -trace.log_prob_sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    old_vars = deepcopy(vars)
    old_loss = loss
    for key in vars:
        vars[key].data += (
            step * torch.randn_like(vars[key]) * vars[key].data.abs().max()
        )
    conditioned = pyro.poutine.condition(model, data=vars)
    trace = pyro.poutine.trace(conditioned).get_trace()
    new_loss = -trace.log_prob_sum()
    log_transition = min(old_loss - new_loss, torch.zeros(()))
    # print(log_transition, step)
    if torch.rand(()) < log_transition.exp():
        # print("accept")
        loss = new_loss
        step *= 2
    else:
        vars = old_vars
        loss = old_loss
        step /= 2

    def res(name):
        parts = re.split(r"(\W+)", name)
        for i in range(len(parts)):
            if parts[i] in vars:
                parts[i] = repr(vars[parts[i]].detach())
        result = "".join(parts)
        return eval(result)

    def plot(*names):
        vecs = [res(v) for v in names]
        plt.plot([a[0] for a in vecs], [-a[1] for a in vecs], lw=0.3, c="gray", alpha=0.03)

    if i % 10 == 0:
        pass  # render

    if (i & (i + 1)) == 0:
        plt.xlim(-300, 300)
        plt.ylim(-200, 400)
        plt.savefig(f"{i}.png")
