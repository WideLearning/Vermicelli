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
    
    OX_len = pyro.sample('OX_len', dist.Normal(90.0, 1.0))
    M = pyro.sample('M', dist.Normal(tensor([10, 55], dtype=torch.float), tensor([2, 3], dtype=torch.float)))
    
    
    n1 = pyro.sample('n1', dist.Normal(tensor([0, 20], dtype=torch.float), tensor([5, 5], dtype=torch.float)))
    n2 = pyro.sample('n2', dist.Normal(tensor([40, 0], dtype=torch.float), tensor([4, 4], dtype=torch.float)))
    n3 = pyro.sample('n3', dist.Normal(tensor([30, -20], dtype=torch.float), tensor([4, 4], dtype=torch.float)))
    
    
    Mmb = pyro.sample('Mmb', dist.Normal(0.2 * M, tensor([2, 2], dtype=torch.float)))
    Mma = pyro.sample('Mma', dist.Normal(0.8 * PERP_LEFT @ M, tensor([5, 5], dtype=torch.float)))
    
    
    Xc3 = pyro.sample('Xc3', dist.Normal(tensor([155, 0], dtype=torch.float), tensor([10, 20], dtype=torch.float)))
    Xc2 = pyro.sample('Xc2', dist.Normal(tensor([122, 120], dtype=torch.float), tensor([10, 10], dtype=torch.float)))
    Xc1 = pyro.sample('Xc1', dist.Normal(tensor([35, 220], dtype=torch.float), tensor([15, 10], dtype=torch.float)))
    
    
    Xe = pyro.sample('Xe', dist.Normal(tensor([80, 0], dtype=torch.float), tensor([1, 1], dtype=torch.float)))
    
    
    eb1 = pyro.sample('eb1', dist.Normal(tensor([-40, -40], dtype=torch.float), tensor([5, 5], dtype=torch.float)))
    eb2 = pyro.sample('eb2', dist.Normal(tensor([0, -50], dtype=torch.float), tensor([5, 5], dtype=torch.float)))
    eb3 = pyro.sample('eb3', dist.Normal(tensor([40, -40], dtype=torch.float), tensor([5, 5], dtype=torch.float)))
    
    
    ee1 = pyro.sample('ee1', dist.Normal(tensor([-30, 10], dtype=torch.float), tensor([2, 2], dtype=torch.float)))
    ee2 = pyro.sample('ee2', dist.Normal(tensor([-20, -20], dtype=torch.float), tensor([4, 4], dtype=torch.float)))
    ee3 = pyro.sample('ee3', dist.Normal(tensor([26, -8], dtype=torch.float), tensor([4, 4], dtype=torch.float)))
    ee4 = pyro.sample('ee4', dist.Normal(tensor([30, 7], dtype=torch.float), tensor([5, 3], dtype=torch.float)))
    
    
    h0a = pyro.sample('h0a', dist.Normal(tensor([0, -290], dtype=torch.float), tensor([10, 40], dtype=torch.float)))
    h0b = pyro.sample('h0b', dist.Normal(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_Y @ Xc2), tensor([20, 20], dtype=torch.float)))
    h0be = pyro.sample('h0be', dist.Normal(0, tensor([30, 30], dtype=torch.float)))
    h0c = pyro.sample('h0c', dist.Normal(1.1 * ((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc3), tensor([20, 20], dtype=torch.float)))
    h0ce = pyro.sample('h0ce', dist.Normal(0, tensor([30, 30], dtype=torch.float)))
    h0d = pyro.sample('h0d', dist.Normal(1.5 * ((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc2), tensor([50, 50], dtype=torch.float)))
    h0de = pyro.sample('h0de', dist.Normal(0, tensor([60, 60], dtype=torch.float)))
    h1a = pyro.sample('h1a', dist.Normal(tensor([0, -290], dtype=torch.float), tensor([10, 40], dtype=torch.float)))
    h1b = pyro.sample('h1b', dist.Normal(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_Y @ Xc2), tensor([20, 20], dtype=torch.float)))
    h1be = pyro.sample('h1be', dist.Normal(0, tensor([30, 30], dtype=torch.float)))
    h1c = pyro.sample('h1c', dist.Normal(1.1 * ((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc3), tensor([20, 20], dtype=torch.float)))
    h1ce = pyro.sample('h1ce', dist.Normal(0, tensor([30, 30], dtype=torch.float)))
    h1d = pyro.sample('h1d', dist.Normal(1.5 * ((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc2), tensor([50, 50], dtype=torch.float)))
    h1de = pyro.sample('h1de', dist.Normal(0, tensor([60, 60], dtype=torch.float)))
    


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
        plot("n3", "n2", "n1", "(MIRROR_X @ n2)", "(MIRROR_X @ n3)")
        plot("(M - Mma)", "(M - Mmb)", "(M + Mma)", "(M + Mmb)", "(M - Mma)", "(M + Mma)")
        plot("(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + eb1)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + eb2)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + eb3)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + eb1)")
        plot("(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ eb1)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ eb2)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ eb3)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ eb1)")
        plot("((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc3)", "((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc2)", "((-OX_len * tensor([0, 1], dtype=torch.float)) + Xc1)", "((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xc1)", "((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xc2)", "((-OX_len * tensor([0, 1], dtype=torch.float)) - Xc3)")
        plot("(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + ee1)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + ee2)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + ee3)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + ee4)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + Xe) + ee1)")
        plot("(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ ee1)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ ee2)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ ee3)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ ee4)", "(((-OX_len * tensor([0, 1], dtype=torch.float)) + MIRROR_X @ Xe) + MIRROR_X @ ee1)")
        plot("h0a", "h0b", "h0c", "h0d")
        plot("h0a", "(h0be + MIRROR_X @ h0b)", "(h0ce + MIRROR_X @ h0c)", "(h0de + MIRROR_X @ h0d)")
        plot("h1a", "h1b", "h1c", "h1d")
        plot("h1a", "(h1be + MIRROR_X @ h1b)", "(h1ce + MIRROR_X @ h1c)", "(h1de + MIRROR_X @ h1d)")
        

    if (i & (i + 1)) == 0:
        plt.xlim(-300, 300)
        plt.ylim(-200, 400)
        plt.savefig(f"{i}.png")
