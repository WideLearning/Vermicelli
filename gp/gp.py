import torch as t
from beartype import beartype as typed
from beartype.typing import Callable, Literal
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from scipy import stats  # type: ignore
from torch import Tensor as TT
from tqdm import tqdm  # type: ignore


class GP(t.nn.Module):
    @typed
    def __init__(
        self,
        n_random: int = 100,
        kernel: Literal["gauss", "laplace", "cauchy"] = "gauss",
        kernel_scale: float = 1.0,
    ):
        super().__init__()

        # this one probably should be Gaussian in any case
        self.theta = t.nn.Parameter(t.randn(n_random))

        # this can be Gaussian / Cauchy / Laplace
        eps = 1 / (n_random + 1)
        ppf = t.linspace(0 + eps, 1 - eps, n_random)
        periods_data = {
            "gauss": stats.norm,
            "laplace": stats.cauchy,
            "cauchy": stats.laplace,
        }[kernel].ppf(ppf)
        self.periods = t.tensor(periods_data, dtype=t.float32)
        self.shifts = (t.randint(0, 2, (n_random,))) * t.pi / 2
        self.sqrt2 = 2**0.5

        self.kernel_scale = kernel_scale

    @typed
    def get_w(self, x: Float[TT, "batch"]) -> Float[TT, "batch n_random"]:
        # TODO why kernel correlations look a bit different near zero?
        k = self.periods[None, :]
        b = self.shifts[None, :]
        return self.sqrt2 * t.cos(k * x[:, None] / self.kernel_scale + b)

    @typed
    def reinit(self) -> None:
        self.theta.data = t.randn(self.n_random)

    @typed
    def forward(self, x: Float[TT, "batch"]) -> Float[TT, "batch"]:
        w = self.get_w(x)
        return w @ self.theta


@typed
def fit(
    model: t.nn.Module,
    x_train: Float[TT, "batch"],
    y_train: Float[TT, "batch"],
    lr: float = 0.01,
    epochs: int = 10,
) -> t.nn.Module:
    optim = t.optim.Adam(model.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        y_pred = model(x_train)
        loss = t.nn.functional.mse_loss(y_pred, y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.item())
    return model


@typed
def smooth_sample(
    model: t.nn.Module,
    x_train: Float[TT, "data ..."],
    y_train: Float[TT, "data ..."],
    x_test: Float[TT, "data ..."],
    force_schedule: list[Callable[[float], Float[TT, "..."]]],
    T_schedule: Callable[[float], float],
    n_samples: int = 10,
    lr: float = 0.1,
) -> Float[TT, "history ..."]:
    model = fit(model, x_train, y_train, epochs=10)

    def get_theta() -> list[Float[TT, "..."]]:
        return [p.data.clone() for p in model.parameters()]

    def set_theta(theta: list[Float[TT, "..."]]) -> None:
        assert len(theta) == len(list(model.parameters()))
        for p, v in zip(model.parameters(), theta):
            assert p.shape == v.shape
            p.data = v.clone()

    def logposterior(T: float) -> Float[TT, ""]:
        result = t.tensor(0.0)
        with t.no_grad():
            for w in model.parameters():
                result -= 0.5 * t.sum(w**2)
            result -= t.nn.functional.mse_loss(model(x_train), y_train)
        return result / T

    def get_force(time: float) -> list[Float[TT, "..."]]:
        with t.no_grad():
            return [gp(time) for gp in force_schedule]

    def add(
        x: list[Float[TT, "..."]], y: list[Float[TT, "..."]], k: float | Float[TT, ""]
    ) -> list[Float[TT, "..."]]:
        with t.no_grad():
            assert len(x) == len(y)
            return [x[i] + k * y[i] for i in range(len(x))]

    theta = get_theta()
    set_theta(theta)

    samples: list[Float[TT, "..."]] = []
    for time in range(n_samples):
        theta_old = get_theta()
        T = T_schedule(time)
        F = get_force(time)
        lp_old = logposterior(T)

        lef = t.tensor(1e-9)
        rig = t.tensor(1.0)
        for _ in range(10):
            mid = t.sqrt(lef * rig)
            theta_new = add(theta_old, F, mid * lr)
            set_theta(theta_new)
            lp_new = logposterior(T)
            need = t.exp(lp_new - lp_old)
            if mid < need:
                lef = mid
            else:
                rig = mid
        mid = t.sqrt(lef * rig)
        theta_new = add(theta_old, F, mid * lr)
        set_theta(theta_new)

        with t.no_grad():
            samples.append(model(x_test))

    return t.stack(samples)


def kernel_demo():
    plt.figure(figsize=(10, 10))

    gp = GP(n_random=10**4, kernel="laplace", kernel_scale=1.0)
    r = 10
    x = t.linspace(-r, r, 200)
    w = gp.get_w(x)
    c = t.corrcoef(w).detach()

    for i in [10, len(c) // 2, len(c) - 10]:
        plt.plot(x, c[i])
    plt.show()

    plt.contourf(c)
    plt.show()


def deep_gp_demo():
    plt.figure(figsize=(10, 10))
    for _ in range(4):
        gp = t.nn.Sequential(
            GP(n_random=10**3, kernel="gauss", kernel_scale=3.0),
            GP(n_random=10**3, kernel="gauss", kernel_scale=10.0),
        )
        x = t.linspace(-10, 10, 200)
        y = gp(x).detach()
        plt.plot(x, y)
    plt.show()


def fitting_demo():
    x_train = t.tensor([-3, -2, 2], dtype=t.float32)
    y_train = t.tensor([1, 0, 1], dtype=t.float32)

    plt.figure(figsize=(10, 10))
    for _ in range(10):
        gp = t.nn.Sequential(
            GP(n_random=10**3, kernel="gauss", kernel_scale=3.0),
            GP(n_random=10**3, kernel="gauss", kernel_scale=10.0),
        )
        gp = fit(gp, x_train, y_train, lr=0.003, epochs=100)
        x = t.linspace(-10, 10, 200)
        y = gp(x).detach()
        plt.plot(x, y, "k-", lw=0.5)

    plt.plot(x_train, y_train, "ro", ms=5)
    plt.show()


def smooth_sampling_demo():
    model = GP(n_random=10**3, kernel="cauchy", kernel_scale=3.0)
    force_schedule = [
        lambda time: t.randn((10**3,)),
    ]
    x_train = t.tensor([-3, -2, 2], dtype=t.float32)
    y_train = t.tensor([1, 0, 10], dtype=t.float32)
    x_test = t.linspace(-10, 10, 200)
    samples = smooth_sample(
        model,
        x_train,
        y_train,
        x_test,
        n_samples=10000,
        T_schedule=lambda time: 0.1,
        force_schedule=force_schedule,
        lr=10.0,
    )[::10]

    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap("plasma")
    for i, sample in enumerate(samples):
        color = cmap(i / len(samples))
        plt.plot(x_test, sample, color=color, lw=0.2)
    std = samples.std(0)
    mu = samples.mean(0)
    lw = 2
    plt.plot(x_test, mu, "k--", lw=lw, label="mean")
    plt.plot(x_test, mu - 2 * std, "k--", lw=lw, label="mean +/- 2 std")
    plt.plot(x_test, mu + 2 * std, "k--", lw=lw, label="mean +/- 2 std")
    plt.plot(x_train, y_train, "ko", ms=5)
    plt.legend()
    plt.tight_layout()
    plt.show()


smooth_sampling_demo()
