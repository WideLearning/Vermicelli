from imports import *
from optimizer import LocalSWA
from math import sqrt
from torchvision.utils import make_grid
from data import PointDataset


@typed
def init(weight: Float[TT, "n m"] | Float[TT, "n"], std: float):
    weight.data = t.randn_like(weight.data)
    weight.data *= std / weight.std()


class MLP(nn.Module):
    @typed
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

        for h in self.hidden_layers:
            init(h.weight, sqrt(2 / hidden_size))
            init(h.bias, 1e-5)

    @typed
    def forward(
        self, x: Float[TT, "batch_size input_size"]
    ) -> Float[TT, "batch_size output_size"]:
        x = F.mish(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.mish(layer(x))
        x = self.output_layer(x)
        return x


class ResMLP(MLP):
    @typed
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__(input_size, hidden_size, output_size, n_layers)
        for h in self.hidden_layers:
            init(h.weight, sqrt(2 / (hidden_size * n_layers**2)))

    @typed
    def forward(
        self, x: Float[TT, "batch_size input_size"]
    ) -> Float[TT, "batch_size output_size"]:
        x = F.mish(self.input_layer(x))
        for layer in self.hidden_layers:
            x = x + F.mish(layer(x))
        x = self.output_layer(x)
        return x


class BiResMLP(MLP):
    @typed
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__(input_size, hidden_size, output_size, n_layers)
        self.output_layers = nn.ModuleList(
            nn.Linear(hidden_size, output_size) for _ in range(n_layers)
        )
        for h in self.hidden_layers:
            init(h.weight, sqrt(2 / (hidden_size * n_layers**2)))
        for o in self.output_layers:
            init(o.weight, sqrt(1 / (hidden_size * n_layers)))

    @typed
    def forward(
        self, x: Float[TT, "batch_size input_size"]
    ) -> Float[TT, "batch_size output_size"]:
        x = F.mish(self.input_layer(x))
        y = t.zeros(self.output_size, device=x.device)
        for hidden, output in zip(self.hidden_layers, self.output_layers):
            y = y + output(x)
            x = x + F.mish(hidden(x))
        y = y + self.output_layer(x)
        return y


class DenseMLP(MLP):
    @typed
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__(input_size, hidden_size, output_size, n_layers)
        self.layer_attention = nn.Parameter(
            t.triu(t.ones((n_layers + 1, n_layers + 1)))
        )
        for h in self.hidden_layers:
            init(h.weight, sqrt(2 / (hidden_size * n_layers)))

    @typed
    def forward(
        self, x: Float[TT, "batch_size input_size"]
    ) -> Float[TT, "batch_size output_size"]:
        x = self.input_layer(x)
        intermediate = t.zeros((self.n_layers + 1, *x.shape))
        intermediate[0] = x
        for i, hidden in enumerate(self.hidden_layers):
            previous_activation = hidden(F.mish(intermediate[i]))
            intermediate = intermediate + ein.einsum(
                self.layer_attention[i],
                previous_activation,
                "layers, ... -> layers ...",
            )
        x = self.output_layer(F.mish(intermediate[-1]))
        return x


class MLPModel(pl.LightningModule):
    @typed
    def __init__(
        self,
        store: PointDataset,
        model_type: str,
        lr: float = 1e-3,
        k: int = 100,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.store = store
        model_constructor = {
            "MLP": MLP,
            "ResMLP": ResMLP,
            "BiResMLP": BiResMLP,
        }[model_type]
        self.model = model_constructor(*args, **kwargs)
        self.lr = lr
        self.k = k

    @typed
    def forward(self, x: TT) -> TT:
        return self.model(x)

    @typed
    def configure_optimizers(self) -> t.optim.Optimizer:
        inner = t.optim.Adam(self.parameters(), lr=self.lr)
        outer = LocalSWA(inner, k=self.k)
        return outer

    @typed
    def training_step(self, batch: list[TT], batch_idx: int) -> Float[TT, ""]:
        x, y, w = batch
        assert_type(x, Float[TT, "batch_size input_size"])
        assert_type(y, Float[TT, "batch_size 3"])
        assert_type(w, Float[TT, "batch_size"])
        y_hat = self(x)
        num = (F.mse_loss(y_hat, y, reduction="none").mean(dim=-1) * w).sum()
        denom = w.sum() + 1e-8
        loss = num / denom
        assert_type(loss, Float[TT, ""])
        self.log("train_loss", loss)
        return loss

    @typed
    def on_validation_epoch_start(self) -> None:
        try:
            optimizer = self.trainer.optimizers[0]
            optimizer._backup_and_load_cache()
        except:
            pass

    @typed
    def on_validation_epoch_end(self) -> None:
        try:
            optimizer = self.trainer.optimizers[0]
            optimizer._clear_and_load_backup()
        except:
            pass

        image = make_grid(
            [
                self.store.predictions(self.model, i)
                for i in range(len(self.store.sizes))
            ]
        )
        self.logger.log_image("predictions", image)

    @typed
    def validation_step(self, batch: list[TT], batch_idx: int) -> None:
        x, y, w = batch
        assert_type(x, Float[TT, "batch_size input_size"])
        assert_type(y, Float[TT, "batch_size 3"])
        assert_type(w, Float[TT, "batch_size"])
        y_hat = self(x)
        num = (F.mse_loss(y_hat, y, reduction="none").mean(dim=-1) * w).sum()
        denom = w.sum() + 1e-8
        loss = num / denom
        assert_type(loss, Float[TT, ""])
        self.log("val_loss", loss)
