from imports import *


class ResMLP(nn.Module):
    @typed
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    @typed
    def forward(
        self, x: Float[TT, "batch_size input_size"]
    ) -> Float[TT, "batch_size output_size"]:
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = x + F.relu(layer(x))
        x = self.output_layer(x)
        return x


class BiResMLP(nn.Module):
    @typed
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)
        )
        self.output_layers = nn.ModuleList(
            nn.Linear(hidden_size, output_size) for _ in range(n_layers)
        )
        self.final_output = nn.Linear(hidden_size, output_size)

    @typed
    def forward(
        self, x: Float[TT, "batch_size input_size"]
    ) -> Float[TT, "batch_size output_size"]:
        x = F.relu(self.input_layer(x))
        y = t.zeros(self.output_size)
        for hidden, output in zip(self.hidden_layers, self.output_layers):
            y = y + output(x)
            x = x + F.relu(hidden(x))
        y = y + self.final_output(x)
        return y


class PointModel(pl.LightningModule):
    @typed
    def __init__(self, model):
        super().__init__()
        self.model = model

    @typed
    def forward(self, x: TT) -> TT:
        return self.model(x)

    @typed
    def configure_optimizers(self) -> t.optim.Optimizer:
        optimizer = t.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
    def validation_step(self, batch: tuple[TT, TT, TT], batch_idx: int) -> None:
        x, y, masked = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y) * (1 - masked)
        self.log("val_loss", loss)
