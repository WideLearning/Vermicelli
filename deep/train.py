from imports import *
from dvclive import Live  # type: ignore
from dvclive.lightning import DVCLiveLogger  # type: ignore
from data import PointDataset
from models import ResMLP, BiResMLP, PointModel
import yaml

store = PointDataset(id_bits=8)
store.load("img/vertical.png")
store.load("img/sine.png")
store.load("img/squares.png")
store.load("img/triangles.png")
dl = t.utils.data.DataLoader(store, batch_size=64, shuffle=True, num_workers=4)
val_dl = t.utils.data.DataLoader(store, batch_size=128, shuffle=False, num_workers=4)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

model = PointModel(
    store=store,
    model_type=params["model_type"],
    input_size=43,
    hidden_size=params["hidden_size"],
    output_size=3,
    n_layers=params["n_layers"],
    lr=params["lr"],
    k=params["k"],
)

trainer = pl.Trainer(
    logger=DVCLiveLogger(
        log_model=True,
        # run_name="",
    ),
    check_val_every_n_epoch=5,
    max_epochs=params["max_epochs"],
)
trainer.fit(model, train_dataloaders=dl, val_dataloaders=val_dl)
