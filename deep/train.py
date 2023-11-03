from imports import *
from dvclive import Live  # type: ignore
from dvclive.lightning import DVCLiveLogger  # type: ignore
from data import PointDataset
from models import ResMLP, BiResMLP, PointModel

store = PointDataset(id_bits=8)
store.load("img/vertical.png")
store.load("img/sine.png")
store.load("img/squares.png")
store.load("img/triangles.png")
dl = t.utils.data.DataLoader(store, batch_size=64, shuffle=True, num_workers=4)
val_dl = t.utils.data.DataLoader(store, batch_size=128, shuffle=False, num_workers=4)

model = PointModel(
    store=store,
    model_type="ResMLP",
    input_size=43,
    hidden_size=64,
    output_size=3,
    n_layers=10,
)
trainer = pl.Trainer(
    logger=DVCLiveLogger(
        log_model=True,
        # run_name="",
    ),
    check_val_every_n_epoch=5,
    max_epochs=100,
)
trainer.fit(model, train_dataloaders=dl, val_dataloaders=val_dl)



