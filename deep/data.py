from imports import *
from PIL import Image


@typed
def read_picture(filename: str) -> Float[TT, "h w 3"]:
    arr = t.tensor(np.array(Image.open(filename))[..., :3], dtype=t.float)
    return arr / 255


class ImageStore:
    @typed
    def __init__(self):
        self.images = []
        self.mask_color = t.tensor([0, 1, 0], dtype=t.float)

    @typed
    def load(self, filename: str) -> None:
        self.images.append(read_picture(filename))

    @typed
    def plot(self, index: int) -> None:
        plt.subplot(1, 2, 1)
        plt.imshow(self.images[index])
        plt.subplot(1, 2, 2)
        masked = (self.images[index] == self.mask_color).all(dim=2).float()
        plt.set_cmap("gray")
        plt.imshow(masked)
        plt.tight_layout()
        plt.show()


def trigonometric_features(x: float, k: int) -> Float[TT, "k"]:
    assert 0 <= x <= 1
    r = 2 ** t.arange(k)
    return t.cat(
        [t.tensor([x]), t.cos(2 * t.pi * r * x), t.sin(2 * t.pi * r * x)], dim=0
    )


class PointDataset(ImageStore, t.utils.data.Dataset):
    @typed
    def __init__(self, id_bits: int):
        super().__init__()
        self.sizes: list[int] = []
        self.max_id = 2**id_bits
        self.id_features = id_bits
        self.image_features = 6

    @typed
    def load(self, filename: str) -> None:
        super().load(filename)
        shape = self.images[-1].shape
        self.sizes.append(shape[0] * shape[1])

    @typed
    def __len__(self) -> int:
        return sum(self.sizes)

    @typed
    def get_features(self, id: int, i: int, j: int) -> Float[TT, "features"]:
        id_features = trigonometric_features(id / self.max_id, self.id_features)
        max_i, max_j = self.images[id].shape[:2]
        i_features = trigonometric_features(i / max_i, self.image_features)
        j_features = trigonometric_features(j / max_j, self.image_features)
        return t.cat([id_features, i_features, j_features], dim=0)

    @typed
    def __getitem__(self, index: int) -> tuple[Float[TT, "features"], Float[TT, "3"], Float[TT, ""]]:
        id = 0
        while index >= self.sizes[id]:
            index -= self.sizes[id]
            id += 1
        shape = self.images[id].shape
        i = index // shape[1]
        j = index % shape[1]
        label = self.images[id][i, j]
        masked = (label == self.mask_color).all().float()
        return self.get_features(id, i, j), label, 1 - masked

    @typed
    def predictions(self, model, id: int) -> Float[TT, "h w 3"]:
        shape = self.images[id].shape
        x = [
            self.get_features(id, i, j)
            for i in range(shape[0])
            for j in range(shape[1])
        ]
        y = model(t.stack(x)).detach().cpu().reshape(shape).clip(0, 1)
        return y
