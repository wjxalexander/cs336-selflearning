from pathlib import Path
import requests
import pickle
import gzip


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"


def load_mnist():
    """Download MNIST once, then return ((x_train, y_train), (x_valid, y_valid))."""
    PATH.mkdir(parents=True, exist_ok=True)
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).write_bytes(content)

    with gzip.open(PATH / FILENAME, "rb") as f:
        (x_train, y_train), (x_valid, y_valid), _ = pickle.load(f, encoding="latin-1")

    return (x_train, y_train), (x_valid, y_valid)
