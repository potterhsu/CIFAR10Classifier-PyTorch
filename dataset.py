import torch.utils.data as data
import lmdb
import example_pb2
import numpy as np
from PIL import Image
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, path_to_lmdb_dir):
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = [key for key, _ in txn.cursor()]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        with self._reader.begin() as txn:
            value = txn.get(self._keys[index])

        example = example_pb2.Example()
        example.ParseFromString(value)

        image = np.fromstring(example.image, dtype=np.uint8)
        image = image.reshape([32, 32, 3])
        image = Image.fromarray(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = transform(image)
        label = example.label

        return image, label
