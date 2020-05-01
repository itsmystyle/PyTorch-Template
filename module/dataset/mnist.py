import torchvision.transforms as transforms

from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, data, mode="train", num_class=10, transform=None):

        if mode == "train":
            self.data = data["Train"]
        elif mode == "valid":
            self.data = data["Valid"]
        else:
            raise NotImplementedError

        self.num_class = num_class

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transofmr = transform

    def __len__(self):
        return self.data["X"].shape[0]

    def __getitem__(self, index):
        return {"X": self.transform(self.data["X"][index]), "y": self.data["y"][index]}

    @property
    def num_classes(self):
        return self.num_class
