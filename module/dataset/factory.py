import pickle

from torch.utils.data import DataLoader

from module.dataset import MNIST


def create_train_valid_dataset(args, mode=None):

    with open(args.data_dir, "rb") as fin:
        data = pickle.load(fin)

    trainset = MNIST(data, mode="train")
    train_dataloader = DataLoader(
        trainset, batch_size=args.bach_size, num_workers=args.num_workers, shuffle=True,
    )

    validset = MNIST(data, mode="valid")
    valid_dataloader = DataLoader(
        validset, batch_size=args.bach_size, num_workers=args.num_workers, shuffle=False,
    )

    if mode == "test":
        testset = MNIST(data, mode="test")
        test_dataloader = DataLoader(
            testset, batch_size=args.bach_size, num_workers=args.num_workers, shuffle=False,
        )

        return {
            "trainset": trainset,
            "train_dataloader": train_dataloader,
            "validset": validset,
            "valid_dataloader": valid_dataloader,
            "testset": testset,
            "test_dataloader": test_dataloader,
        }

    return {
        "trainset": trainset,
        "train_dataloader": train_dataloader,
        "validset": validset,
        "valid_dataloader": valid_dataloader,
    }
