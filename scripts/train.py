import logging

import torch

from module.argparser import ArgParser
from module.dataset.factory import create_train_valid_dataset
from module.model.factory import create_model
from module.trainer import Trainer

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device use to train {}".format(device))

    # Preparing arguments
    parser = ArgParser()
    args = parser.parse()

    # Preparing dataset
    if args.data_dir is None:
        raise Exception("Data directory is not given.")
    data = create_train_valid_dataset(args)

    # Preparing model
    model = create_model(args, data["trainset"].num_classes)

    # Preparing trainer
    trainer = Trainer(
        args=args,
        train_dataloader=data["train_dataloader"],
        valid_dataloader=data["valid_dataloader"],
        model=model,
    )
    trainer.fit(args.epochs)
