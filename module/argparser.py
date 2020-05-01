import argparse


class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PyTorch Model Arguments Parser.")

        # data arguments
        self.parser.add_argument("--data", type=str, help="Data path. (e.g. data.csv)")
        self.parser.add_argument("-d", "--data_dir", type=str, help="Data directory.")
        self.parser.add_argument("--test_data", type=str, help="Testing data.")

        # model arguments
        self.parser.add_argument(
            "-md", "--model_dir", type=str, help="Directory to save model weights and logs."
        )
        self.parser.add_argument("-m", "--model", type=str, help="Model weight path.")

        # optimizer arguments
        self.parser.add_argument(
            "-o",
            "--multi_optimizer",
            action="store_true",
            help="Whether to train with multiple optimizer.",
        )
        self.parser.add_argument(
            "-lr", "--learning_rate", type=float, default=1e-5, help="Learning rate."
        )
        self.parser.add_argument(
            "-eps", "--adam_epsilon", type=float, default=1e-8, help="AdamW eposilon."
        )
        self.parser.add_argument(
            "-wd", "--weight_decay", type=float, default=0.0, help="Weight decay."
        )

        # loss arguments
        self.parser.add_argument(
            "-ls", "--smoothing", action="store_true", help="Whether to smooth label."
        )
        self.parser.add_argument(
            "-lt", "--triplet", action="store_true", help="Whether to use triplet loss."
        )
        self.parser.add_argument(
            "-lc", "--center", action="store_true", help="Whether to use center loss."
        )
        self.parser.add_argument(
            "-la",
            "--arcface",
            action="store_true",
            help="Whether to use arcface margin product loss.",
        )
        self.parser.add_argument("--scale", type=float, default=30.0, help="Scale for arcface.")
        self.parser.add_argument("--margin", type=float, default=0.5, help="Margin for arcface.")
        self.parser.add_argument(
            "--dist", type=str, default="euc", help="Which distrance function to user, euc or cos.",
        )
        self.parser.add_argument(
            "--feature_dim", type=int, default=512, help="Final features dimension."
        )

        # training
        self.parser.add_argument("-b", "--bach_size", type=int, default=16, help="Batch size.")
        self.parser.add_argument("-e", "--epochs", type=int, default=20, help="Epochs.")

        # utils
        self.parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
        self.parser.add_argument(
            "--num_workers", type=int, default=8, help="Number of workers for dataloader."
        )

    def parse(self):
        return self.parser.parse_args()
