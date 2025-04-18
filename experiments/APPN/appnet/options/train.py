from .base import BaseOptions


class TrainOptions(BaseOptions):
    """Parse arguments for training"""

    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument("--batch_size", required=True, type=int, help="batch size for training")
        parser.add_argument("--log_dir", default="runs", type=str, help="directory to store log file")
        parser.add_argument(
            "--task_name",
            required=True,
            type=str,
            help="task name, experiment name. The final path of your logs is <log_dir>/<task_name>/<timestamp>",
        )
        parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
        parser.add_argument("--momentum", default=0.90, type=float, help="momentum for optimizer")
        parser.add_argument("--decay", default=0.0005, type=float, help="weight decay for optimizer")
        parser.add_argument("--gamma", default=0.1, type=float, help="gamma for lr scheduler")
        parser.add_argument(
            "--milestones",
            metavar="N",
            nargs="+",
            type=int,
            default=[10, 25, 40],
            help="milestones to reduce learning rate",
        )
        parser.add_argument("--epochs", default=50, type=int, help="number of epochs for training")
        parser.add_argument("--channels", default=512, type=int, help="number of epochs for training")
        parser.add_argument(
            "--smooth_kernel", default=5, type=int, help="kernel size of blur operation applied to error scores"
        )
        self.parser = parser

    def parse(self):
        args = super().parse()
        args.phase = "train"
        return args
