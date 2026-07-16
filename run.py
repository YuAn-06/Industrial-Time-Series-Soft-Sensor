"""Backward-compatible CLI-argument entry point for a full train/test run."""

from runner import train_test
from utils import Parse_arguments, print_args


if __name__ == "__main__":
    args = Parse_arguments()
    print_args(args)
    train_test(args)
