import argparse
import subprocess
import sys


def run_script(module_name):
    subprocess.run([sys.executable, "-m", module_name])

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Hand Gesture Recognition System"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["collect", "train", "run"],
        help="Mode to run the system"
    )

    args = parser.parse_args()

    if args.mode == "collect":
        run_script("scripts.collect_data")

    elif args.mode == "train":
        run_script("scripts.train_model")

    elif args.mode == "run":
        run_script("scripts.realtime_inference")


if __name__ == "__main__":
    main()

