import os
from huggingface_hub import login
from argparse import ArgumentParser


def login_huggingface():
    """
    Logs in to Huggingface using the provided API token.
    Requires one of the following:
    1. Argument to be passed.
    2. HUGGINGFACE_API_TOKEN environment variable to be set.
    3. Else, it will ask for the API token in the CLI.
    """
    ArgumentParser().add_argument("-t", "--api_token", type=str)
    args = ArgumentParser().parse_args()
    api_token = None
    if args.api_token:
        api_token = args.api_token
    elif not os.environ.get("HUGGINGFACE_API_TOKEN"):
        api_token = os.environ["HUGGINGFACE_API_TOKEN"]
    login(api_token)
    print("[Info] ~ Logged in to Huggingface.", flush=True)


if __name__ == "__main__":
    login_huggingface()
