from time import sleep

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError as exc:
    raise SystemExit("tqdm is not installed. Install it with: pip install tqdm") from exc


def main():
    for _ in tqdm(range(100), desc="tqdm demo", unit="step"):
        sleep(0.03)

    print("tqdm demo done")


if __name__ == "__main__":
    main()
