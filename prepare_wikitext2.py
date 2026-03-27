import argparse
import os
import urllib.request


WIKITEXT2_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WikiText-2 and save a CPU-friendly subset.")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--max-train-chars", type=int, default=350000)
    parser.add_argument("--max-valid-chars", type=int, default=60000)
    parser.add_argument("--max-test-chars", type=int, default=60000)
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train = fetch_text(WIKITEXT2_URLS["train"])[: args.max_train_chars]
    valid = fetch_text(WIKITEXT2_URLS["valid"])[: args.max_valid_chars]
    test = fetch_text(WIKITEXT2_URLS["test"])[: args.max_test_chars]

    train_path = os.path.join(args.out_dir, "wikitext2_train_small.txt")
    valid_path = os.path.join(args.out_dir, "wikitext2_valid_small.txt")
    test_path = os.path.join(args.out_dir, "wikitext2_test_small.txt")
    merged_path = os.path.join(args.out_dir, "wikitext2_small_all.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train)
    with open(valid_path, "w", encoding="utf-8") as f:
        f.write(valid)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(train + "\n" + valid + "\n" + test)

    print(f"Saved: {train_path} ({len(train)} chars)")
    print(f"Saved: {valid_path} ({len(valid)} chars)")
    print(f"Saved: {test_path} ({len(test)} chars)")
    print(f"Saved: {merged_path} ({len(train) + len(valid) + len(test)} chars)")


if __name__ == "__main__":
    main()
