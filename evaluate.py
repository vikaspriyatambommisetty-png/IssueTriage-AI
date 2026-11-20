import argparse
from collections import Counter

from triage import LABELS, analyze, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", action="store_true", help="Send the evaluation reports to Hugging Face.")
    args = parser.parse_args()
    matrix = Counter()
    rows = dataset("evaluation.csv")
    for row in rows:
        expected = row["label"]
        try:
            actual = analyze(row["report"], cloud=args.cloud).label
        except ValueError as error:
            raise SystemExit(f"Evaluation incomplete: {error}") from error
        matrix[expected, actual] += 1
        if expected != actual:
            print(f"Expected {expected}, got {actual}: {row['report']}")
    correct = sum(matrix[label, label] for label in LABELS)
    print(f"\n{'Hugging Face' if args.cloud else 'Local classifier'}: {correct}/{len(rows)} correct")
    print("Expected / predicted:", " | ".join(LABELS))
    for label in LABELS:
        print(label, [matrix[label, prediction] for prediction in LABELS])


if __name__ == "__main__":
    main()
