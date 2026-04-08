import argparse
import pandas as pd
from pathlib import Path


KNOWN_QUERY_COLUMNS = [
    "query",
    "prompt",
    "instruction",
    "input",
    "question",
]

KNOWN_BASE_RESPONSE_COLUMNS = [
    "base_response",
    "baseline_response",
    "safe_response",
    "reference_response",
    "response_base",
    "response_baseline",
    "response_safe",
]

KNOWN_JAILBREAK_RESPONSE_COLUMNS = [
    "jailbreak_response",
    "attack_response",
    "attacked_response",
    "response_jailbreak",
    "response_attack",
]

KNOWN_JAILBREAK_SUCCESS_COLUMNS = [
    "jailbreak_success",
    "jailbreak_label",
    "is_jailbroken",
    "jailbroken",
]

KNOWN_HUMAN_SAME_COLUMNS = [
    "human_same",
    "same_as_baseline",
    "same_as_reference",
    "same",
    "human_match",
]


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def normalize_binary(series):
    def normalize(value):
        if pd.isna(value):
            return 0
        if isinstance(value, bool):
            return int(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "ok", "okey"}:
            return 1
        if text in {"0", "false", "no", "n"}:
            return 0
        try:
            return 1 if float(text) != 0 else 0
        except ValueError:
            return 0

    return series.apply(normalize).astype(int)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare annotation CSV with query/base response/jailbreak response and 0/1 labels."
    )
    parser.add_argument("input_csv", help="Path to the annotation CSV file")
    parser.add_argument("output_csv", nargs="?", default="outputs/annotation_cleaned.csv",
                        help="Path to save the cleaned output CSV")
    parser.add_argument("--query-col", help="Explicit column name for the query text")
    parser.add_argument("--base-response-col", help="Explicit column name for the baseline response")
    parser.add_argument("--jailbreak-response-col", help="Explicit column name for the jailbreak response")
    parser.add_argument("--jailbreak-success-col", help="Explicit column name for jailbreak success label")
    parser.add_argument("--human-same-col", help="Explicit column name for human same label")
    args = parser.parse_args()

    src_path = Path(args.input_csv)
    if not src_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {src_path}")

    df = pd.read_csv(src_path)

    query_col = args.query_col or find_column(df, KNOWN_QUERY_COLUMNS)
    base_col = args.base_response_col or find_column(df, KNOWN_BASE_RESPONSE_COLUMNS)
    jailbreak_col = args.jailbreak_response_col or find_column(df, KNOWN_JAILBREAK_RESPONSE_COLUMNS)
    jailbreak_success_col = args.jailbreak_success_col or find_column(df, KNOWN_JAILBREAK_SUCCESS_COLUMNS)
    human_same_col = args.human_same_col or find_column(df, KNOWN_HUMAN_SAME_COLUMNS)

    if query_col is None:
        raise ValueError("Could not infer query column. Provide --query-col explicitly.")
    if base_col is None:
        raise ValueError("Could not infer base response column. Provide --base-response-col explicitly.")
    if jailbreak_col is None:
        raise ValueError("Could not infer jailbreak response column. Provide --jailbreak-response-col explicitly.")

    output_df = pd.DataFrame()
    output_df["query"] = df[query_col]
    output_df["base_response"] = df[base_col]
    output_df["jailbreak_response"] = df[jailbreak_col]

    if jailbreak_success_col is not None:
        output_df["jailbreak_success"] = normalize_binary(df[jailbreak_success_col])
    else:
        output_df["jailbreak_success"] = 0

    if human_same_col is not None:
        output_df["human_same"] = normalize_binary(df[human_same_col])
    else:
        output_df["human_same"] = 0

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved cleaned annotation file to: {output_path}")
    print("Columns:", ", ".join(output_df.columns))


if __name__ == "__main__":
    main()
