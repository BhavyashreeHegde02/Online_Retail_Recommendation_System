from __future__ import annotations

import argparse
from pathlib import Path

from retail_cleaning_pipeline import CleaningConfig, clean_online_retail, save_cleaned_to_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean OnlineRetail.xlsx and save analysis-ready Excel output")
    parser.add_argument("--input", default="OnlineRetail.xlsx", help="Input Excel path")
    parser.add_argument("--output", default="OnlineRetail_cleaned.xlsx", help="Output Excel path")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    outputs = clean_online_retail(str(input_path), cfg=CleaningConfig())
    save_cleaned_to_excel(outputs, str(output_path))

    print(f"Saved cleaned dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
