from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.fundamentals import (
    enrich_research_frame_with_fundamentals,
    load_fundamentals_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join latest available fundamentals into an existing research frame."
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--fundamentals", default="data/reference/fundamentals.csv")
    parser.add_argument("--output", default="data/processed/research_frame_enriched.csv")
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    fundamentals = load_fundamentals_csv(args.fundamentals)
    enriched = enrich_research_frame_with_fundamentals(research_frame, fundamentals)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    print(
        f"wrote {len(enriched):,} enriched rows with {len(enriched.columns)} columns "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
