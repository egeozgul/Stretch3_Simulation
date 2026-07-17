from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR_DEFAULT = Path(__file__).resolve().parent / "data"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "marc_vs_baseline_returns_by_instruction.png"

# label -> (baseline_csv, marc_csv)
INSTRUCTION_FILES: Dict[str, Tuple[str, str]] = {
	"stay": (
		"baseline_return_stay.csv",
		"mac_iac_marc_return_stay.csv",
	),
	"left cutting": (
		"baseline_return_left_cutting.csv",
		"mac_iac_marc_return_left_cutting.csv",
	),
	"right cutting": (
		"baseline_return_right_cutting.csv",
		"mac_iac_marc_return_right_cutting.csv",
	),
}

METHOD_COLORS: Dict[str, str] = {
	"Baseline": "#D55E00",
	"MARC": "#0072B2",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compare MARC vs Baseline per-instruction returns (stay, "
			"left cutting, right cutting). Renders one subplot per instruction."
		)
	)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
	parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
	parser.add_argument(
		"--smooth-window",
		type=int,
		default=25,
		help="Rolling-average window for smoothing the per-episode returns.",
	)
	parser.add_argument(
		"--sample-frac",
		type=float,
		default=1.0,
		help="Fraction of rows to keep (random sampling). 1.0 = keep all.",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--share-y",
		action="store_true",
		help="Share y-axis across subplots for direct scale comparison.",
	)
	return parser.parse_args()


def _find_returns_columns(df: pd.DataFrame):
	mean_candidates = [
		col
		for col in df.columns
		if "returns_instruction" in col.lower()
		and not col.endswith("__MIN")
		and not col.endswith("__MAX")
	]
	if not mean_candidates:
		raise ValueError(f"No Returns_Instruction column in {list(df.columns)}")
	mean_col = mean_candidates[0]
	min_col = f"{mean_col}__MIN" if f"{mean_col}__MIN" in df.columns else mean_col
	max_col = f"{mean_col}__MAX" if f"{mean_col}__MAX" in df.columns else mean_col
	return mean_col, min_col, max_col


def _load_csv(path: Path, instruction_label: str, method: str) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing data file: {path}")

	df = pd.read_csv(path)
	mean_col, min_col, max_col = _find_returns_columns(df)

	out = pd.DataFrame(
		{
			"episode": pd.to_numeric(df["Episode"], errors="coerce"),
			"returns": pd.to_numeric(df[mean_col], errors="coerce"),
			"returns_min": pd.to_numeric(df[min_col], errors="coerce"),
			"returns_max": pd.to_numeric(df[max_col], errors="coerce"),
			"instruction": instruction_label,
			"method": method,
		}
	)
	out = out.dropna(subset=["episode", "returns"]).sort_values("episode").reset_index(drop=True)
	return out


def _random_sample(df: pd.DataFrame, sample_frac: float, seed: int) -> pd.DataFrame:
	if not (0.0 < sample_frac <= 1.0):
		raise ValueError("sample_frac must be in (0, 1].")
	if sample_frac >= 1.0:
		return df
	parts = []
	for (instr, method), part in df.groupby(["instruction", "method"], sort=False):
		n_keep = max(1, int(round(len(part) * sample_frac)))
		parts.append(part.sample(n=n_keep, random_state=seed).sort_values("episode"))
	return pd.concat(parts, ignore_index=True)


def _smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
	if window < 1:
		raise ValueError("smooth-window must be >= 1")
	parts = []
	for (instr, method), part in df.groupby(["instruction", "method"], sort=False):
		part = part.sort_values("episode").copy()
		part["returns_smooth"] = part["returns"].rolling(window=window, min_periods=1).mean()
		band = (part["returns_max"] - part["returns_min"]).abs() / 2.0
		part["band_smooth"] = band.rolling(window=window, min_periods=1).mean()
		parts.append(part)
	return pd.concat(parts, ignore_index=True)


def plot_comparison(df: pd.DataFrame, output_path: Path, smooth_window: int, share_y: bool) -> None:
	labels = list(INSTRUCTION_FILES.keys())
	n = len(labels)

	fig, axes = plt.subplots(
		1, n, figsize=(5.5 * n, 5.5), sharey=share_y, squeeze=False
	)
	axes = axes[0]

	for ax, label in zip(axes, labels):
		for method in ("Baseline", "MARC"):
			part = df[(df["instruction"] == label) & (df["method"] == method)]
			if part.empty:
				continue
			color = METHOD_COLORS[method]
			ax.plot(
				part["episode"],
				part["returns_smooth"],
				label=method,
				color=color,
				linewidth=2,
			)
			ax.fill_between(
				part["episode"],
				part["returns_smooth"] - part["band_smooth"],
				part["returns_smooth"] + part["band_smooth"],
				color=color,
				alpha=0.18,
			)

		ax.set_title(label)
		ax.set_xlabel("Episodes")
		ax.grid(alpha=0.25)
		ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
		ax.legend(loc="best")

	axes[0].set_ylabel("Returns")
	fig.suptitle(
		f"MARC vs Baseline returns by instruction (smoothed, window={smooth_window})",
		y=1.02,
	)
	fig.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = parse_args()

	frames = []
	for label, (baseline_name, marc_name) in INSTRUCTION_FILES.items():
		frames.append(_load_csv(args.data_dir / baseline_name, label, "Baseline"))
		frames.append(_load_csv(args.data_dir / marc_name, label, "MARC"))
	combined = pd.concat(frames, ignore_index=True)

	sampled = _random_sample(combined, sample_frac=args.sample_frac, seed=args.seed)
	smoothed = _smooth(sampled, window=args.smooth_window)

	plot_comparison(smoothed, args.output, smooth_window=args.smooth_window, share_y=args.share_y)

	print("Row counts per (instruction, method):")
	print(sampled.groupby(["instruction", "method"]).size())
	print(f"\nSaved plot to: {args.output}")


if __name__ == "__main__":
	main()
