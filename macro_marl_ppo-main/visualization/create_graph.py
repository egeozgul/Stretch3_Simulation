from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Plot MARC vs baseline returns by episode with mean aggregation, "
			"stderr range, random point sampling, and running-average smoothing."
		)
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "data",
		help="Directory containing marc.csv and baseline.csv",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).resolve().parent / "comparison_marc_vs_baseline.png",
		help="Output image file path",
	)
	parser.add_argument(
		"--sample-frac",
		type=float,
		default=0.7,
		help="Fraction of points to keep per method using random sampling (0, 1]",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible point sampling",
	)
	parser.add_argument(
		"--smooth-window",
		type=int,
		default=8,
		help="Running average window size",
	)
	return parser.parse_args()


def _find_return_column(df: pd.DataFrame) -> str:
	candidates = [col for col in df.columns if "returns" in col.lower() and "__" not in col]
	if not candidates:
		raise ValueError("Could not find returns column in CSV.")
	return candidates[0]


def _load_method_csv(path: Path, method_name: str) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing data file: {path}")

	df = pd.read_csv(path)
	return_col = _find_return_column(df)
	min_col = f"{return_col}__MIN" if f"{return_col}__MIN" in df.columns else None
	max_col = f"{return_col}__MAX" if f"{return_col}__MAX" in df.columns else None

	out_dict = {
		"episode": pd.to_numeric(df["Episode"], errors="coerce"),
		"returns": pd.to_numeric(df[return_col], errors="coerce"),
		"method": method_name,
	}
	if min_col is not None:
		out_dict["returns_min"] = pd.to_numeric(df[min_col], errors="coerce")
	if max_col is not None:
		out_dict["returns_max"] = pd.to_numeric(df[max_col], errors="coerce")

	out = pd.DataFrame(out_dict)
	out = out.dropna(subset=["episode", "returns"])
	return out


def _aggregate_with_stderr(df: pd.DataFrame) -> pd.DataFrame:
	agg_map = {
		"mean_returns": ("returns", "mean"),
		"std_returns": ("returns", "std"),
		"n": ("returns", "size"),
	}
	if "returns_min" in df.columns:
		agg_map["min_returns"] = ("returns_min", "min")
	if "returns_max" in df.columns:
		agg_map["max_returns"] = ("returns_max", "max")

	grouped = df.groupby(["method", "episode"], as_index=False).agg(**agg_map)
	grouped["std_returns"] = grouped["std_returns"].fillna(0.0)
	grouped["stderr"] = grouped["std_returns"] / (grouped["n"].clip(lower=1) ** 0.5)

	# If rows are already grouped (n=1), stderr becomes 0; approximate a visible band from min/max.
	if "min_returns" in grouped.columns and "max_returns" in grouped.columns:
		band_fallback = (grouped["max_returns"] - grouped["min_returns"]).abs() / 2.0
		grouped["stderr"] = grouped["stderr"].where(grouped["stderr"] > 0, band_fallback)
		grouped["stderr"] = grouped["stderr"].fillna(0.0)

	selected = grouped[["method", "episode", "mean_returns", "stderr"]]
	return pd.DataFrame(selected)


def _random_sample_by_method(df: pd.DataFrame, sample_frac: float, seed: int) -> pd.DataFrame:
	if not (0.0 < sample_frac <= 1.0):
		raise ValueError("sample_frac must be in the interval (0, 1].")

	sampled_parts = []
	for method, part in df.groupby("method", sort=False):
		n_total = len(part)
		n_keep = max(1, int(round(n_total * sample_frac)))
		sampled = part.sample(n=n_keep, random_state=seed).sort_values("episode")
		sampled_parts.append(sampled)

	sampled_df = pd.concat(sampled_parts, ignore_index=True)
	return sampled_df


def _smooth_running_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
	if window < 1:
		raise ValueError("smooth_window must be >= 1")

	smoothed_parts = []
	for method, part in df.groupby("method", sort=False):
		part = part.sort_values("episode").copy()
		part["mean_smooth"] = part["mean_returns"].rolling(window=window, min_periods=1).mean()
		part["stderr_smooth"] = part["stderr"].rolling(window=window, min_periods=1).mean()
		smoothed_parts.append(part)

	return pd.concat(smoothed_parts, ignore_index=True)


def plot_comparison(df: pd.DataFrame, output_path: Path, smooth_window: int) -> None:
	plt.figure(figsize=(11, 6))

	colors = {
		"MARC": "#0072B2",
		"Baseline": "#D55E00",
	}

	for method, part in df.groupby("method", sort=False):
		method_name = str(method)
		color = colors.get(method_name)

		plt.plot(
			part["episode"],
			part["mean_smooth"],
			label=f"{method_name} (mean, smooth={smooth_window})",
			linewidth=2,
			color=color,
		)
		plt.fill_between(
			part["episode"],
			part["mean_smooth"] - part["stderr_smooth"],
			part["mean_smooth"] + part["stderr_smooth"],
			alpha=0.22,
			color=color,
		)

	plt.xlabel("Episodes")
	plt.ylabel("Returns")
	plt.title("MARC vs Baseline: Returns over Episodes")
	plt.legend()
	plt.grid(alpha=0.25)
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=200)
	plt.close()


def main() -> None:
	args = parse_args()

	baseline_path = args.data_dir / "baseline.csv"
	marc_path = args.data_dir / "marc.csv"

	baseline_df = _load_method_csv(baseline_path, "Baseline")
	marc_df = _load_method_csv(marc_path, "MARC")
	combined = pd.concat([baseline_df, marc_df], ignore_index=True)

	grouped = _aggregate_with_stderr(combined)
	sampled = _random_sample_by_method(grouped, sample_frac=args.sample_frac, seed=args.seed)
	smoothed = _smooth_running_average(sampled, window=args.smooth_window)

	plot_comparison(smoothed, args.output, smooth_window=args.smooth_window)
	print(f"Saved graph to: {args.output}")


if __name__ == "__main__":
	main()
