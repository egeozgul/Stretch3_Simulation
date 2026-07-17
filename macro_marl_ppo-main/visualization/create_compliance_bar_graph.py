from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR_DEFAULT = Path(__file__).resolve().parent / "data"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "compliance_bar_marc_vs_baseline.png"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Bar graph of instruction-compliance for MARC vs Baseline. "
		)
	)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
	parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
	parser.add_argument(
		"--baseline-file",
		type=str,
		default="baseline_compliance.csv",
	)
	parser.add_argument(
		"--marc-files",
		nargs="+",
		default=[
			"mac_iac_marc_compliance.csv",
			"mac_iac_marc-compliance2.csv",
		],
		help="One or more CSVs that will be averaged to form the MARC condition.",
	)
	parser.add_argument(
		"--phase-fractions",
		nargs=3,
		type=float,
		default=(1.0 / 3.0, 2.0 / 3.0, 1.0),
		metavar=("EARLY_END", "MID_END", "LATE_END"),
		help=(
			"Fraction of the training run (by row index) that defines the end of the "
			"early / mid / late phases. Defaults split into thirds."
		),
	)
	return parser.parse_args()


def _find_compliance_column(df: pd.DataFrame) -> Tuple[str, str, str]:
	"""Return (mean_col, min_col, max_col) for the compliance metric."""
	mean_candidates = [
		col
		for col in df.columns
		if "compliance" in col.lower() and not col.endswith("__MIN") and not col.endswith("__MAX")
	]
	if not mean_candidates:
		raise ValueError(f"Could not find Instruction_Compliance column in columns: {list(df.columns)}")
	mean_col = mean_candidates[0]
	min_col = f"{mean_col}__MIN"
	max_col = f"{mean_col}__MAX"
	if min_col not in df.columns:
		min_col = mean_col
	if max_col not in df.columns:
		max_col = mean_col
	return mean_col, min_col, max_col


def _load_compliance_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing data file: {path}")
	df = pd.read_csv(path)
	mean_col, min_col, max_col = _find_compliance_column(df)

	out = pd.DataFrame(
		{
			"row_idx": np.arange(len(df), dtype=np.int64),
			"compliance": pd.to_numeric(df[mean_col], errors="coerce"),
			"compliance_min": pd.to_numeric(df[min_col], errors="coerce"),
			"compliance_max": pd.to_numeric(df[max_col], errors="coerce"),
		}
	)
	out = out.dropna(subset=["compliance"]).reset_index(drop=True)
	out["row_idx"] = np.arange(len(out), dtype=np.int64)
	return out


def _combine_marc_runs(frames: List[pd.DataFrame]) -> pd.DataFrame:
	"""Align multiple MARC runs by row index (they share sampling cadence) and
	collapse to a per-step mean + across-run stderr."""
	min_len = min(len(f) for f in frames)
	trimmed = [f.iloc[:min_len].reset_index(drop=True) for f in frames]

	stacked = np.stack([f["compliance"].to_numpy() for f in trimmed], axis=0)
	mean_per_step = stacked.mean(axis=0)
	std_per_step = stacked.std(axis=0, ddof=0)
	n_runs = stacked.shape[0]
	stderr_per_step = std_per_step / np.sqrt(max(n_runs, 1))

	return pd.DataFrame(
		{
			"row_idx": np.arange(min_len, dtype=np.int64),
			"compliance": mean_per_step,
			"compliance_stderr": stderr_per_step,
			"n_runs": n_runs,
		}
	)


def _baseline_to_step_frame(df: pd.DataFrame) -> pd.DataFrame:
	"""Baseline CSV already reports per-step mean + min/max across its grouped runs;
	convert min/max band into a symmetric stderr-like proxy so the bar error bars
	are comparable to MARC's across-run stderr."""
	half_band = (df["compliance_max"] - df["compliance_min"]).abs() / 2.0
	return pd.DataFrame(
		{
			"row_idx": df["row_idx"].to_numpy(),
			"compliance": df["compliance"].to_numpy(),
			"compliance_stderr": half_band.to_numpy(),
		}
	)


def _phase_masks(n_rows: int, fracs: Tuple[float, float, float]) -> List[Tuple[str, np.ndarray]]:
	idx = np.arange(n_rows)
	early_end = int(round(fracs[0] * n_rows))
	mid_end = int(round(fracs[1] * n_rows))
	late_end = int(round(fracs[2] * n_rows))
	return [
		("Early", (idx < early_end)),
		("Mid", (idx >= early_end) & (idx < mid_end)),
		("Late", (idx >= mid_end) & (idx < late_end)),
	]


def _weighted_summary(values: np.ndarray, stderrs: np.ndarray) -> Tuple[float, float]:
	"""Return (mean, pooled-stderr) over a subset of steps.

	We treat each step as an independent sample of the underlying compliance
	process; the reported stderr combines (a) variance of the per-step means
	with (b) the average per-step noise, giving a more honest bar error bar
	than either alone.
	"""
	values = values[np.isfinite(values)]
	stderrs = stderrs[np.isfinite(stderrs)]
	if values.size == 0:
		return float("nan"), float("nan")

	mean = float(values.mean())
	between_var = float(values.var(ddof=0))
	within_var = float(np.mean(stderrs ** 2)) if stderrs.size > 0 else 0.0
	n = values.size
	pooled_stderr = float(np.sqrt((between_var + within_var) / max(n, 1)))
	return mean, pooled_stderr


def build_summary(
	baseline_frame: pd.DataFrame,
	marc_frame: pd.DataFrame,
	phase_fracs: Tuple[float, float, float],
) -> pd.DataFrame:
	rows = []

	overall_b = _weighted_summary(
		baseline_frame["compliance"].to_numpy(),
		baseline_frame["compliance_stderr"].to_numpy(),
	)
	overall_m = _weighted_summary(
		marc_frame["compliance"].to_numpy(),
		marc_frame["compliance_stderr"].to_numpy(),
	)
	rows.append({"phase": "Overall", "method": "Baseline", "mean": overall_b[0], "stderr": overall_b[1]})
	rows.append({"phase": "Overall", "method": "MARC", "mean": overall_m[0], "stderr": overall_m[1]})

	n_rows_baseline = len(baseline_frame)
	n_rows_marc = len(marc_frame)
	for (phase_name, mask_b), (_, mask_m) in zip(
		_phase_masks(n_rows_baseline, phase_fracs),
		_phase_masks(n_rows_marc, phase_fracs),
	):
		b_mean, b_err = _weighted_summary(
			baseline_frame.loc[mask_b, "compliance"].to_numpy(),
			baseline_frame.loc[mask_b, "compliance_stderr"].to_numpy(),
		)
		m_mean, m_err = _weighted_summary(
			marc_frame.loc[mask_m, "compliance"].to_numpy(),
			marc_frame.loc[mask_m, "compliance_stderr"].to_numpy(),
		)
		rows.append({"phase": phase_name, "method": "Baseline", "mean": b_mean, "stderr": b_err})
		rows.append({"phase": phase_name, "method": "MARC", "mean": m_mean, "stderr": m_err})

	return pd.DataFrame(rows)


def plot_bar_graph(summary: pd.DataFrame, output_path: Path, n_marc_runs: int) -> None:
	phases = ["Overall", "Early", "Mid", "Late"]
	methods = ["Baseline", "MARC"]
	colors = {"Baseline": "#D55E00", "MARC": "#0072B2"}

	x = np.arange(len(phases))
	bar_width = 0.38

	fig, ax = plt.subplots(figsize=(10, 6))

	for i, method in enumerate(methods):
		means = []
		errs = []
		for phase in phases:
			row = summary[(summary["phase"] == phase) & (summary["method"] == method)].iloc[0]
			means.append(row["mean"])
			errs.append(row["stderr"])
		offset = (i - 0.5) * bar_width
		bars = ax.bar(
			x + offset,
			means,
			width=bar_width,
			yerr=errs,
			capsize=4,
			color=colors[method],
			edgecolor="black",
			linewidth=0.8,
			label=method,
		)
		for bar, mean_val in zip(bars, means):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.01,
				f"{mean_val:.3f}",
				ha="center",
				va="bottom",
				fontsize=9,
			)

	ax.set_xticks(x)
	ax.set_xticklabels(phases)
	ax.set_ylabel("Instruction compliance (0-1)")
	ax.set_ylim(0.0, 1.05)
	ax.set_title("Instruction compliance: MARC vs Baseline")
	ax.legend(loc="lower right")
	ax.grid(axis="y", alpha=0.3)
	ax.set_axisbelow(True)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def main() -> None:
	args = parse_args()

	baseline_raw = _load_compliance_csv(args.data_dir / args.baseline_file)
	baseline_frame = _baseline_to_step_frame(baseline_raw)

	marc_raw_frames = [_load_compliance_csv(args.data_dir / name) for name in args.marc_files]
	marc_frame = _combine_marc_runs(marc_raw_frames)

	summary = build_summary(baseline_frame, marc_frame, tuple(args.phase_fractions))

	plot_bar_graph(summary, args.output, n_marc_runs=len(marc_raw_frames))

	with pd.option_context("display.float_format", "{:.4f}".format):
		print("Compliance summary:")
		print(summary.pivot(index="phase", columns="method", values="mean").reindex(["Overall", "Early", "Mid", "Late"]))
		print()
		print("Stderr summary:")
		print(summary.pivot(index="phase", columns="method", values="stderr").reindex(["Overall", "Early", "Mid", "Late"]))

	print(f"\nSaved bar graph to: {args.output}")


if __name__ == "__main__":
	main()
