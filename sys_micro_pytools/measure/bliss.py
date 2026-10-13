"""Bliss independence scores for two inputs and one or more responses.

Each row is one replicate. The two input columns are the dose (or treatment)
axes. Each response is a raw signal where larger values mean more viability.
Viability is that signal divided by its mean on the double-control rows
(both inputs at their control levels, 0 and 0 by default).

Bliss independence on viability is::

    E_expected(x, y) = E(x, control_y) * E(control_x, y)
    excess(x, y) = E_expected(x, y) - E_observed(x, y)

Positive excess means the observed viability is lower than independence
(synergy for an inhibitory assay). Negative excess means antagonism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CONTROL_RTOL = 1e-5
_CONTROL_ATOL = 1e-8


def parse_exclude_pairs(
    tokens: Optional[Sequence[str]],
) -> list[tuple[str, str]]:
    """Parse ``column=value`` tokens into pairs.

    Each token is one pair. The value is everything after the first ``=``,
    so values may contain spaces or extra ``=`` characters.
    """
    pairs: list[tuple[str, str]] = []
    for token in tokens or ():
        text = str(token).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"Expected column=value, got {text!r}. "
                "Example: --exclude Cell_type=dead --exclude Well=A12"
            )
        column, value = text.split("=", 1)
        column = column.strip()
        if not column:
            raise ValueError(f"Missing column name in exclude pair {text!r}.")
        pairs.append((column, value.strip()))
    return pairs


def exclude_rows(
    df: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    """Drop rows that match any ``(column, value)`` pair.

    Matching is numeric (with a small tolerance) when the column and the
    value are both numeric, and string equality otherwise. A row is removed
    when it matches at least one pair.

    Returns the filtered frame and the number of rows matched by each pair
    before the pairs are combined.
    """
    if not pairs:
        return df.copy(), {}

    matched = pd.Series(False, index=df.index)
    counts: dict[tuple[str, str], int] = {}
    for column, raw in pairs:
        if column not in df.columns:
            raise ValueError(
                f"Exclude column {column!r} is not in the table. "
                f"Columns: {list(df.columns)}"
            )
        column_match = _equal_mask(df[column], raw)
        counts[(column, raw)] = int(column_match.sum())
        matched = matched | column_match
    return df.loc[~matched].copy(), counts


def run_bliss(
    df: pd.DataFrame,
    input_vars: Sequence[str],
    response_vars: Sequence[str],
    output_dir: Path | str,
    exclude: Optional[Sequence[tuple[str, str]]] = None,
    control_values: Sequence[Any] = (0, 0),
) -> dict[str, dict[str, pd.DataFrame]]:
    """Score Bliss independence and write tables and heatmaps.

    Parameters
    ----------
    df :
        One row per replicate. Must already contain the input and response
        columns (field-level counts should be aggregated before calling).
    input_vars :
        Two column names, the Bliss dose axes.
    response_vars :
        One or more signal columns. Each is normalized to its own
        double-control mean and scored separately.
    output_dir :
        Directory for CSV and PNG outputs. Created if it does not exist.
    exclude :
        ``(column, value)`` pairs. A row is dropped when it matches any pair.
    control_values :
        Control level of each input, in the same order as ``input_vars``.
        Single-agent effects are read from these edges. Default ``(0, 0)``.

    Returns
    -------
    dict
        Per response: ``dose_summary``, ``observed``, ``expected``,
        ``excess``, and ``summary`` frames.
    """
    input1, input2 = _two_names(input_vars, "input_vars")
    responses = _response_names(response_vars)
    control_raw = _two_values(control_values, "control_values")
    overlap = sorted(set(responses) & {input1, input2})
    if overlap:
        raise ValueError(f"Input and response variables overlap: {overlap}.")
    _require_columns(df, [input1, input2, *responses])
    _reject_derived_collisions(df, responses)

    work = df.copy()
    filtered, match_counts = exclude_rows(work, list(exclude or ()))
    n_excluded = len(work) - len(filtered)
    print(
        f"Rows: {len(work)} in, {n_excluded} excluded, {len(filtered)} kept."
    )
    for (column, raw), n_matched in match_counts.items():
        if n_matched == 0:
            print(f"Warning: exclude pair {column}={raw} matched 0 rows.")
    if filtered.empty:
        raise ValueError("No rows left after exclusions.")
    filtered = filtered.reset_index(drop=True)

    filtered[input1], control1 = _prepare_input(filtered[input1], control_raw[0], input1)
    filtered[input2], control2 = _prepare_input(filtered[input2], control_raw[1], input2)
    n_missing_input = int(filtered[[input1, input2]].isna().any(axis=1).sum())
    if n_missing_input:
        print(f"Dropped {n_missing_input} row(s) with missing input values.")
        filtered = filtered.dropna(subset=[input1, input2]).copy()
    if filtered.empty:
        raise ValueError("No rows left after dropping missing input values.")

    results: dict[str, dict[str, pd.DataFrame]] = {}
    stems: dict[str, str] = {}
    for response in responses:
        stem = _file_stem(response)
        if stem in stems.values():
            raise ValueError(
                f"Response names {response!r} and "
                f"{[k for k, v in stems.items() if v == stem]!r} "
                "produce the same output filename."
            )
        stems[response] = stem
        filtered[response] = _prepare_response(filtered[response], response)
        results[response] = _score_response(
            filtered, input1, input2, response, control1, control2
        )
        _attach_row_metrics(filtered, results[response].pop("row_metrics"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    row_out = output_path / "row_level.csv"
    filtered.to_csv(row_out, index=False)
    print(f"Wrote {row_out} ({len(filtered)} rows)")

    for response, stem in stems.items():
        _write_response(
            output_path,
            stem,
            response,
            input1,
            input2,
            control1,
            control2,
            results[response],
        )
    print(f"Outputs written to: {output_path.resolve()}")
    return results


def _score_response(
    df: pd.DataFrame,
    input1: str,
    input2: str,
    response: str,
    control1: Any,
    control2: Any,
) -> dict[str, pd.DataFrame]:
    usable = df.loc[df[response].notna(), [input1, input2, response]].copy()
    if usable.empty:
        raise ValueError(f"No non-missing values for response {response!r}.")

    control_mask = usable[input1].eq(control1) & usable[input2].eq(control2)
    controls = usable.loc[control_mask, response]
    if controls.empty:
        raise ValueError(
            f"No control rows for {response!r} where "
            f"{input1}={control1!r} and {input2}={control2!r}. "
            f"{input1} levels: {_level_list(usable[input1])}. "
            f"{input2} levels: {_level_list(usable[input2])}."
        )
    control_mean = float(controls.mean())
    if not np.isfinite(control_mean) or control_mean <= 0:
        raise ValueError(
            f"Control mean for {response!r} must be finite and positive, "
            f"got {control_mean}."
        )

    usable["viability"] = usable[response] / control_mean
    usable["inhibition"] = 1.0 - usable["viability"]

    observed = _mean_matrix(usable, input1, input2, "viability")
    _require_reference_edges(observed, input1, input2, control1, control2)
    expected, excess = _bliss_matrices(observed, input1, input2, control1, control2)

    dose_summary = (
        usable.groupby([input1, input2], as_index=False)
        .agg(
            n=(response, "size"),
            response_mean=(response, "mean"),
            response_std=(response, "std"),
            viability_mean=("viability", "mean"),
            viability_std=("viability", "std"),
            inhibition_mean=("inhibition", "mean"),
            inhibition_std=("inhibition", "std"),
        )
        .sort_values([input1, input2], kind="mergesort")
    )
    dose_summary.insert(0, "response_var", response)
    dose_summary = dose_summary.merge(
        _align_dose_keys(
            _matrix_to_long(expected, "bliss_expected_viability", input1, input2),
            dose_summary,
            input1,
            input2,
        ),
        on=[input1, input2],
        how="left",
    )
    dose_summary = dose_summary.merge(
        _align_dose_keys(
            _matrix_to_long(excess, "bliss_excess", input1, input2),
            dose_summary,
            input1,
            input2,
        ),
        on=[input1, input2],
        how="left",
    )

    summary = _summarize_excess(
        excess,
        response=response,
        input1=input1,
        input2=input2,
        control1=control1,
        control2=control2,
        control_mean=control_mean,
        n_rows=len(usable),
        n_control_rows=int(control_mask.sum()),
    )
    row_metrics = pd.DataFrame(
        {
            f"{response}_control_mean": control_mean,
            f"{response}_viability": usable["viability"],
            f"{response}_inhibition": usable["inhibition"],
        },
        index=usable.index,
    )
    return {
        "row_metrics": row_metrics,
        "dose_summary": dose_summary,
        "observed": observed,
        "expected": expected,
        "excess": excess,
        "summary": summary,
    }


def _attach_row_metrics(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    for column in metrics.columns:
        if column.endswith("_control_mean"):
            df[column] = metrics[column].iloc[0]
            continue
        df[column] = np.nan
        df.loc[metrics.index, column] = metrics[column]


def _write_response(
    output_dir: Path,
    stem: str,
    response: str,
    input1: str,
    input2: str,
    control1: Any,
    control2: Any,
    result: Mapping[str, pd.DataFrame],
) -> None:
    dose_summary = result["dose_summary"]
    observed = result["observed"]
    expected = result["expected"]
    excess = result["excess"]
    summary = result["summary"]

    dose_summary.to_csv(output_dir / f"{stem}_dose_summary_bliss.csv", index=False)
    observed.to_csv(output_dir / f"{stem}_matrix_observed.csv")
    expected.to_csv(output_dir / f"{stem}_matrix_expected.csv")
    excess.to_csv(output_dir / f"{stem}_matrix_excess.csv")
    summary.to_csv(output_dir / f"{stem}_bliss_summary.csv", index=False)

    print(f"\n{response}")
    print("Bliss excess = expected viability - observed viability (>0 synergy, <0 antagonism)")
    print(summary.to_string(index=False))

    _save_heatmap(
        observed,
        f"Observed viability ({response} / control)",
        output_dir / f"{stem}_heatmap_observed.png",
        input1,
        input2,
        cmap="YlGnBu",
        vmin=min(0.0, float(np.nanmin(observed.to_numpy(dtype=float)))),
        vmax=max(1.0, float(np.nanmax(observed.to_numpy(dtype=float)))),
        cbar_label="Viability",
    )
    _save_heatmap(
        expected,
        "Bliss-expected viability (E1 x E2)",
        output_dir / f"{stem}_heatmap_expected.png",
        input1,
        input2,
        cmap="YlGnBu",
        vmin=min(0.0, float(np.nanmin(expected.to_numpy(dtype=float)))),
        vmax=max(1.0, float(np.nanmax(expected.to_numpy(dtype=float)))),
        cbar_label="Expected viability",
    )
    abs_max = float(np.nanmax(np.abs(excess.to_numpy(dtype=float))))
    abs_max = max(abs_max, 0.05)
    _save_heatmap(
        excess,
        "Bliss excess (expected - observed)\n>0 synergy, <0 antagonism",
        output_dir / f"{stem}_heatmap_excess.png",
        input1,
        input2,
        cmap="RdBu_r",
        center=0.0,
        vmin=-abs_max,
        vmax=abs_max,
        cbar_label="Bliss excess",
    )
    _save_single_agent_plot(
        dose_summary,
        input1,
        input2,
        control1,
        control2,
        response,
        output_dir / f"{stem}_single_agent_dose_response.png",
    )


def _mean_matrix(df: pd.DataFrame, input1: str, input2: str, value: str) -> pd.DataFrame:
    matrix = (
        df.groupby([input1, input2], as_index=False)[value]
        .mean()
        .pivot(index=input1, columns=input2, values=value)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    matrix.index.name = input1
    matrix.columns.name = input2
    return matrix


def _require_reference_edges(
    matrix: pd.DataFrame,
    input1: str,
    input2: str,
    control1: Any,
    control2: Any,
) -> None:
    if control1 not in matrix.index:
        raise ValueError(
            f"No {input1}={control1!r} reference. Levels: {list(matrix.index)}"
        )
    if control2 not in matrix.columns:
        raise ValueError(
            f"No {input2}={control2!r} reference. Levels: {list(matrix.columns)}"
        )
    missing1 = matrix.index[matrix[control2].isna()].tolist()
    missing2 = matrix.columns[matrix.loc[control1].isna()].tolist()
    if missing1 or missing2:
        raise ValueError(
            "Bliss reference edges are incomplete. "
            f"Missing {input1} alone ({input2}={control2!r}) at {input1}={missing1}. "
            f"Missing {input2} alone ({input1}={control1!r}) at {input2}={missing2}."
        )


def _bliss_matrices(
    observed: pd.DataFrame,
    input1: str,
    input2: str,
    control1: Any,
    control2: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(expected, excess)`` from a viability matrix.

    ``expected - observed`` is positive when viability is lower than
    Bliss independence.
    """
    effect1 = observed[control2].astype(float)
    effect2 = observed.loc[control1].astype(float)
    expected = pd.DataFrame(
        np.outer(effect1.to_numpy(dtype=float), effect2.to_numpy(dtype=float)),
        index=observed.index,
        columns=observed.columns,
    )
    expected.index.name = input1
    expected.columns.name = input2
    excess = expected - observed
    excess.index.name = input1
    excess.columns.name = input2
    # Single-agent edges match by construction; clear float dust.
    excess.loc[control1, :] = 0.0
    excess.loc[:, control2] = 0.0
    return expected, excess


def _summarize_excess(
    excess: pd.DataFrame,
    *,
    response: str,
    input1: str,
    input2: str,
    control1: Any,
    control2: Any,
    control_mean: float,
    n_rows: int,
    n_control_rows: int,
) -> pd.DataFrame:
    """Summarize combination doses only (single-agent edges excluded)."""
    combo = excess.copy()
    combo.loc[control1, :] = np.nan
    combo.loc[:, control2] = np.nan
    values = combo.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError(
            f"No combination doses for {response!r}. "
            "Bliss excess needs rows where both inputs differ from their controls."
        )
    return pd.DataFrame(
        [
            {
                "response_var": response,
                "input1": input1,
                "input2": input2,
                "control1": control1,
                "control2": control2,
                "control_mean": control_mean,
                "n_rows": n_rows,
                "n_control_rows": n_control_rows,
                "n_combo_doses": int(values.size),
                "mean_bliss_excess": float(np.mean(values)),
                "median_bliss_excess": float(np.median(values)),
                "min_bliss_excess": float(np.min(values)),
                "max_bliss_excess": float(np.max(values)),
                "sum_bliss_excess": float(np.sum(values)),
                "frac_synergistic": float(np.mean(values > 0)),
                "frac_antagonistic": float(np.mean(values < 0)),
            }
        ]
    )


def _matrix_to_long(
    matrix: pd.DataFrame, value_name: str, input1: str, input2: str
) -> pd.DataFrame:
    named = matrix.copy()
    named.index.name = input1
    named.columns.name = input2
    return named.reset_index().melt(
        id_vars=input1, var_name=input2, value_name=value_name
    )


def _save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    input1: str,
    input2: str,
    cmap: str,
    cbar_label: str,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel(input2)
    ax.set_ylabel(input1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_single_agent_plot(
    dose_summary: pd.DataFrame,
    input1: str,
    input2: str,
    control1: Any,
    control2: Any,
    response: str,
    path: Path,
) -> None:
    alone1 = dose_summary.loc[dose_summary[input2].eq(control2)].sort_values(input1)
    alone2 = dose_summary.loc[dose_summary[input1].eq(control1)].sort_values(input2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    _errorbar_axis(axes[0], alone1, input1, f"{input1} alone")
    _errorbar_axis(axes[1], alone2, input2, f"{input2} alone")
    for ax in axes:
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_ylabel("Viability")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Single-agent dose responses ({response})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _errorbar_axis(ax: plt.Axes, frame: pd.DataFrame, x_col: str, label: str) -> None:
    yerr = frame["viability_std"].fillna(0.0)
    x_values = frame[x_col]
    if pd.api.types.is_numeric_dtype(x_values):
        x_plot: Any = x_values
    else:
        x_plot = np.arange(len(frame))
        ax.set_xticks(x_plot)
        ax.set_xticklabels(x_values.astype(str))
    ax.errorbar(
        x_plot,
        frame["viability_mean"],
        yerr=yerr,
        marker="o",
        capsize=3,
        label=label,
    )
    ax.set_xlabel(x_col)
    ax.legend()


def _align_dose_keys(
    long: pd.DataFrame,
    dose_summary: pd.DataFrame,
    input1: str,
    input2: str,
) -> pd.DataFrame:
    """Cast melted dose keys back to the summary dtypes so the merge matches."""
    aligned = long.copy()
    for key in (input1, input2):
        if pd.api.types.is_numeric_dtype(dose_summary[key]):
            aligned[key] = pd.to_numeric(aligned[key])
        else:
            aligned[key] = aligned[key].astype(dose_summary[key].dtype)
    return aligned


def _prepare_input(series: pd.Series, control_raw: Any, name: str) -> tuple[pd.Series, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    numeric_non_missing = int(numeric.notna().sum())
    if non_missing == 0:
        raise ValueError(f"Input column {name!r} has no values.")
    if numeric_non_missing == non_missing:
        try:
            control = float(control_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Input column {name!r} is numeric, but control value "
                f"{control_raw!r} is not."
            ) from exc
        values = numeric.astype(float)
        close = np.isclose(
            values.to_numpy(dtype=float),
            control,
            rtol=_CONTROL_RTOL,
            atol=_CONTROL_ATOL,
            equal_nan=False,
        )
        return values.mask(close, control), control

    text = series.astype("string").str.strip()
    control = str(control_raw).strip()
    return text.astype(object).where(text.notna(), np.nan), control


def _prepare_response(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if int(invalid.sum()):
        raise ValueError(
            f"Response {name!r} has {int(invalid.sum())} non-numeric value(s)."
        )
    values = numeric.astype(float)
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"Response {name!r} contains infinite values.")
    return values


def _equal_mask(series: pd.Series, raw: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        mapped = {
            "1": True,
            "true": True,
            "t": True,
            "yes": True,
            "0": False,
            "false": False,
            "f": False,
            "no": False,
        }.get(raw.strip().lower())
        if mapped is None:
            return pd.Series(False, index=series.index)
        return series.eq(mapped).fillna(False)

    try:
        numeric_raw = float(raw)
        raw_is_numeric = True
    except (TypeError, ValueError):
        numeric_raw = None
        raw_is_numeric = False

    numeric = pd.to_numeric(series, errors="coerce")
    column_is_numeric = pd.api.types.is_numeric_dtype(series) or (
        raw_is_numeric
        and int(numeric.notna().sum()) == int(series.notna().sum())
        and int(series.notna().sum()) > 0
    )
    if raw_is_numeric and column_is_numeric:
        close = np.isclose(
            numeric.to_numpy(dtype=float),
            float(numeric_raw),
            rtol=_CONTROL_RTOL,
            atol=_CONTROL_ATOL,
            equal_nan=False,
        )
        return pd.Series(close, index=series.index)

    text = series.astype("string").str.strip()
    return text.eq(raw.strip()).fillna(False)


def _two_names(values: Sequence[str], label: str) -> tuple[str, str]:
    names = tuple(values)
    if len(names) != 2:
        raise ValueError(f"{label} must contain exactly two columns, got {names!r}.")
    if names[0] == names[1]:
        raise ValueError(f"{label} columns must be different, got {names[0]!r} twice.")
    return names[0], names[1]


def _two_values(values: Sequence[Any], label: str) -> tuple[Any, Any]:
    items = tuple(values)
    if len(items) != 2:
        raise ValueError(f"{label} must contain exactly two values, got {items!r}.")
    return items[0], items[1]


def _response_names(values: Sequence[str]) -> list[str]:
    names = [str(name) for name in values]
    if not names:
        raise ValueError("response_vars must contain at least one column.")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate response variables: {names!r}.")
    return names


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Columns: {list(df.columns)}"
        )
    duplicated = [
        column for column in columns if not isinstance(df[column], pd.Series)
    ]
    if duplicated:
        raise ValueError(f"Duplicated columns: {duplicated}.")


def _reject_derived_collisions(df: pd.DataFrame, responses: Sequence[str]) -> None:
    taken = []
    for response in responses:
        for suffix in ("_viability", "_inhibition", "_control_mean"):
            name = f"{response}{suffix}"
            if name in df.columns:
                taken.append(name)
    if taken:
        raise ValueError(
            "Derived columns already exist in the input: "
            + ", ".join(taken)
            + ". Rename them or choose different response columns."
        )


def _file_stem(name: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return stem.strip("._") or "response"


def _level_list(series: pd.Series) -> list[Any]:
    levels = series.dropna().unique().tolist()
    try:
        return sorted(levels)
    except TypeError:
        return levels
