"""
fcs_matrixfit.py
================
Correlation modelling by linear decomposition -- the "matrix method".

Where fcs_fit takes a model with a handful of named parameters and searches
for the values that minimise the residual, this module fixes a large basis of
candidate curves up front and asks only how much of each is present.  The
answer comes from one linear solve, with no initial guesses, no bounds, no
iteration and no possibility of converging to a local minimum.

    G  ~  A d        A : (m, n) basis, one column per candidate decay time
                     d : (n,) amplitudes, the thing being fitted

The amplitudes are constrained to d >= 0 on physical grounds, except for the
final baseline column which is free to take either sign.  The spread of d
across the grid is the point of the method: a result concentrated on a few
neighbouring columns says the decay time is well determined, a broad one says
it is not, without either answer having been presupposed by the choice of
model.

The linear algebra and the basis itself live in fcs_linsolve.  This module is
the analysis layer: reading correlation CSVs, deciding what sigma means,
running one solve per dataset, plotting, and writing the exports.

Each dataset produces TWO figures -- the curve with its residuals, and the
amplitude distribution -- because they are functions of different quantities
and must not be forced onto a shared axis.  Both go through
fcs_plottools.show_figure, so each carries the standard plot-controls panel.

Entry point
-----------
    run_matrix_dialog(fcs_data, parent, workspace_order)

reached from Model Data > "Correlation (Matrix Method)".

Outputs
-------
One timestamped folder per run, under the same ``fits/`` directory the other
fitters use, containing:

    <stem>_report.txt          human-readable summary, one block per dataset
    <stem>_curves.csv          tau, G, sigma, fit, residuals   (long format)
    <stem>_distribution.csv    tau_char, d, d_err per basis vector (long format)
    <stem>_summary.csv/.xlsx   one row of scalars per dataset

The distribution is kept in its own file rather than folded into the summary
table: with a fine grid there are hundreds of amplitudes per dataset, and a
parameter table one row per dataset cannot hold them without becoming
unreadable.  Pairing each amplitude with its characteristic time in the same
file is what makes the distribution directly plottable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import fcs_export
import fcs_linsolve
import fcs_plottools
from fcs_fitcommon import new_fit_dir, slug_name, write_params_table


# ── Module-level settings ─────────────────────────────────────────────────────

# Characteristic lag time, in seconds, of the basis column at taufactor = 1.
# This is the anchor the taufactor grid is expressed against; fcs_linsolve's
# basis constants place the column's two component times either side of it.
_TAUFACTOR_ANCHOR_S = 1.0e-6

# Lag times whose sigma is missing, zero or negative cannot be weighted and are
# dropped from the fit.  Fewer than this many surviving points is not a fit.
_MIN_POINTS = 8


# ── Sigma handling ────────────────────────────────────────────────────────────

def resolve_sigma(sigma, meta, weighted=True):
    """Decide what uncertainty to use for one correlation curve.

    fcs_corr writes the ``G_std_definition`` header key only when ``G_std``
    holds a standard error of the mean.  Its absence means the file predates
    that convention and the column holds the POPULATION sigma over segments,
    larger by sqrt(n_segments).  Both are in circulation under the same column
    name, so the header key is the only thing that distinguishes them.

    Converting to a standard error puts chi2_red on a scale where 1 means "fits
    to within the noise", and puts the amplitude errors on an absolute scale,
    since this module does not rescale them by sqrt(chi2_red) when a sigma is
    supplied.

    Returns
    -------
    (sigma_out, note) : (ndarray or None, str)
        ``sigma_out`` is None when unweighted or when no usable sigma exists.
    """
    if not weighted or sigma is None:
        return None, "unweighted"

    sig = np.asarray(sigma, dtype=float).ravel()
    if not np.any(np.isfinite(sig) & (sig > 0.0)):
        return None, "no usable G_std column; fitting unweighted"

    meta = meta or {}
    if "G_std_definition" in meta:
        return sig, "G_std is the standard error of the mean (header flag present)"

    n_seg = _meta_int(meta, "n_segments", 1)
    if n_seg > 1:
        return (sig / np.sqrt(float(n_seg)),
                "G_std predates the G_std_definition flag, so it is the "
                "population sigma; divided by sqrt(%d segments) to give the "
                "standard error" % n_seg)
    return sig, ("G_std predates the G_std_definition flag; n_segments is 1 or "
                 "absent, so it is used as given")


def _meta_int(meta, key, default):
    try:
        return int(float(str(meta.get(key, default)).strip()))
    except (TypeError, ValueError):
        return default


# ── Core fit ──────────────────────────────────────────────────────────────────

def fit_dataset(ds, tf_min=None, tf_max=None, per_decade=None,
                weighted=True, include_baseline=True):
    """Decompose one correlation curve onto the basis.

    Parameters
    ----------
    ds : dict
        As produced by fcs_fit's correlation dataset source: keys ``name``,
        ``path``, ``meta``, ``tau`` (SECONDS), ``G``, ``sigma``.
    tf_min, tf_max, per_decade : optional
        Taufactor grid; defaults come from fcs_linsolve's module constants,
        which fcs_prefs can override.
    weighted : bool
        Use the curve's G_std, resolved to a standard error.
    include_baseline : bool
        Fit a free constant column alongside the basis.

    Returns
    -------
    dict with the solve result, the grid, the surviving data, and the summary
    scalars.  Never raises for ordinary data problems; check ``notes``.
    """
    notes = []

    tau = np.asarray(ds["tau"], dtype=float).ravel()
    G = np.asarray(ds["G"], dtype=float).ravel()
    if tau.size != G.size:
        raise ValueError("%s: tau has %d points but G has %d"
                         % (ds.get("name", "?"), tau.size, G.size))

    sigma, sigma_note = resolve_sigma(ds.get("sigma"), ds.get("meta"), weighted)
    notes.append(sigma_note)

    # ---- drop unusable lags -------------------------------------------------
    keep = np.isfinite(tau) & np.isfinite(G) & (tau > 0.0)
    if sigma is not None:
        keep &= np.isfinite(sigma) & (sigma > 0.0)
    n_dropped = int(np.sum(~keep))
    if n_dropped:
        notes.append("dropped %d of %d lag points with a missing or "
                     "non-positive tau, G or sigma" % (n_dropped, tau.size))
    if int(np.sum(keep)) < _MIN_POINTS:
        raise ValueError("%s: only %d usable lag points, need at least %d"
                         % (ds.get("name", "?"), int(np.sum(keep)), _MIN_POINTS))

    tau_k = tau[keep]
    G_k = G[keep]
    sig_k = None if sigma is None else sigma[keep]

    # ---- basis --------------------------------------------------------------
    grid = fcs_linsolve.log_taufactor_grid(tf_min, tf_max, per_decade)
    n_tau = grid.size
    # tau is in seconds: load_correlation_csv converts a tau_ms column on read,
    # so the units are known here and passed explicitly rather than defaulted.
    A = fcs_linsolve.build_basis(tau_k, grid, t_units="s",
                                 include_baseline=include_baseline)
    mask = fcs_linsolve.basis_nonneg_mask(n_tau, include_baseline)

    result = fcs_linsolve.solve_linear(G_k, A, sigma=sig_k, nonneg=mask,
                                       rescale_errors="auto")
    notes.extend(result.warnings)

    # ---- summary scalars ----------------------------------------------------
    amp = result.d[:n_tau]
    amp_err = result.d_err[:n_tau]
    tau_char = grid * _TAUFACTOR_ANCHOR_S

    baseline = float(result.d[n_tau]) if include_baseline else 0.0
    baseline_err = float(result.d_err[n_tau]) if include_baseline else float("nan")

    # G0 and <N> use the BASIS amplitudes only.  The baseline is an additive
    # offset of the curve, not a correlating population, and folding it into
    # the amplitude sum would bias <N> in whichever direction it happened to
    # take -- upward for a negative baseline, which is the common case.
    G0 = float(np.sum(amp))
    N = 1.0 / G0 if G0 > 0.0 else float("nan")

    if G0 > 0.0:
        w = amp / G0
        log_tau = np.log10(tau_char)
        mean_log = float(np.sum(w * log_tau))
        var_log = float(np.sum(w * (log_tau - mean_log) ** 2))
        tau_mean = 10.0 ** mean_log
        width_dec = float(np.sqrt(max(var_log, 0.0)))
    else:
        tau_mean = float("nan")
        width_dec = float("nan")
        notes.append("total basis amplitude is not positive; <N> and the "
                     "distribution moments are undefined")

    live = int(np.sum(amp > 0.0))

    return {
        "name": ds.get("name", "?"),
        "path": ds.get("path"),
        "meta": ds.get("meta", {}),
        "tau": tau_k, "G": G_k, "sigma": sig_k,
        "G_fit": result.G_fit,
        "residual": result.residuals,
        "residual_weighted": result.residuals_weighted,
        "taufactor": grid, "tau_char": tau_char,
        "d": amp, "d_err": amp_err,
        "pinned": result.pinned[:n_tau],
        "baseline": baseline, "baseline_err": baseline_err,
        "include_baseline": include_baseline,
        "G0": G0, "N": N, "tau_mean": tau_mean, "width_decades": width_dec,
        "n_live": live, "n_basis": n_tau,
        "chi2": result.chi2, "chi2_red": result.chi2_red, "dof": result.dof,
        "cond": result.cond, "rank": result.rank,
        "weighted": sig_k is not None,
        "error_rescaled": result.error_rescaled,
        "n_dropped": n_dropped,
        "notes": notes,
        "result": result,
    }


def fit_datasets(datasets, **kw):
    """Run :func:`fit_dataset` over a list, one independent solve each.

    Each curve gets its own d.  A shared-d fit across datasets is a different
    problem and is deliberately not attempted here.
    """
    out = []
    for ds in datasets:
        out.append(fit_dataset(ds, **kw))
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────
#
# TWO figures, not one.  The curve and its residuals genuinely share an axis --
# both are functions of lag time -- so they belong in one figure with a real
# sharex link.  The amplitude distribution is a function of a DIFFERENT
# quantity: the characteristic time a basis column stands for, which spans the
# taufactor grid rather than the measured lag range.  Stacking it under the
# curve in one gridspec aligned the two axes visually while their limits
# differed, which reads as a shared axis and is not one.  Separate figures also
# mean each gets its own plot-controls panel.
#
# Both x axes are in MILLISECONDS, matching every other correlation figure in
# the suite (fcs_corr.plot_correlation, fcs_fit.plot_global_fit), so the
# distribution can be read directly against the curve above it.

def plot_matrix_curve(fit, show=True):
    """Correlation curve with the matrix fit, over a residual strip.

    Returns (fig, axes) with axes = array([ax, axr]).
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8.0, 6.0))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    tau_ms = fit["tau"] * 1e3
    if fit["sigma"] is not None:
        ax.errorbar(tau_ms, fit["G"], yerr=fit["sigma"], fmt="o", ms=3,
                    color="0.35", ecolor="0.75", elinewidth=0.8, capsize=0,
                    label="G (data)", zorder=2)
    else:
        ax.plot(tau_ms, fit["G"], "o", ms=3, color="0.35", label="G (data)",
                zorder=2)
    ax.plot(tau_ms, fit["G_fit"], "-", lw=1.8, color="#c0392b",
            label="A·d (matrix fit)", zorder=3)
    ax.set_xscale("log")
    ax.set_ylabel("G(τ)", fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    fcs_plottools.adaptive_legend(ax, base_fontsize=9, loc="upper right")
    ax.set_title("Matrix method — %s\n%d basis vectors  ·  %d non-zero  ·  "
                 "χ²/dof = %.4g"
                 % (fit["name"], fit["n_basis"], fit["n_live"],
                    fit["chi2_red"]), fontsize=10)
    plt.setp(ax.get_xticklabels(), visible=False)

    res = fit["residual_weighted"] if fit["weighted"] else fit["residual"]
    axr.axhline(0.0, color="grey", linewidth=0.8)
    axr.plot(tau_ms, res, "o", ms=2.5, color="#2c6fbb")
    axr.set_xscale("log")
    axr.set_xlabel("Lag time τ (ms)", fontsize=12)
    axr.set_ylabel("resid/σ" if fit["weighted"] else "resid", fontsize=10)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    axr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    axes = np.array([ax, axr])
    if show:
        fcs_plottools.show_figure(fig, axes)
    return fig, axes


def plot_matrix_distribution(fit, show=True):
    """Recovered amplitudes against the characteristic time of each column.

    Its own figure: the x axis is the taufactor grid, not the measured lag
    range, and the two must not be forced to share limits.

    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.0, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    tau_ms = fit["tau_char"] * 1e3
    d = fit["d"]
    ax.vlines(tau_ms, 0.0, d, color="#2c6fbb", lw=1.2)
    live = d > 0.0
    if np.any(live):
        ax.plot(tau_ms[live], d[live], "o", ms=3.5, color="#2c6fbb",
                label="amplitude d")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(tau_ms[0] / 1.5, tau_ms[-1] * 1.5)
    ax.set_xlabel("Characteristic time (ms)", fontsize=12)
    ax.set_ylabel("Amplitude d", fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    fcs_plottools.adaptive_legend(ax, base_fontsize=9, loc="upper right")
    ax.set_title("Amplitude distribution — %s" % fit["name"], fontsize=10)

    txt = ("Σd = %.4g    ⟨N⟩ = %.4g\nmean τ = %.4g ms    width = %.3g dec\n"
           "baseline = %.4g    cond = %.3g"
           % (fit["G0"], fit["N"], fit["tau_mean"] * 1e3,
              fit["width_decades"], fit["baseline"], fit["cond"]))
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


def plot_matrix_fit(fit, show=True):
    """Both figures for one dataset.

    Returns ((curve_fig, curve_axes), (dist_fig, dist_ax)).
    """
    return (plot_matrix_curve(fit, show=show),
            plot_matrix_distribution(fit, show=show))


# ── Export ────────────────────────────────────────────────────────────────────

def _stem(name_override=None, when=None):
    """Shared stem for one run's output files: timestamp, then optional label.

    Same order as fcs_fitcommon.new_fit_dir, so a directory listing sorts
    chronologically and a file moved out of its folder still says when it was
    made.
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = slug_name(name_override)
    return "%s_%s" % (stamp, slug) if slug else "matrixfit_%s" % stamp


def _write_curves(path, fits):
    """Long-format curve CSV: every dataset stacked, distinguished by name."""
    cv = fcs_export.csv_value
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write("# FCS matrix-method fit — curves\n")
        fh.write("# written : %s\n" % datetime.now().isoformat(timespec="seconds"))
        fh.write("# datasets : %d\n" % len(fits))
        fh.write("# note : long format; filter on the dataset column\n")
        fh.write("dataset,tau_s,G,sigma,G_fit,residual,residual_weighted\n")
        for f in fits:
            nm = f["name"].replace(",", "_")
            sig = f["sigma"]
            for i in range(f["tau"].size):
                fh.write("%s,%s,%s,%s,%s,%s,%s\n" % (
                    nm,
                    cv(f["tau"][i]), cv(f["G"][i]),
                    cv(sig[i] if sig is not None else np.nan),
                    cv(f["G_fit"][i]), cv(f["residual"][i]),
                    cv(f["residual_weighted"][i])))
    print("[matrixfit] wrote %s" % path)
    return Path(path)


def _write_distribution(path, fits):
    """Long-format amplitude CSV: d paired with the time each column stands for.

    tau_char_s is what makes this directly plottable -- the distribution is
    read as (tau_char_s, d), with taufactor kept alongside as the internal
    coordinate for anyone reconstructing the basis.
    """
    cv = fcs_export.csv_value
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write("# FCS matrix-method fit — amplitude distribution\n")
        fh.write("# written : %s\n" % datetime.now().isoformat(timespec="seconds"))
        fh.write("# taufactor anchor : taufactor 1 = %g s\n" % _TAUFACTOR_ANCHOR_S)
        fh.write("# note : plot d against tau_char_s; d_err is blank where the "
                 "amplitude is pinned at the d >= 0 bound\n")
        fh.write("dataset,index,taufactor,tau_char_s,t1_s,t2_s,d,d_err,pinned\n")
        for f in fits:
            nm = f["name"].replace(",", "_")
            t1, t2 = fcs_linsolve.basis_component_times(f["taufactor"])
            for i in range(f["d"].size):
                fh.write("%s,%d,%s,%s,%s,%s,%s,%s,%s\n" % (
                    nm, i,
                    cv(f["taufactor"][i]), cv(f["tau_char"][i]),
                    cv(t1[i]), cv(t2[i]),
                    cv(f["d"][i]), cv(f["d_err"][i]),
                    "yes" if f["pinned"][i] else "no"))
    print("[matrixfit] wrote %s" % path)
    return Path(path)


_SUMMARY_HEADER = [
    "dataset", "G0_sum_d", "N", "tau_mean_s", "width_decades",
    "baseline", "baseline_err", "chi2_red", "dof", "cond",
    "n_basis", "n_live", "n_lags", "n_dropped", "weighted",
]


def _summary_rows(fits):
    rows = []
    for f in fits:
        rows.append([
            f["name"], f["G0"], f["N"], f["tau_mean"], f["width_decades"],
            f["baseline"], f["baseline_err"], f["chi2_red"], f["dof"],
            f["cond"], f["n_basis"], f["n_live"], f["tau"].size,
            f["n_dropped"], "yes" if f["weighted"] else "no",
        ])
    return rows


def _report_lines(fits, grid_desc):
    L = []
    L.append("FCS matrix-method fit report")
    L.append("=" * 64)
    L.append("method     : linear decomposition onto a fixed basis, d >= 0")
    L.append("fitted     : %s" % datetime.now().isoformat(timespec="seconds"))
    L.append("datasets   : %d  (one independent solve each)" % len(fits))
    L.append("basis      : %s" % grid_desc)
    L.append("anchor     : taufactor 1 = %g s characteristic time"
             % _TAUFACTOR_ANCHOR_S)
    L.append("")
    L.append("G0 and <N> are computed from the BASIS amplitudes only; the")
    L.append("baseline column is an additive offset, not a correlating")
    L.append("population, and is excluded from both.")
    L.append("")
    for f in fits:
        L.append("-" * 64)
        L.append("dataset    : %s" % f["name"])
        if f.get("path") is not None:
            L.append("source     : %s" % f["path"])
        L.append("lags used  : %d  (%d dropped)" % (f["tau"].size, f["n_dropped"]))
        L.append("weighting  : %s" % ("per-point sigma" if f["weighted"]
                                      else "UNWEIGHTED"))
        L.append("chi2/dof   : %.6g   (dof %d)" % (f["chi2_red"], f["dof"]))
        L.append("cond       : %.4g   rank %d" % (f["cond"], f["rank"]))
        L.append("G0 = sum d : %.6g" % f["G0"])
        L.append("<N>        : %.6g" % f["N"])
        L.append("mean tau   : %.6g s" % f["tau_mean"])
        L.append("width      : %.4g decades" % f["width_decades"])
        if f["include_baseline"]:
            L.append("baseline   : %.6g +/- %.6g" % (f["baseline"],
                                                     f["baseline_err"]))
        L.append("non-zero   : %d of %d basis columns"
                 % (f["n_live"], f["n_basis"]))
        if f["notes"]:
            L.append("notes:")
            for nline in f["notes"]:
                L.append("  - %s" % nline)
    L.append("-" * 64)
    return L


def export_matrix_fit(fits, out_source, name_override=None, grid_desc=""):
    """Write report, curves, distribution and summary for one run.

    Returns (out_dir, report, curves, distribution, summary_csv).
    """
    out_dir = new_fit_dir(Path(out_source), name_override)
    stem = _stem(name_override)

    report_path = out_dir / ("%s_report.txt" % stem)
    report_path.write_text("\n".join(_report_lines(fits, grid_desc)) + "\n",
                           encoding="utf-8")
    print("[matrixfit] wrote %s" % report_path)

    curves = _write_curves(out_dir / ("%s_curves.csv" % stem), fits)
    dist = _write_distribution(out_dir / ("%s_distribution.csv" % stem), fits)

    summary_csv, _x = write_params_table(
        out_dir / ("%s_summary.csv" % stem),
        ["FCS matrix-method fit — summary",
         "method : linear decomposition onto a fixed basis, d >= 0",
         "basis : %s" % grid_desc,
         "anchor : taufactor 1 = %g s" % _TAUFACTOR_ANCHOR_S,
         "note : G0 and N exclude the baseline column",
         "amplitudes : see the companion _distribution.csv"],
        _SUMMARY_HEADER, _summary_rows(fits),
        log_tag="matrixfit", sheet_title="matrix fit summary")

    return out_dir, report_path, curves, dist, summary_csv


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_matrix_dialog(fcs_data=None, parent=None, workspace_order=None):
    """Entry point: select correlation CSVs, set the grid, solve, plot, export.

    Reuses fcs_fit's correlation dataset picker, so multi-select, workspace
    ordering and the browse-for-CSV fallback all behave exactly as they do for
    the ordinary correlation fit.
    """
    import fcs_fit

    init_dir = fcs_fit.default_dataset_dir(fcs_data)

    def _after_datasets(loaded):
        _setup_dialog(parent, loaded)

    fcs_fit.select_datasets_dialog(parent, init_dir, _after_datasets,
                                   order=workspace_order)


def _setup_dialog(parent, datasets):
    """Screen — grid span, weighting and baseline, then run."""
    import tkinter as tk
    from tkinter import messagebox

    win = tk.Toplevel(parent)
    win.title("Matrix method — setup")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Correlation — matrix method",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text="%d dataset%s; each is solved independently."
                       % (len(datasets), "" if len(datasets) == 1 else "s"),
             font=("Helvetica", 9), fg="grey").pack()

    grid_f = tk.LabelFrame(win, text="Basis", padx=10, pady=6)
    grid_f.pack(fill="x", padx=12, pady=(8, 4))

    # Prefilled from the fcs_linsolve constants, which fcs_prefs may already
    # have overridden; editable here for a one-off run without disturbing the
    # stored preference.
    rows = [("Shortest taufactor", "%g" % fcs_linsolve._TAUFACTOR_MIN),
            ("Longest taufactor", "%g" % fcs_linsolve._TAUFACTOR_MAX),
            ("Vectors per decade", "%d" % fcs_linsolve._TAUFACTOR_PER_DECADE)]
    entries = []
    for label, val in rows:
        r = tk.Frame(grid_f)
        r.pack(fill="x", pady=2)
        tk.Label(r, text=label + ":", anchor="w",
                 font=("Helvetica", 9), width=20).pack(side="left")
        var = tk.StringVar(master=win, value=val)
        tk.Entry(r, textvariable=var, width=12, justify="right",
                 font=("Helvetica", 9)).pack(side="right")
        entries.append(var)
    tf_min_var, tf_max_var, per_dec_var = entries

    count_lbl = tk.Label(grid_f, text="", font=("Helvetica", 8), fg="grey",
                         anchor="w")
    count_lbl.pack(fill="x")

    def _refresh_count(*_):
        try:
            g = fcs_linsolve.log_taufactor_grid(
                float(tf_min_var.get()), float(tf_max_var.get()),
                int(per_dec_var.get()))
        except (ValueError, TypeError):
            count_lbl.config(text="      (enter a valid span)")
            return
        count_lbl.config(
            text="      %d basis vectors, characteristic times %.3g to %.3g s"
                 % (g.size, g[0] * _TAUFACTOR_ANCHOR_S,
                    g[-1] * _TAUFACTOR_ANCHOR_S))

    for v in entries:
        v.trace_add("write", _refresh_count)
    _refresh_count()

    opt_f = tk.LabelFrame(win, text="Options", padx=10, pady=6)
    opt_f.pack(fill="x", padx=12, pady=4)

    have_sigma = all(ds.get("sigma") is not None
                     and np.isfinite(np.asarray(ds["sigma"], float)).any()
                     for ds in datasets)
    weight_var = tk.BooleanVar(master=win, value=have_sigma)
    tk.Checkbutton(opt_f, variable=weight_var, anchor="w",
                   state="normal" if have_sigma else "disabled",
                   text=("Weight by σ (G_std)" if have_sigma
                         else "Weight by σ — no G_std in every dataset")
                   ).pack(fill="x")

    base_var = tk.BooleanVar(master=win, value=True)
    tk.Checkbutton(opt_f, variable=base_var, anchor="w",
                   text="Fit a free baseline column").pack(fill="x")

    tk.Label(opt_f, text="      Amplitudes are constrained to d ≥ 0; the "
                         "baseline is not.",
             font=("Helvetica", 8), fg="grey", anchor="w",
             justify="left").pack(fill="x")

    name_f = tk.Frame(win)
    name_f.pack(fill="x", padx=12, pady=(4, 2))
    tk.Label(name_f, text="Label (optional):", anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    name_var = tk.StringVar(master=win, value="")
    tk.Entry(name_f, textvariable=name_var, width=22,
             font=("Helvetica", 9)).pack(side="right")

    export_var = tk.BooleanVar(master=win, value=True)
    tk.Checkbutton(win, variable=export_var, anchor="w",
                   text="Write report, curves, distribution and summary"
                   ).pack(fill="x", padx=16)

    def _run():
        try:
            tf_min = float(tf_min_var.get())
            tf_max = float(tf_max_var.get())
            per_dec = int(per_dec_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid input",
                                 "The taufactor span must be numeric and the "
                                 "count per decade a whole number.", parent=win)
            return
        try:
            grid = fcs_linsolve.log_taufactor_grid(tf_min, tf_max, per_dec)
        except ValueError as exc:
            messagebox.showerror("Invalid basis", str(exc), parent=win)
            return

        try:
            fits = fit_datasets(datasets, tf_min=tf_min, tf_max=tf_max,
                                per_decade=per_dec,
                                weighted=weight_var.get(),
                                include_baseline=base_var.get())
        except ValueError as exc:
            messagebox.showerror("Fit failed", str(exc), parent=win)
            return

        win.destroy()

        for f in fits:
            # Two windows per dataset: curve+residuals, and the distribution.
            plot_matrix_fit(f, show=True)

        if export_var.get():
            desc = ("%d vectors, taufactor %g to %g at %d per decade"
                    % (grid.size, tf_min, tf_max, per_dec))
            export_matrix_fit(fits, datasets[0]["path"],
                              name_var.get().strip() or None, desc)

    btns = tk.Frame(win)
    btns.pack(pady=10)
    tk.Button(btns, text="Solve", width=12, command=_run, pady=4
              ).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4
              ).pack(side="left", padx=6)

    win.wait_window()


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest():
    """Headless checks.  Run:  python fcs_matrixfit.py"""
    import tempfile

    rng = np.random.RandomState(4242)
    tau = np.logspace(-6, -1, 140)
    grid = fcs_linsolve.log_taufactor_grid(1.0, 1.0e3, 20)
    A = fcs_linsolve.build_basis(tau, grid, t_units="s")
    truth = np.zeros(grid.size + 1)
    truth[np.argmin(np.abs(grid * _TAUFACTOR_ANCHOR_S - 3.0e-5))] = 0.25
    truth[-1] = -0.002
    G_clean = A.dot(truth)
    n_seg = 50
    sem = 0.002 * (1.0 + np.abs(G_clean))
    G = G_clean + sem * rng.randn(tau.size)

    # --- sigma convention -----------------------------------------------------
    s_new, note_new = resolve_sigma(sem, {"G_std_definition": "sem", "n_segments": "50"})
    assert np.allclose(s_new, sem), "SEM files must be used as given"
    s_old, note_old = resolve_sigma(sem * np.sqrt(n_seg), {"n_segments": "50"})
    assert np.allclose(s_old, sem), "population sigma must be converted to SEM"
    assert "sqrt(50" in note_old, note_old
    s_none, note_none = resolve_sigma(None, {})
    assert s_none is None
    s_bad, _ = resolve_sigma(np.zeros_like(sem), {"G_std_definition": "sem"})
    assert s_bad is None, "an all-zero sigma column is not usable"
    print("sigma convention: SEM passthrough, population->SEM, absent, unusable")

    # --- a fit, both sigma conventions, must agree ---------------------------
    base = {"name": "synthetic", "path": None, "tau": tau, "G": G}
    f_new = fit_dataset(dict(base, sigma=sem,
                             meta={"G_std_definition": "sem", "n_segments": "50"}),
                        tf_min=1.0, tf_max=1.0e3, per_decade=20)
    f_old = fit_dataset(dict(base, sigma=sem * np.sqrt(n_seg),
                             meta={"n_segments": "50"}),
                        tf_min=1.0, tf_max=1.0e3, per_decade=20)
    assert np.allclose(f_new["d"], f_old["d"]), "same curve, different header"
    assert np.isclose(f_new["chi2_red"], f_old["chi2_red"])
    print("old- and new-convention headers give the same fit: chi2_red %.4f"
          % f_new["chi2_red"])

    # --- what the old path would have done -----------------------------------
    f_raw = fit_dataset(dict(base, sigma=sem * np.sqrt(n_seg), meta={}),
                        tf_min=1.0, tf_max=1.0e3, per_decade=20)
    ratio = f_raw["chi2_red"] / f_new["chi2_red"]
    print("  ignoring the convention scales chi2_red by %.4f (= 1/n_segments)"
          % ratio)
    assert 0.015 < ratio < 0.025
    # The amplitudes themselves do NOT move: scaling every sigma by the same
    # constant rescales the whole objective and leaves its minimum where it
    # was.  What moves is chi2_red, and -- because a supplied sigma is taken as
    # absolute and not rescaled by sqrt(chi2_red) -- the error bars, by exactly
    # that constant.
    assert np.allclose(f_raw["d"], f_new["d"]), "amplitudes are unaffected"
    live = ~f_new["pinned"] & ~f_raw["pinned"] & np.isfinite(f_new["d_err"])
    if np.any(live):
        err_ratio = float(np.median(f_raw["d_err"][live] / f_new["d_err"][live]))
        print("  and inflates the amplitude errors by %.3f (= sqrt(n_segments) "
              "= %.3f)" % (err_ratio, np.sqrt(n_seg)))
        assert abs(err_ratio - np.sqrt(n_seg)) / np.sqrt(n_seg) < 0.02

    # --- summary scalars ------------------------------------------------------
    assert np.all(f_new["d"] >= 0.0), "amplitudes must be non-negative"
    assert f_new["baseline"] < 0.0, "baseline must be free to go negative"
    assert np.isclose(f_new["G0"], np.sum(f_new["d"]))
    assert not np.isclose(f_new["G0"], np.sum(f_new["d"]) + f_new["baseline"]), \
        "the baseline must not be inside G0"
    assert np.isclose(f_new["N"], 1.0 / f_new["G0"])
    assert 1e-5 < f_new["tau_mean"] < 1e-4, f_new["tau_mean"]
    print("G0 %.5f (baseline %.5f excluded), <N> %.3f, mean tau %.4g s, "
          "width %.3g dec" % (f_new["G0"], f_new["baseline"], f_new["N"],
                              f_new["tau_mean"], f_new["width_decades"]))

    # --- N must not move when only the baseline does -------------------------
    shifted = dict(base, sigma=sem, meta={"G_std_definition": "sem"})
    shifted["G"] = G + 0.05
    f_shift = fit_dataset(shifted, tf_min=1.0, tf_max=1.0e3, per_decade=20)
    assert abs(f_shift["N"] - f_new["N"]) / f_new["N"] < 0.05, \
        "a pure offset must be absorbed by the baseline, not by <N>"
    print("a +0.05 offset moves the baseline to %.5f; <N> %.3f -> %.3f"
          % (f_shift["baseline"], f_new["N"], f_shift["N"]))

    # --- dropped lags ---------------------------------------------------------
    bad_sig = sem.copy()
    bad_sig[:5] = 0.0
    f_drop = fit_dataset(dict(base, sigma=bad_sig,
                              meta={"G_std_definition": "sem"}),
                         tf_min=1.0, tf_max=1.0e3, per_decade=20)
    assert f_drop["n_dropped"] == 5 and f_drop["tau"].size == tau.size - 5
    print("lags with an unusable sigma are dropped and reported: %d"
          % f_drop["n_dropped"])

    # --- exports --------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "data" / "analysis" / "g1_correlation_cross.csv"
        src.parent.mkdir(parents=True)
        src.touch()
        out_dir, rep, cur, dist, summ = export_matrix_fit(
            [f_new, f_old], src, "self test", "61 vectors, 1 to 1e3")
        assert out_dir.parent.name == "fits"
        for p in (rep, cur, dist, summ):
            assert p.exists() and p.stat().st_size > 0
        stems = {p.name.split("_report")[0].split("_curves")[0]
                 .split("_distribution")[0].split("_summary")[0]
                 for p in (rep, cur, dist, summ)}
        assert len(stems) == 1, "the four files must share one stem: %s" % stems

        dl = dist.read_text(encoding="utf-8-sig").splitlines()
        body = [l for l in dl if not l.startswith("#")]
        assert body[0].startswith("dataset,index,taufactor,tau_char_s")
        assert len(body) - 1 == 2 * f_new["d"].size
        cols = body[1].split(",")
        assert np.isclose(float(cols[3]), float(cols[2]) * _TAUFACTOR_ANCHOR_S)

        cl = [l for l in cur.read_text(encoding="utf-8-sig").splitlines()
              if not l.startswith("#")]
        assert cl[0] == "dataset,tau_s,G,sigma,G_fit,residual,residual_weighted"
        assert len(cl) - 1 == 2 * f_new["tau"].size

        sl = [l for l in summ.read_text(encoding="utf-8-sig").splitlines()
              if not l.startswith("#")]
        assert sl[0] == ",".join(_SUMMARY_HEADER)
        assert len(sl) - 1 == 2
        print("exports: report, curves (%d rows), distribution (%d rows), "
              "summary (2 rows), shared stem"
              % (len(cl) - 1, len(body) - 1))

    # --- plotting, headless ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (cfig, caxes), (dfig, dax) = plot_matrix_fit(f_new, show=False)
    assert len(caxes) == 2, "curve figure is curve + residual strip"
    assert cfig is not dfig, "the distribution must be its own figure"
    # The residual strip shares the lag axis; the distribution must NOT be
    # linked to it, in either direction.
    assert caxes[1] in caxes[0].get_shared_x_axes().get_siblings(caxes[0])
    assert dax not in caxes[0].get_shared_x_axes().get_siblings(caxes[0]), \
        "the distribution axis must not share x with the lag axis"
    assert caxes[0].get_xlabel() == "" and caxes[1].get_xlabel().endswith("(ms)")
    assert dax.get_xlabel().endswith("(ms)")
    # Limits are free to differ, which is the whole point of splitting them.
    print("  curve x-range  %.4g to %.4g ms" % caxes[0].get_xlim())
    print("  dist  x-range  %.4g to %.4g ms" % dax.get_xlim())
    plt.close(cfig)
    plt.close(dfig)
    print("two independent figures, lag axis in ms, no cross-figure x sharing")

    print("fcs_matrixfit: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
