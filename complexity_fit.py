"""Unified complexity-classification module.

Pure module: no tkinter, no matplotlib. Parses perf result .txt files,
fits a family of complexity models with a relative (log-log-equivalent)
least-squares criterion, selects the best model via BIC + hierarchical
bootstrap, and reports a Verdict that abstains when the data does not
support a confident classification.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr


# ============================================================
#  Parser
# ============================================================

@dataclass
class Measurement:
    folder: str
    width: int
    height: int
    n_pixels: int
    time_reps: list = field(default_factory=list)
    instr_reps: list = field(default_factory=list)
    cycles_reps: list = field(default_factory=list)
    cache_refs_reps: list = field(default_factory=list)
    cache_miss_reps: list = field(default_factory=list)


_FOLDER_RE = re.compile(r'=== FOLDER:\s*([^\n]+?)\s*(?:===)?\s*$')
_DIMENSIONS_RE = re.compile(r'^IMAGE_DIMENSIONS:\s+(\d+)x(\d+)')
_FOLDER_DIMS_RE = re.compile(r'^(\d+)x(\d+)$')
_WARNING_RE = re.compile(r'^WARNING:\s*(.+)$')

_PATTERNS = {
    'cycles':       re.compile(r'^\s*(\d[\d,]*)\s+cycles:u'),
    'instructions': re.compile(r'^\s*(\d[\d,]*)\s+instructions:u'),
    'cache_refs':   re.compile(r'^\s*(\d[\d,]*)\s+cache-references:u'),
    'cache_misses': re.compile(r'^\s*(\d[\d,]*)\s+cache-misses:u'),
    'time':         re.compile(r'^\s*([0-9]*\.[0-9]+)\s+seconds time elapsed'),
}


class PerfFileFormatError(Exception):
    """Raised when a file has none of the expected '=== FOLDER:' markers --
    not a recognized perf-results file at all."""


class PerfMeasurementsMissingError(Exception):
    """Raised when a file has the expected folder/repetition structure but
    zero actual measurements -- e.g. perf itself failed to run on the
    machine that generated the file, so every repetition only logged a
    warning instead of counters."""


# ------------------------------------------------------------
#  Protocolo v2: medicion por diferencia dentro del proceso
#
#  Por imagen se miden bloques con perf: LI (libs + carga de la imagen, sin
#  llamar a la funcion), L1 (LI + 1 llamada) y LK (LI + K llamadas, K >= 2).
#  El coste de UNA llamada en regimen (sin arranque, sin carga de la imagen ni
#  la init de la primera llamada) es (LK - L1) / (K - 1).
# ------------------------------------------------------------

PROTOCOL_MARK = "[PROTOCOL v2]"
_METRICS = ('cycles', 'instructions', 'cache_refs', 'cache_misses', 'time')
_BLOCK_RE = re.compile(r'^\[BLOCK\s+(\w+)\s+K=(\d+)\]')


def correct_warm(l1_reps: list, lk_reps: list, k: int) -> dict:
    """Per-call cost in steady state from paired L1/LK repetitions.

    l1_reps / lk_reps: lists of {metric: value} dicts (one per repetition).
    Returns {metric: [values]}. Repetitions are paired by index; a metric is
    only emitted for a repetition when both blocks have it AND the difference
    is positive (a non-positive difference is noise swamping the signal and
    would break the relative-error fit, so it is dropped rather than clamped).
    """
    out = {m: [] for m in _METRICS}
    if k < 2:
        return out
    for a, b in zip(l1_reps, lk_reps):
        for m in _METRICS:
            if m in a and m in b:
                v = (b[m] - a[m]) / (k - 1)
                if v > 0:
                    out[m].append(v)
    return out


def correct_cold(li_reps: list, l1_reps: list) -> dict:
    """Cost of ONE call (first call, cold) = L1 - LI.

    LI (libs + image, no call) is measured in a separate process each time, so
    its repetitions are not paired with L1's: the median of LI is subtracted
    from every L1 repetition. Non-positive differences are dropped (noise)."""
    out = {m: [] for m in _METRICS}
    for m in _METRICS:
        base = [r[m] for r in li_reps if m in r]
        if not base:
            continue
        b = float(np.median(base))
        for r in l1_reps:
            if m in r and r[m] - b > 0:
                out[m].append(r[m] - b)
    return out


def _collect_v2(lines: list, estimator: str = "cold"):
    """Read a protocol-v2 file. Returns (raw, dims_by_folder, found_folder_marker,
    first_warning) in the same shape as the legacy parser produces.

    estimator: "cold" -> cost of the first call, L1 - LI (default).
               "warm" -> steady-state cost, (LK - L1) / (K - 1); only valid if
                         every call of the loop costs the same (it does not if
                         the function caches or mutates its input)."""
    raw = {m: defaultdict(list) for m in _METRICS}
    dims_by_folder = defaultdict(list)
    images = {}  # (folder, image) -> {"LI": [metrics], "L1": [metrics], "LK": [metrics], "k": int}
    folder = image = label = None
    k = 0
    in_perf = False
    cur = {}
    found_folder_marker = False
    first_warning = None

    for raw_line in lines:
        line = raw_line.strip()
        if first_warning is None:
            wm = _WARNING_RE.match(line)
            if wm:
                first_warning = wm.group(1).strip()

        m = re.search(r'=== FOLDER:\s+([^\n]+)', line)
        if m:
            found_folder_marker = True
            folder = m.group(1).replace(' ===', '').strip()
            image = label = None
            in_perf = False
            continue
        m = re.search(r'--- IMAGE:\s+([^\n]+)', line)
        if m:
            image = m.group(1).replace(' ---', '').strip()
            label = None
            in_perf = False
            continue
        m = _BLOCK_RE.match(line)
        if m:
            label, k = m.group(1), int(m.group(2))
            in_perf = False
            continue
        if "Performance counter stats for" in line:
            in_perf = True
            cur = {}
            continue
        if in_perf and "seconds time elapsed" in line:
            tm = _PATTERNS['time'].search(line)
            if tm:
                cur['time'] = float(tm.group(1))
            if folder and image and label in ("LI", "L1", "LK") and 'instructions' in cur:
                rec = images.setdefault((folder, image), {"LI": [], "L1": [], "LK": [], "k": 0})
                rec[label].append(cur)
                if label == "LK":
                    rec["k"] = k
            in_perf = False
            continue
        if "IMAGE_DIMENSIONS:" in line and folder:
            dm = _DIMENSIONS_RE.search(line)
            if dm:
                dims_by_folder[folder].append((int(dm.group(1)), int(dm.group(2))))
            continue
        if in_perf:
            for key in ('cycles', 'instructions', 'cache_refs', 'cache_misses'):
                pm = _PATTERNS[key].search(line)
                if pm:
                    cur[key] = float(pm.group(1).replace(',', ''))

    for (fld, _img), rec in images.items():
        if estimator == "warm":
            corrected = correct_warm(rec["L1"], rec["LK"], rec["k"])
        else:
            corrected = correct_cold(rec["LI"], rec["L1"])
        for metric, vals in corrected.items():
            raw[metric][fld].extend(vals)
    return raw, dims_by_folder, found_folder_marker, first_warning


def parse_performance_file(txt_file: str, estimator: str = "cold") -> list:
    """Parse a perf results .txt file into a list of Measurement, sorted by n_pixels.

    Repetitions are kept raw (not averaged) so the bootstrap can resample them.

    Raises PerfFileFormatError if the file has no '=== FOLDER:' markers at all
    (unrecognized format), or PerfMeasurementsMissingError if it has that
    structure but no actual measurements were recorded in it (e.g. perf
    wasn't installed on the machine that produced the file) -- both are
    silent failures otherwise: an empty measurement list downstream just
    reads as "0 points after filtering", which doesn't point at the cause.

    estimator (protocol-v2 files only): "cold" = L1 - LI (default), "warm" =
    (LK - L1) / (K - 1). See _collect_v2.
    """
    raw = {k: defaultdict(list) for k in
           ['cycles', 'instructions', 'cache_refs', 'cache_misses', 'time']}
    dims_by_folder = defaultdict(list)

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_folder = None
    in_perf_block = False
    found_folder_marker = False
    first_warning = None

    legacy_lines = lines
    if any(PROTOCOL_MARK in l for l in lines):
        raw, dims_by_folder, found_folder_marker, first_warning = _collect_v2(lines, estimator)
        legacy_lines = []

    for raw_line in legacy_lines:
        line = raw_line.strip()

        if first_warning is None:
            wm = _WARNING_RE.match(line)
            if wm:
                first_warning = wm.group(1).strip()

        m = re.search(r'=== FOLDER:\s+([^\n]+)', line)
        if m:
            found_folder_marker = True
            current_folder = m.group(1).replace(' ===', '').strip()
            in_perf_block = False
            continue

        if "Performance counter stats for" in line:
            in_perf_block = True
            continue

        if in_perf_block and "seconds time elapsed" in line:
            m = _PATTERNS['time'].search(line)
            if m and current_folder:
                raw['time'][current_folder].append(float(m.group(1)))
            in_perf_block = False
            continue

        if "IMAGE_DIMENSIONS:" in line and current_folder:
            m = _DIMENSIONS_RE.search(line)
            if m:
                dims_by_folder[current_folder].append((int(m.group(1)), int(m.group(2))))
            continue

        if in_perf_block and current_folder:
            for key in ['cycles', 'instructions', 'cache_refs', 'cache_misses']:
                m = _PATTERNS[key].search(line)
                if m:
                    raw[key][current_folder].append(float(m.group(1).replace(',', '')))

    folders = set().union(*(set(raw[m].keys()) for m in raw))
    measurements = []
    for folder in folders:
        if dims_by_folder.get(folder):
            widths = [d[0] for d in dims_by_folder[folder]]
            heights = [d[1] for d in dims_by_folder[folder]]
            width = int(round(np.median(widths)))
            height = int(round(np.median(heights)))
        else:
            fm = _FOLDER_DIMS_RE.match(folder)
            if not fm:
                continue
            width, height = int(fm.group(1)), int(fm.group(2))

        measurements.append(Measurement(
            folder=folder,
            width=width,
            height=height,
            n_pixels=width * height,
            time_reps=raw['time'][folder],
            instr_reps=raw['instructions'].get(folder, []),
            cycles_reps=raw['cycles'].get(folder, []),
            cache_refs_reps=raw['cache_refs'].get(folder, []),
            cache_miss_reps=raw['cache_misses'].get(folder, []),
        ))

    measurements.sort(key=lambda m: m.n_pixels)

    if not measurements:
        if not found_folder_marker:
            raise PerfFileFormatError(
                f"unrecognized format: no '=== FOLDER:' markers "
                f"found in {txt_file}"
            )
        if first_warning:
            raise PerfMeasurementsMissingError(
                f"{txt_file} has the folder and repetition structure "
                f"but zero measurements; possible cause quoted in the file: "
                f"\"{first_warning}\""
            )
        raise PerfMeasurementsMissingError(
            f"{txt_file} has the folder and repetition structure but "
            "zero measurements (cycles/instructions/time elapsed); check that "
            "perf ran correctly when it was generated"
        )

    return measurements


# ============================================================
#  Shape filter
# ============================================================

class ShapeFilter(Enum):
    NONE = "none"                    # all points, heterogeneous
    SQUARE = "square"                # width == height
    SQUARE_NO_POW2 = "square_no_pow2"  # square and side is NOT a power of two
    ASPECT_MATCHED = "aspect_matched"  # largest group sharing an aspect ratio


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def filter_measurements(measurements: list, strategy: ShapeFilter = ShapeFilter.SQUARE):
    """Apply a ShapeFilter strategy. Returns (kept, n_discarded)."""
    if strategy == ShapeFilter.NONE:
        kept = list(measurements)
    elif strategy == ShapeFilter.SQUARE:
        kept = [m for m in measurements if m.width == m.height]
    elif strategy == ShapeFilter.SQUARE_NO_POW2:
        kept = [m for m in measurements if m.width == m.height and not _is_pow2(m.width)]
    elif strategy == ShapeFilter.ASPECT_MATCHED:
        groups = defaultdict(list)
        for m in measurements:
            # round to reduce float noise in the ratio bucket key
            ratio = round(m.width / m.height, 3)
            groups[ratio].append(m)
        kept = max(groups.values(), key=len) if groups else []
        kept = sorted(kept, key=lambda m: m.n_pixels)
    else:
        raise ValueError(f"unknown ShapeFilter: {strategy}")

    n_discarded = len(measurements) - len(kept)
    return kept, n_discarded


def pow2_fraction(measurements: list) -> float:
    """Fraction of measurements whose side (assuming square) is a power of two.
    For non-square points, uses width."""
    if not measurements:
        return 0.0
    return sum(1 for m in measurements if _is_pow2(m.width)) / len(measurements)


# ============================================================
#  Point aggregation (robust)
# ============================================================

@dataclass
class AggregatedPoint:
    n_pixels: int
    value: float          # median
    mad: float
    cv: float              # coefficient of variation (std/mean)
    n_reps: int
    reps: list = field(default_factory=list)


def aggregate_point(n_pixels: int, reps: list) -> AggregatedPoint:
    arr = np.asarray(reps, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    cv = std / mean if mean != 0 else 0.0
    return AggregatedPoint(n_pixels=n_pixels, value=med, mad=mad, cv=cv,
                            n_reps=len(arr), reps=list(arr))


def aggregate_measurements(measurements: list, field_name: str) -> list:
    points = []
    for m in measurements:
        reps = getattr(m, field_name)
        if not reps:
            continue
        points.append(aggregate_point(m.n_pixels, reps))
    return points


# ============================================================
#  Candidate models: y = a*f(N) + b  (free-exponent model: y = a*N**p + b)
# ============================================================

def _f_const(N):
    return np.ones_like(np.asarray(N, dtype=float))

def _f_sqrt(N):
    return np.sqrt(N)

def _f_linear(N):
    return N

def _f_nlogn(N):
    return N * np.log2(np.maximum(N, 1.0))

def _f_n15(N):
    return N ** 1.5

def _f_quadratic(N):
    return N ** 2

def _f_cubic(N):
    return N ** 3


CANDIDATE_MODELS = [
    {"name": "O(1)",        "f": _f_const,     "k": 1, "free_exponent": False},
    {"name": "O(sqrt N)",   "f": _f_sqrt,      "k": 2, "free_exponent": False},
    {"name": "O(N)",        "f": _f_linear,    "k": 2, "free_exponent": False},
    {"name": "O(N log N)",  "f": _f_nlogn,     "k": 2, "free_exponent": False},
    {"name": "O(N^1.5)",    "f": _f_n15,       "k": 2, "free_exponent": False},
    {"name": "O(N^2)",      "f": _f_quadratic, "k": 2, "free_exponent": False},
    {"name": "O(N^3)",      "f": _f_cubic,     "k": 2, "free_exponent": False},
    {"name": "O(N^p) free", "f": None,         "k": 3, "free_exponent": True},
]


def _model_func_fixed(f):
    def _fn(N, a, b):
        return a * f(N) + b
    return _fn


def _model_func_free_exponent(N, a, p, b):
    return a * np.asarray(N, dtype=float) ** p + b


# ============================================================
#  Fitting
# ============================================================

def compute_r_squared_relative(y, y_pred):
    """R2 on relative residuals: 1 - sum(((y-yhat)/y)^2) / sum(((y-mean)/y)^2).
    This is what model selection uses (consistent with the sigma=y fit), but
    dividing by y in ss_tot overweights small points and can read misleadingly
    high (e.g. a flat O(1) fit reads ~0.9995 relative even though it explains
    none of the variance) -- report compute_r_squared_classical alongside it
    for a reader, never use it for selection."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel_res = (y - y_pred) / y
    rel_tot = (y - np.mean(y)) / y
    ss_res = np.sum(rel_res ** 2)
    ss_tot = np.sum(rel_tot ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def compute_r_squared_classical(y, y_pred):
    """Textbook R2: 1 - SS_res/SS_tot, unweighted. Scale-invariant (unaffected
    by normalizing y and y_pred by the same constant), so it's fine to
    compute on normalized values. For reporting only -- not used to select."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def compute_bic(n, rss_rel, k):
    if n <= 0:
        return np.inf
    if rss_rel <= 0:
        return -np.inf  # perfect fit: best possible BIC, not a failure
    return n * np.log(rss_rel / n) + k * np.log(n)


@dataclass
class ModelFit:
    name: str
    a: float
    b: float
    p: Optional[float]   # only for free-exponent model
    p_stderr: Optional[float]
    r2: float             # relative R2 -- what BIC/selection is based on
    bic: float
    equation: str
    discarded_reason: Optional[str] = None
    r2_classical: Optional[float] = None  # textbook R2, for display only


def _estimate_p0_g(g, y):
    """p0 from two endpoints, in terms of the already-shape-normalized g
    (g[-1] == 1 at the largest point) rather than re-evaluating f."""
    denom = g[-1] - g[0]
    if denom == 0:
        a0 = 1.0
    else:
        a0 = (y[-1] - y[0]) / denom
    b0 = y[0] - a0 * g[0]
    return a0, b0


def fit_model(x_raw, y_raw, model: dict) -> ModelFit:
    """Fit one candidate model with relative (sigma=y) least squares.

    The shape function f is normalized on its OUTPUT, not its input: fit
    against g(u) = f(x_raw) / f(x_scale) rather than f(x_raw / x_scale).
    Normalizing the input broke O(N log N) -- x_raw/x_scale lands in (0, 1],
    and f_nlogn's np.maximum(N, 1.0) safety clamp (meant to avoid log(0) for
    tiny real N) then clamps every single point to log2(1) = 0, collapsing
    the model to a constant identical to O(1). Normalizing the output
    evaluates f only at the real, always-large x_raw values, so the clamp
    never fires, and de-normalization is then exact for ANY f (not just pure
    powers): a_phys = a_n * y_scale / f(x_scale) -- no per-model exponent
    lookup table needed."""
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    n = len(x_raw)

    x_scale = x_raw.max()
    y_scale = y_raw.max()
    y = y_raw / y_scale

    name = model["name"]

    try:
        if model["free_exponent"]:
            # g(u) = (u*x_scale)**p / x_scale**p = u**p exactly -- a pure
            # power is already scale-invariant this way, no clamp involved.
            u = x_raw / x_scale
            log_ratio_y = np.log(y[-1] / y[0]) if y[0] > 0 else 0.0
            log_ratio_x = np.log(u[-1] / u[0]) if u[0] > 0 else 1.0
            p0_p = log_ratio_y / log_ratio_x if log_ratio_x != 0 else 1.0
            a0 = y[-1] / (u[-1] ** p0_p) if u[-1] > 0 else 1.0
            p0 = (a0, p0_p, 0.0)
            popt, pcov = curve_fit(
                _model_func_free_exponent, u, y, p0=p0, sigma=y, maxfev=20000,
                bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),  # b >= 0
            )
            a_n, p_exp, b_n = popt
            y_pred_n = _model_func_free_exponent(u, *popt)
            p_stderr = float(np.sqrt(pcov[1, 1])) if pcov is not None and pcov.size else None
            f_scale = x_scale ** p_exp

        elif name == "O(1)":
            # f is constant: a and b are redundant (a*1 + b == (a+b)*1), so
            # curve_fit can't identify them separately. Single parameter.
            a0 = float(np.mean(y))

            def const_func(_x, a):
                return np.full_like(_x, a, dtype=float)

            popt, pcov = curve_fit(const_func, x_raw, y, p0=(a0,), sigma=y, maxfev=20000)
            a_n = popt[0]
            b_n = 0.0
            p_exp = None
            p_stderr = None
            y_pred_n = const_func(x_raw, a_n)
            f_scale = 1.0

        else:
            f = model["f"]
            f_vals = f(x_raw)          # evaluated at the real, large N -- no clamp issue
            f_scale = float(f(np.array([x_scale]))[0])
            g = f_vals / f_scale       # g[-1] == 1 at the largest point

            a0, b0 = _estimate_p0_g(g, y)

            def g_func(gg, a, b):
                return a * gg + b

            # b es el overhead (arranque/carga): nunca puede ser negativo
            popt, pcov = curve_fit(g_func, g, y, p0=(a0, max(b0, 0.0)), sigma=y, maxfev=20000,
                                   bounds=([-np.inf, 0.0], [np.inf, np.inf]))
            a_n, b_n = popt
            p_exp = None
            p_stderr = None
            y_pred_n = g_func(g, a_n, b_n)

        if a_n <= 0:
            return ModelFit(name=name, a=np.nan, b=np.nan, p=p_exp, p_stderr=p_stderr,
                             r2=np.nan, bic=np.inf, equation="",
                             discarded_reason="a <= 0 (not a growing model)")

        r2 = compute_r_squared_relative(y, y_pred_n)
        r2_classical = compute_r_squared_classical(y, y_pred_n)
        rel_res = (y - y_pred_n) / y
        rss_rel = np.sum(rel_res ** 2)
        bic = compute_bic(n, rss_rel, model["k"])

        # de-normalize: exact for any f, not just pure powers
        a_phys = a_n * y_scale / f_scale
        b_phys = b_n * y_scale

        equation = _format_equation(name, a_phys, b_phys, p_exp)

        return ModelFit(name=name, a=a_phys, b=b_phys, p=p_exp, p_stderr=p_stderr,
                         r2=r2, bic=bic, equation=equation, discarded_reason=None,
                         r2_classical=r2_classical)

    except Exception as e:
        return ModelFit(name=name, a=np.nan, b=np.nan, p=None, p_stderr=None,
                         r2=np.nan, bic=np.inf, equation="",
                         discarded_reason=f"fit failed: {e}")


def _format_equation(name, a, b, p_exp) -> str:
    if name == "O(N log N)":
        return f"y = ({a:.4e})*N*log2(N) + ({b:.4e})"
    if name == "O(N^p) free":
        return f"y = ({a:.4e})*N^{p_exp:.3f} + ({b:.4e})"
    if name == "O(1)":
        return f"y = ({a:.4e})"
    if name == "O(sqrt N)":
        return f"y = ({a:.4e})*sqrt(N) + ({b:.4e})"
    if name == "O(N^1.5)":
        return f"y = ({a:.4e})*N^1.5 + ({b:.4e})"
    if name == "O(N^2)":
        return f"y = ({a:.4e})*N^2 + ({b:.4e})"
    if name == "O(N^3)":
        return f"y = ({a:.4e})*N^3 + ({b:.4e})"
    return f"y = ({a:.4e})*N + ({b:.4e})"


def fit_all_models(x, y) -> list:
    return [fit_model(x, y, model) for model in CANDIDATE_MODELS]


# ============================================================
#  Asymptotic regime diagnosis
# ============================================================

def local_slopes(x, y) -> list:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes = []
    for i in range(len(x) - 1):
        dlogx = np.log(x[i + 1]) - np.log(x[i])
        dlogy = np.log(y[i + 1]) - np.log(y[i])
        slopes.append(float(dlogy / dlogx) if dlogx != 0 else 0.0)
    return slopes


def corrected_local_slopes(x, y, b):
    """Local slopes of (y - b): the fitted offset already estimates the
    additive overhead, so subtracting it before computing local slopes
    removes the contamination that keeps the raw slope from ever reaching
    the true exponent within a handful of geometrically-spaced points.
    Points where y - b is non-positive (log undefined) are dropped from the
    diagnostic rather than falling back to the raw series wholesale.
    Returns (slopes, n_points_dropped, n_points_used)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_corr = y - b
    mask = y_corr > 0
    n_dropped = int(np.sum(~mask))
    x_used = x[mask]
    y_used = y_corr[mask]
    return local_slopes(x_used, y_used), n_dropped, int(len(x_used))


ASYMPTOTIC_WINDOW = 3
DRIFT_P_THRESHOLD = 0.10       # Spearman trend test significance
DRIFT_RANGE_FRACTION = 0.15    # window range / |mean|, combined with a significant trend
NOISE_RANGE_FRACTION = 0.50    # window range / |mean|, regardless of trend


def asymptotic_reached(slopes: list, window: int = ASYMPTOTIC_WINDOW) -> bool:
    """Not a dispersion-vs-noise-scale test: the real question is whether the
    spread in the last `window` corrected local slopes is systematic drift or
    random scatter, which a nonparametric trend test answers directly instead
    of needing any measurement-noise scale (repetition CV turned out to be
    the wrong scale for cross-resolution variation -- see discussion).

    Two conditions, both needed:
    - drift: a significant Spearman trend (p < DRIFT_P_THRESHOLD) between
      window position and slope, AND the window's range exceeds
      DRIFT_RANGE_FRACTION of its mean. Without the range side, a legitimate
      monotonic convergence like [0.95, 0.99, 1.00] would be rejected for
      being monotonic.
    - excess noise: range exceeds NOISE_RANGE_FRACTION of the mean,
      regardless of trend significance -- catches a series so noisy it lost
      monotonicity (e.g. a 179% range) that would otherwise slip past the
      trend test.

    Converged is the negation of both."""
    if len(slopes) < window:
        return False
    last = np.asarray(slopes[-window:], dtype=float)

    if np.allclose(last, last[0]):
        return True  # identical slopes: perfect convergence, Spearman is undefined (nan)

    mean_val = abs(np.mean(last))
    rng = float(np.max(last) - np.min(last))
    rng_frac = rng / mean_val if mean_val != 0 else np.inf

    _, pval = spearmanr(np.arange(len(last)), last)
    drift = (not np.isnan(pval)) and (pval < DRIFT_P_THRESHOLD) and (rng_frac > DRIFT_RANGE_FRACTION)
    excess_noise = rng_frac > NOISE_RANGE_FRACTION

    return not drift and not excess_noise


# ============================================================
#  Bootstrap + BIC model selection
# ============================================================

BIC_MARGIN = 6.0  # Kass-Raftery "strong evidence" threshold, reused for every BIC comparison below
CANONICAL_ORDERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
EXPONENT_SNAP_TOL = 0.15  # matches asymptotic_reached's tol; both are "close enough" gates


def _select_best_and_runnerup(fits: list):
    """Pick the model with lowest BIC, with two adjustments needed to make the
    free-exponent model behave as spec 1.4 intends ("solo gana si ninguno de
    los canonicos encaja"):

    1. The free-exponent model is nested over every canonical model (one
       extra parameter), so with few data points and low noise it can fit
       sampling noise better than any canonical model and win BIC by a hair
       even when a canonical model is the true generator. Require it to beat
       the best canonical model's BIC by a Kass-Raftery "strong evidence"
       margin before it is allowed to win at all.
    2. Even past that margin, a fitted exponent p landing within
       EXPONENT_SNAP_TOL of a canonical order (e.g. p=1.0014 on real
       quadratic.txt data) means the data IS that canonical class and the
       margin was won on noise/model misspecification, not on genuinely
       falling outside the canonical family. Snap back to the canonical
       model in that case; only let the free model win when p is clearly
       away from every canonical order (e.g. a genuine O(N^2.5)).

    Returns (winner, runner_up). The runner-up is always drawn from the same
    pool the winner was chosen from (canonical vs canonical when the winner
    is canonical, canonical vs free when the winner is free) -- comparing a
    canonical winner against a free-exponent model that was just deliberately
    snapped away would manufacture a spurious "tie" every time the snap
    rule fires, since that's exactly the model the winner was preferred over."""
    valid = [f for f in fits if f.discarded_reason is None and np.isfinite(f.bic)]
    if not valid:
        return None, None
    canonical = sorted([f for f in valid if f.name != "O(N^p) free"], key=lambda f: f.bic)
    free = next((f for f in valid if f.name == "O(N^p) free"), None)

    if free is None or not canonical:
        ranked = sorted(valid, key=lambda f: f.bic)
        return ranked[0], (ranked[1] if len(ranked) > 1 else None)

    best_canonical = canonical[0]
    runner_up_canonical = canonical[1] if len(canonical) > 1 else None

    if best_canonical.bic - free.bic < BIC_MARGIN:
        return best_canonical, runner_up_canonical

    near_canonical = free.p is not None and min(
        abs(free.p - order) for order in CANONICAL_ORDERS
    ) <= EXPONENT_SNAP_TOL
    if near_canonical:
        return best_canonical, runner_up_canonical
    return free, best_canonical


def _select_best(fits: list):
    return _select_best_and_runnerup(fits)[0]


def bootstrap_select(measurements: list, field_name: str, n_iter: int = 500,
                      rng: Optional[np.random.Generator] = None):
    """Hierarchical bootstrap: resample reps per folder, recompute median,
    refit all models, track which model wins (lowest BIC) each iteration."""
    if rng is None:
        rng = np.random.default_rng(0)

    win_counts = defaultdict(int)
    valid_iters = 0

    for _ in range(n_iter):
        x_boot = []
        y_boot = []
        for m in measurements:
            reps = getattr(m, field_name)
            if not reps:
                continue
            resampled = rng.choice(reps, size=len(reps), replace=True)
            x_boot.append(m.n_pixels)
            y_boot.append(float(np.median(resampled)))

        if len(x_boot) < 3:
            continue

        x_boot = np.array(x_boot, dtype=float)
        y_boot = np.array(y_boot, dtype=float)
        fits = fit_all_models(x_boot, y_boot)
        best = _select_best(fits)
        if best is not None:
            win_counts[best.name] += 1
            valid_iters += 1

    if valid_iters == 0:
        return {}, 0
    frequencies = {name: count / valid_iters for name, count in win_counts.items()}
    return frequencies, valid_iters


# ============================================================
#  Verdict
# ============================================================

@dataclass
class Verdict:
    model_name: Optional[str]
    confidence: float
    equation: str
    r2: float
    bic: float
    ranking: list
    local_slopes: list             # raw, uncorrected -- informative: shows overhead's weight
    local_slopes_corrected: list   # overhead (fitted b) subtracted before taking slopes
    asymptotic_reached: bool
    n_points_used: int
    n_points_discarded: int
    warnings: list
    abstain_reason: Optional[str]
    model_a: Optional[float] = None   # physical-units coefficients of the winning
    model_b: Optional[float] = None   # model, so a caller can plot the curve without
    model_p: Optional[float] = None   # redefining the model family (see model_value())
    r2_classical: Optional[float] = None  # textbook R2 of the winner, for display
    signal_overhead_ratios: list = field(default_factory=list)  # (y-b)/b per point
    n_useful_points: int = 0  # points where signal_overhead_ratio >= USEFUL_RATIO_THRESHOLD
    bic_best_model: Optional[str] = None       # point-estimate winner by BIC, even if abstaining
    bootstrap_top_model: Optional[str] = None  # most frequent winner across bootstrap iterations
    bootstrap_top_confidence: float = 0.0      # that model's own win frequency
    fit_a: Optional[float] = None  # coefficients of the BIC-best fit, kept even when the verdict
    fit_b: Optional[float] = None  # abstains (model_a/b/p are None then), so a UI can still show
    fit_p: Optional[float] = None  # the curve scipy found -- labeled as unvalidated


USEFUL_RATIO_THRESHOLD = 1.0  # algorithm must cost at least as much as process startup
MIN_USEFUL_POINTS_FOR_ASYMPTOTIC = 3


def signal_to_overhead_ratios(y, b) -> list:
    """|(y - b) / b| per point: how many times the fitted overhead the
    algorithmic signal is. A ratio < 1 means the point mostly measures
    interpreter/import startup, not the algorithm under study."""
    y = np.asarray(y, dtype=float)
    if b == 0:
        return [np.inf if v > 0 else 0.0 for v in y]
    # abs: se asume que el coste siempre crece con N, asi que un b ajustado
    # negativo no debe invertir el signo de la razon.
    return [float(abs((v - b) / b)) for v in y]


def count_useful_points(ratios: list, threshold: float = USEFUL_RATIO_THRESHOLD) -> int:
    return sum(1 for r in ratios if r >= threshold)


def model_value(name: str, N, a: float, b: float, p: Optional[float] = None):
    """Evaluate a fitted model at N, in physical units -- lets a caller (e.g. a
    plotting UI) draw the winning curve without redefining the model family."""
    N = np.asarray(N, dtype=float)
    if name == "O(N^p) free":
        return a * N ** p + b
    model = next(m for m in CANDIDATE_MODELS if m["name"] == name)
    return a * model["f"](N) + b


MIN_POINTS = 5
MIN_DIAGNOSTIC_POINTS = 4  # need >=3 corrected local slopes for the asymptotic-regime window
CONFIDENCE_THRESHOLD = 0.60
POW2_WARNING_FRACTION = 0.70


def classify(measurements: list, field_name: str = "time_reps",
             shape_filter: ShapeFilter = ShapeFilter.ASPECT_MATCHED,
             n_bootstrap: int = 500,
             rng: Optional[np.random.Generator] = None) -> Verdict:
    """Full pipeline: filter -> aggregate -> fit -> diagnose -> bootstrap -> verdict."""
    warnings_list = []

    filtered, n_discarded = filter_measurements(measurements, shape_filter)

    points = aggregate_measurements(filtered, field_name)
    n_used = len(points)

    if n_used < MIN_POINTS:
        if not measurements:
            # Distinct from the filter trimming points down: there was
            # nothing to filter in the first place. Usually means the caller
            # got past a parse that raised no exception but still yielded no
            # measurements (e.g. built the list manually), since
            # parse_performance_file itself now raises on this.
            abstain_reason = (
                "no measurements to classify: the input list is "
                "empty (0 measurements before any filter)"
            )
        else:
            abstain_reason = (
                f"only {n_used} points after the {shape_filter.value} filter, "
                f"at least {MIN_POINTS} are needed to discriminate between models"
            )
        return Verdict(
            model_name=None, confidence=0.0, equation="", r2=float("nan"),
            bic=float("inf"), ranking=[], local_slopes=[], local_slopes_corrected=[],
            asymptotic_reached=False,
            n_points_used=n_used, n_points_discarded=n_discarded,
            warnings=warnings_list,
            abstain_reason=abstain_reason,
        )

    frac_pow2 = pow2_fraction(filtered)
    if frac_pow2 > POW2_WARNING_FRACTION:
        warnings_list.append(
            f"{frac_pow2:.0%} of the points used have a power of two side: "
            "the sample is biased by cache aliasing and the estimated exponent is "
            "an upper bound, not the algorithmic exponent"
        )

    x = np.array([p.n_pixels for p in points], dtype=float)
    y = np.array([p.value for p in points], dtype=float)

    fits = fit_all_models(x, y)
    best, runner_up = _select_best_and_runnerup(fits)

    ranking_fits = sorted(
        [f for f in fits if f.discarded_reason is None and np.isfinite(f.bic)],
        key=lambda f: f.bic,
    )

    if best is None:
        return Verdict(
            model_name=None, confidence=0.0, equation="", r2=float("nan"),
            bic=float("inf"),
            ranking=[{"name": f.name, "bic": f.bic, "r2": f.r2,
                      "r2_classical": f.r2_classical,
                      "discarded_reason": f.discarded_reason} for f in fits],
            local_slopes=local_slopes(x, y), local_slopes_corrected=[],
            asymptotic_reached=False,
            n_points_used=n_used, n_points_discarded=n_discarded,
            warnings=warnings_list,
            abstain_reason="no model converged to a valid fit",
        )

    # Signal-vs-overhead diagnostic, always exposed via Verdict.signal_overhead_ratios /
    # n_useful_points. When fewer than half the points actually carry signal
    # above the overhead, also raise it as a warning: the bootstrap measures
    # stability against measurement noise (repetitions of the same size are
    # very consistent), not against a shortage of informative points, so it
    # can report 100% confidence on a fit that really rests on 2 of 6 points.
    # Same failure mode as the O(N) vs O(N log N) tie, solved the same way
    # there with a BIC margin -- here there's no model-selection knob to
    # apply it to, so surface it as a warning instead of silently trusting
    # the bootstrap number. Not an abstention: a well-supported classification
    # (e.g. O(N log N) for an FFT-based algorithm) shouldn't be hidden, it
    # needs to be shown together with this caveat instead.
    signal_overhead_ratios = signal_to_overhead_ratios(y, best.b)
    n_useful_points = count_useful_points(signal_overhead_ratios)
    if n_useful_points < n_used / 2:
        warnings_list.append(
            f"only {n_useful_points}/{n_used} points have algorithmic signal equal "
            "to or greater than the overhead ((y-b)/b >= 1); the fit relies on few "
            "truly informative points and the bootstrap confidence does not reflect this "
            "limitation, because it measures stability against repetition noise, not "
            "against a shortage of useful points"
        )

    raw_slopes = local_slopes(x, y)
    corrected, n_dropped_corr, n_points_corr = corrected_local_slopes(x, y, best.b)
    if n_dropped_corr > 0:
        warnings_list.append(
            f"{n_dropped_corr} point(s) excluded from the asymptotic regime "
            "diagnostic because y - b (fitted overhead) is not positive there"
        )
    reached = asymptotic_reached(corrected) if n_points_corr >= MIN_DIAGNOSTIC_POINTS else False

    ranking = [
        {"name": f.name, "bic": f.bic, "r2": f.r2, "r2_classical": f.r2_classical}
        for f in ranking_fits
    ]

    # Gates run in dependency order -- each only makes sense if the previous
    # one passed: enough points -> a real scaling signal exists (vs O(1)) ->
    # that signal isn't tied with a second model -> its regime has actually
    # converged -> the classification is stable under resampling. Checking
    # the asymptotic regime before asking "is there a signal at all" produces
    # a misleading reason on pure-noise data: it says "need larger images"
    # when no signal will ever appear at any size, because the O(1)/tie gates
    # never got a chance to run.
    o1_fit = next((f for f in fits if f.name == "O(1)"
                    and f.discarded_reason is None and np.isfinite(f.bic)), None)

    abstain_reason = None
    if best.name == "O(1)":
        abstain_reason = (
            "the winning model is O(1): there is no measurable scaling with "
            "image size"
        )
    elif o1_fit is not None and (o1_fit.bic - best.bic) < BIC_MARGIN:
        abstain_reason = (
            f"the winning model ({best.name}) does not beat O(1) by a BIC margin "
            f"of at least {BIC_MARGIN:.0f} (delta={o1_fit.bic - best.bic:.2f}); "
            "there is no measurable scaling against a constant term"
        )
    elif runner_up is not None and (runner_up.bic - best.bic) < BIC_MARGIN:
        abstain_reason = (
            f"tie: {best.name} and {runner_up.name} differ in BIC by "
            f"{runner_up.bic - best.bic:.2f}, below the margin "
            f"({BIC_MARGIN:.0f}) required to prefer one over the other; "
            "the data is not enough to discriminate between them"
        )
    elif n_points_corr < MIN_DIAGNOSTIC_POINTS:
        abstain_reason = (
            f"insufficient data to diagnose the asymptotic regime: "
            f"only {n_points_corr} usable points after excluding y - b <= 0 "
            f"(at least {MIN_DIAGNOSTIC_POINTS} are needed)"
        )
    elif not reached:
        if n_useful_points < MIN_USEFUL_POINTS_FOR_ASYMPTOTIC:
            abstain_reason = (
                f"startup overhead dominates the measurement: only "
                f"{n_useful_points}/{n_used} points have algorithmic signal equal "
                f"to or greater than the overhead ((y-b)/b >= 1). The problem is not a lack "
                "of larger image sizes, it is that the measurement includes the "
                "interpreter startup at every point -- measuring the kernel in a loop inside the "
                "process or subtracting a no-op fixes this without needing larger "
                "images"
            )
        else:
            abstain_reason = (
                "asymptotic regime not reached (sustained drift or excessive dispersion "
                "in the last overhead-corrected local slopes: "
                f"{['%.2f' % s for s in corrected[-3:]]}), larger image sizes "
                "are needed to confirm the complexity"
            )

    confidence = 0.0
    bootstrap_top_model = None
    bootstrap_top_confidence = 0.0
    if abstain_reason is None:
        frequencies, valid_iters = bootstrap_select(
            filtered, field_name, n_iter=n_bootstrap, rng=rng,
        )
        # confidence is specifically best.name's OWN win frequency (the model
        # BIC picked on the point estimate) -- NOT the bootstrap plurality
        # winner. They usually agree (if best.name clears CONFIDENCE_THRESHOLD
        # it's mathematically the plurality too, since two models can't both
        # exceed 50% of the iterations), but when best.name does NOT clear the
        # threshold, a different model can be the bootstrap's actual favorite.
        # Naming best.name "el ganador" in that case while citing its low
        # percentage, with no mention of who actually leads the bootstrap,
        # reads as a contradiction ("winner" with a minority number) rather
        # than what it is: two criteria disagreeing.
        confidence = frequencies.get(best.name, 0.0)
        for r in ranking:
            r["confidence"] = frequencies.get(r["name"], 0.0)

        if frequencies:
            bootstrap_top_model = max(frequencies, key=frequencies.get)
            bootstrap_top_confidence = frequencies[bootstrap_top_model]

        if confidence < CONFIDENCE_THRESHOLD:
            if bootstrap_top_model is not None and bootstrap_top_model != best.name:
                abstain_reason = (
                    f"the model with the best Bayesian criterion (BIC) is {best.name}, "
                    f"but the bootstrap favors {bootstrap_top_model} in "
                    f"{bootstrap_top_confidence:.0%} of the resamples ({best.name} "
                    f"itself only won {confidence:.0%}); the discrepancy "
                    "between the two criteria prevents a confident classification"
                )
            else:
                abstain_reason = (
                    f"the winning model ({best.name}) only won {confidence:.0%} of the "
                    f"bootstrap iterations (threshold {CONFIDENCE_THRESHOLD:.0%}); the fit "
                    "is not stable enough against measurement noise to classify "
                    "with confidence"
                )
    else:
        for r in ranking:
            r["confidence"] = 0.0

    model_name = None if abstain_reason else best.name

    return Verdict(
        model_name=model_name,
        confidence=confidence,
        equation=best.equation,
        r2=best.r2,
        bic=best.bic,
        ranking=ranking,
        local_slopes=raw_slopes,
        local_slopes_corrected=corrected,
        asymptotic_reached=reached,
        n_points_used=n_used,
        n_points_discarded=n_discarded,
        warnings=warnings_list,
        abstain_reason=abstain_reason,
        model_a=best.a if model_name else None,
        model_b=best.b if model_name else None,
        model_p=best.p if model_name else None,
        r2_classical=best.r2_classical,
        signal_overhead_ratios=signal_overhead_ratios,
        n_useful_points=n_useful_points,
        bic_best_model=best.name,
        bootstrap_top_model=bootstrap_top_model,
        bootstrap_top_confidence=bootstrap_top_confidence,
        fit_a=best.a,
        fit_b=best.b,
        fit_p=best.p,
    )
