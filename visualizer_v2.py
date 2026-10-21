"""Analisis y graficado de archivos de resultados; lo usa la pestaña Visualization de Main.py."""
import os

import numpy as np
from matplotlib.figure import Figure

from complexity_fit import (
    ShapeFilter,
    parse_performance_file,
    filter_measurements,
    aggregate_measurements,
    classify,
    model_value,
    PerfFileFormatError,
    PerfMeasurementsMissingError,
)

METRICS = [("instr_reps", "Instructions"), ("time_reps", "Time")]


# ============================================================
#  Analisis
# ============================================================

def analyze_file(txt_file, shape_filter: ShapeFilter):
    """Parsea un archivo de resultados y clasifica tiempo e instrucciones con
    complexity_fit. Devuelve tambien los puntos agregados (mediana + MAD) que
    ya filtro classify(), para poder graficarlos de forma consistente con el
    veredicto.

    Si el archivo no tiene el formato esperado, o lo tiene pero no contiene
    mediciones (p.ej. perf no estaba instalado en la maquina que lo genero),
    devuelve un resultado con 'parse_error' en vez de lanzar o de seguir con
    una lista vacia en silencio."""
    try:
        measurements = parse_performance_file(txt_file)
    except (PerfFileFormatError, PerfMeasurementsMissingError) as e:
        return {"filename": os.path.basename(txt_file), "parse_error": str(e)}

    filtered, n_discarded = filter_measurements(measurements, shape_filter)

    result = {
        "filename": os.path.basename(txt_file),
        "shape_filter": shape_filter,
        "n_discarded": n_discarded,
        "metrics": {},
    }
    for field_name, label in METRICS:
        verdict = classify(measurements, field_name=field_name, shape_filter=shape_filter)
        points = aggregate_measurements(filtered, field_name)
        result["metrics"][field_name] = {
            "label": label,
            "verdict": verdict,
            "points": points,
        }
    return result


def format_verdict_report(filename, metric_label, verdict) -> str:
    """Texto del reporte de un veredicto, tal como se muestra en el log de la GUI."""
    lines = []
    lines.append(f"--- {filename} | {metric_label} ---")
    if verdict.model_name is not None:
        lines.append(
            f"  Verdict: {verdict.model_name}  "
            f"(bootstrap confidence={verdict.confidence:.0%}, "
            f"R²_relative={verdict.r2:.6f}, R²_classical={verdict.r2_classical:.6f})"
        )
        lines.append(f"  Equation: {verdict.equation}")
    else:
        lines.append("  Verdict: ABSTAIN")
        lines.append(f"  Reason: {verdict.abstain_reason}")
        if verdict.bic_best_model is not None:
            lines.append(
                f"  Best BIC fit (not validated): {verdict.bic_best_model}  "
                f"R²_relative={verdict.r2:.6f}"
            )
            lines.append(f"  Equation: {verdict.equation}")

    lines.append(
        f"  Raw local slopes:            "
        f"{['%.2f' % s for s in verdict.local_slopes]}"
    )
    lines.append(
        f"  Corrected local slopes:      "
        f"{['%.2f' % s for s in verdict.local_slopes_corrected]}"
    )
    lines.append(f"  Asymptotic regime reached: {verdict.asymptotic_reached}")
    lines.append(
        f"  Points used: {verdict.n_points_used}  "
        f"(discarded by shape filter: {verdict.n_points_discarded})"
    )
    if verdict.signal_overhead_ratios:
        lines.append(
            f"  Useful points (signal >= overhead): "
            f"{verdict.n_useful_points}/{verdict.n_points_used}  "
            f"ratios (y-b)/b: {['%.2f' % r for r in verdict.signal_overhead_ratios]}"
        )

    if verdict.ranking:
        lines.append(
            "  Ranking (BIC, relative R² [used to choose], classical R², "
            "bootstrap frequency):"
        )
        for r in verdict.ranking:
            marker = " << WINNER" if r["name"] == verdict.model_name else ""
            r2c = r.get("r2_classical")
            r2c_str = f"{r2c:.6f}" if r2c is not None else "n/a"
            lines.append(
                f"    {r['name']:16s}  BIC={r['bic']:12.2f}  "
                f"R²_rel={r['r2']:.6f}  R²_class={r2c_str}  "
                f"freq={r.get('confidence', 0.0):.0%}{marker}"
            )

    if verdict.warnings:
        for w in verdict.warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


def create_plots(analysis_results):
    """Genera figura con 2 subplots: instrucciones y tiempo. Si el
    veredicto abstiene, NO se dibuja curva: solo los puntos (con barras
    de error de MAD) y el motivo escrito en el grafico."""
    fig = Figure(figsize=(6, 5))
    axes = fig.subplots(2, 1)
    axes_by_metric = {"instr_reps": axes[0], "time_reps": axes[1]}
    file_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for idx, result in enumerate(analysis_results):
        color = file_colors[idx % len(file_colors)]
        fname = result["filename"]

        for field_name, label in METRICS:
            ax = axes_by_metric[field_name]
            m = result["metrics"][field_name]
            verdict = m["verdict"]
            points = m["points"]
            if not points:
                continue

            x = np.array([p.n_pixels for p in points], dtype=float)
            y = np.array([p.value for p in points], dtype=float)
            yerr = np.array([p.mad for p in points], dtype=float)

            ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, markersize=5,
                        alpha=0.8, capsize=3, zorder=5, label=f"{fname}")

            if verdict.model_name is not None:
                xx = np.linspace(x.min(), x.max(), 500)
                yy = model_value(verdict.model_name, xx, verdict.model_a,
                                  verdict.model_b, verdict.model_p)
                ax.plot(xx, yy, linestyle='--', color=color, linewidth=2, alpha=0.8,
                        label=(f"  {verdict.model_name}  {verdict.equation}  "
                               f"(R²={verdict.r2:.4f}, conf={verdict.confidence:.0%})"))
            else:
                # Sin veredicto: igual se muestra el mejor ajuste de scipy
                # (menor BIC), punteado y marcado como no validado.
                if verdict.bic_best_model is not None and verdict.fit_a is not None:
                    xx = np.linspace(x.min(), x.max(), 500)
                    yy = model_value(verdict.bic_best_model, xx, verdict.fit_a,
                                      verdict.fit_b, verdict.fit_p)
                    ax.plot(xx, yy, linestyle=':', color=color, linewidth=2, alpha=0.8,
                            label=(f"  [no verdict] best BIC fit: "
                                   f"{verdict.bic_best_model}  {verdict.equation}  "
                                   f"(R²={verdict.r2:.4f})"))

    axes[0].set_xlabel('Total pixels (N)', fontsize=11)
    axes[0].set_ylabel('Instructions', fontsize=11)
    axes[0].set_title('Computational complexity (instructions)', fontsize=12)
    axes[0].tick_params(labelsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(fontsize=8, loc='upper left')

    axes[1].set_xlabel('Total pixels (N)', fontsize=11)
    axes[1].set_ylabel('Time (s)', fontsize=11)
    axes[1].set_title('Time complexity', fontsize=12)
    axes[1].tick_params(labelsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    return fig
