"""Protocolo v2 de medicion con perf: aisla el coste de UNA llamada.

Cada bloque es un proceso nuevo (la memoria del proceso se libera sola):

    L   libs + modulo, sin imagen ni llamada       (una vez por corrida)
    LI  L + carga de la imagen, sin llamada        (R veces por imagen)
    L1  LI + 1 llamada a la funcion                (R veces por imagen)
    LK  LI + K llamadas en bucle, K >= 2           (R veces por imagen)

Coste de una llamada en regimen = (LK - L1) / (K - 1); lo lee complexity_fit
(correct_warm). LI - L es el coste de cargar la imagen y L1 - LI el de la
primera llamada en frio; ambos quedan en los bloques crudos del .txt.

K es adaptativo: se estima el tiempo de una llamada con la primera pareja
LI/L1 y se elige K para que el bucle dure ~K_TARGET_SECONDS, acotado a
[K_MIN, K_MAX].
"""

import math

from complexity_fit import PROTOCOL_MARK, _PATTERNS

LIBS_REPS = 15
K_MIN = 2
K_MAX = 20
K_TARGET_SECONDS = 1.0


def build_code(module_code: str, mode: str, file_dir_wsl: str,
               image_expr: str = "", image_path_wsl: str = "",
               call: str = "", k: int = 0) -> str:
    """Script a medir. mode: 'L' (solo libs), 'LI' (+ imagen), 'LK' (+ k llamadas).
    `call` es la llamada a la funcion con el argumento de imagen ya reemplazado
    por la variable `_img`."""
    lines = [
        "# -*- coding: utf-8 -*-",
        "import sys, os",
        "import numpy as np",
        "from PIL import Image",
        f"sys.path.insert(0, r'{file_dir_wsl}')",
        "",
        module_code,
        "",
        "if __name__ == '__main__':",
    ]
    if mode == "L":
        lines.append("    pass")
        return "\n".join(lines)

    lines.append(f"    _img = {image_expr}")
    if image_path_wsl:
        lines += [
            "    try:",
            f"        w, h = Image.open(r'{image_path_wsl}').size",
            "        print(f'IMAGE_DIMENSIONS: {w}x{h}')",
            "    except Exception as e:",
            "        print(f'IMAGE_DIMENSIONS_ERROR: {e}')",
        ]
    if mode == "LK":
        lines += [
            f"    for _ in range({k}):",
            f"        result = {call}",
        ]
    return "\n".join(lines)


def parse_metrics(text: str) -> dict:
    """{metric: value} de una salida de perf stat (vacio si perf/el script fallo)."""
    out = {}
    for line in text.splitlines():
        for key in ('cycles', 'instructions', 'cache_refs', 'cache_misses', 'time'):
            m = _PATTERNS[key].search(line)
            if m:
                out[key] = float(m.group(1).replace(',', ''))
    return out


def choose_k(t_call):
    """K para que el bucle dure ~K_TARGET_SECONDS. t_call = tiempo estimado de
    una llamada; si no se pudo estimar o es ruido (<= 0), la funcion es
    demasiado rapida para medirla y se usa K_MAX."""
    if t_call is None or t_call <= 0:
        return K_MAX
    return max(K_MIN, min(K_MAX, math.ceil(K_TARGET_SECONDS / t_call)))


def _block(label: str, k: int, rep: int, output: str) -> str:
    return f"  [BLOCK {label} K={k}] Repetition {rep}:\n{output}\n"


def measure_libs_baseline(run_code, module_code: str, file_dir_wsl: str,
                          reps: int = LIBS_REPS, log=None) -> str:
    """Bloques L (libs sin imagen). Se escribe una vez al inicio del archivo."""
    code = build_code(module_code, "L", file_dir_wsl)
    parts = []
    for i in range(1, reps + 1):
        if log:
            log(f"Baseline libs {i}/{reps}")
        parts.append(_block("L", 0, i, run_code(code)))
    return "".join(parts)


def measure_image(run_code, module_code: str, file_dir_wsl: str,
                  image_path_wsl: str, image_expr: str, call: str,
                  reps: int, log=None) -> str:
    """Bloques LI/L1/LK de una imagen, intercalados por repeticion.
    run_code(script) -> salida de perf. Devuelve el texto listo para escribir."""
    code_li = build_code(module_code, "LI", file_dir_wsl, image_expr, image_path_wsl)
    code_l1 = build_code(module_code, "LK", file_dir_wsl, image_expr, image_path_wsl, call, 1)

    parts = {"LI": [], "L1": [], "LK": []}
    out_li = run_code(code_li)
    out_l1 = run_code(code_l1)
    parts["LI"].append(_block("LI", 0, 1, out_li))
    parts["L1"].append(_block("L1", 1, 1, out_l1))

    m_li, m_l1 = parse_metrics(out_li), parse_metrics(out_l1)
    t_call = (m_l1["time"] - m_li["time"]) if "time" in m_li and "time" in m_l1 else None
    k = choose_k(t_call)
    if log:
        est = "n/a" if t_call is None else f"{t_call * 1000:.1f} ms"
        log(f"    1 call ~ {est} -> K={k}")

    code_lk = build_code(module_code, "LK", file_dir_wsl, image_expr, image_path_wsl, call, k)
    parts["LK"].append(_block("LK", k, 1, run_code(code_lk)))

    for i in range(2, reps + 1):
        parts["LI"].append(_block("LI", 0, i, run_code(code_li)))
        parts["L1"].append(_block("L1", 1, i, run_code(code_l1)))
        parts["LK"].append(_block("LK", k, i, run_code(code_lk)))

    header = f"  {PROTOCOL_MARK} K={k} reps={reps}\n"
    return header + "".join(parts["LI"] + parts["L1"] + parts["LK"])
