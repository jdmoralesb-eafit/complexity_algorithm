import os
import subprocess
import tempfile
import textwrap
import platform
import re
import shlex
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import inspect
import importlib.util
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import collections
import dis

# --- Funciones existentes ---
def detect_dependencies(code: str):
    imports = re.findall(r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)', code, re.MULTILINE)
    std_libs = {
        "sys", "os", "math", "re", "subprocess", "platform", "tempfile", "textwrap",
        "inspect", "collections", "shlex", "types", "functools", "itertools",
        "json", "time", "datetime", "pathlib", "logging", "dis"
    }
    external = [pkg.split('.')[0] for pkg in imports if pkg.split('.')[0] not in std_libs]
    return list(set(external))

def check_wsl_dependencies(packages):
    missing_packages = []
    
    import_aliases = {
        "skimage": ["skimage", "skimage.restoration"],
        "Pillow": ["PIL", "PIL.Image"],
        "matplotlib": ["matplotlib", "matplotlib.pyplot"],
        "scipy": ["scipy", "scipy.optimize", "scipy.ndimage"],
        "numpy": ["numpy"]
    }
    
    for pkg in packages:
        aliases = import_aliases.get(pkg, [pkg])
        found = False
        
        for alias in aliases:
            check_cmd = ["wsl", "python3", "-c", f"import {alias}"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                found = True
                break
        
        if not found:
            missing_packages.append(pkg)
    
    return missing_packages

def verify_dependencies_once():
    if platform.system() != "Windows":
        return True
    
    example_code = """
    import numpy as np
    from PIL import Image
    import math
    from math import pi
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    import scipy.optimize
    from scipy import ndimage
    from scipy.ndimage import median_filter
    from skimage.restoration import unwrap_phase
    from scipy.sparse.linalg import svds
    from scipy.interpolate import RegularGridInterpolator
    import tkinter as tk
    from tkinter import messagebox
    """
    
    deps = detect_dependencies(example_code)
    if deps:
        missing_deps = check_wsl_dependencies(deps)
        if missing_deps:
            return False
    
    return True

def measure_function_perf(func_import_path: str, func_call: str, repetitions: int = 5):
    # Obtener la ruta del archivo seleccionado por el usuario
    selected_file_path = getattr(measure_function_perf, 'selected_file_path', None)
    
    if not selected_file_path:
        # Si no hay archivo seleccionado, ejecutar normalmente
        all_results = []
        for rep in range(repetitions):
            result = execute_perf_measurement("", func_call, selected_file_path)
            all_results.append(("Single execution", [result]))
        return all_results
    
    try:
        with open(selected_file_path, 'r', encoding='utf-8') as f:
            module_code = f.read()
    except FileNotFoundError:
        return ["Error: Archivo no encontrado"] * repetitions
    except Exception as e:
        return [f"Error al leer archivo: {e}"] * repetitions
    
    all_results = []
    
    # Extraer el nombre de la función
    func_name_match = re.search(r'^(\w+)\(', func_call)
    if not func_name_match:
        # Ejecución normal
        for rep in range(repetitions):
            result = execute_perf_measurement(module_code, func_call, selected_file_path)
            all_results.append(("Single execution", [result]))
        return all_results
    
    func_name = func_name_match.group(1)
    
    # Extraer el primer parámetro
    first_param_match = re.search(r'\((.*?)=', func_call)
    if first_param_match:
        first_param_name = first_param_match.group(1)
        param_value_match = re.search(f"{first_param_name}=([^,)]+)", func_call)
        
        if param_value_match:
            first_param_value = param_value_match.group(1).strip("'\"")
            
            # Verificar si es un directorio
            if os.path.isdir(first_param_value):
                # Determinar si la función espera una imagen o una carpeta
                expects_image = check_if_function_expects_image(func_name, module_code)
                
                if expects_image:
                    # Procesar cada imagen en la carpeta y subcarpetas
                    all_results = process_images_in_folder(
                        first_param_value, first_param_name, param_value_match.group(1),
                        func_call, module_code, repetitions, selected_file_path
                    )
                else:
                    # Procesar cada subcarpeta (solo si la función espera carpetas)
                    all_results = process_subfolders(
                        first_param_value, first_param_name, param_value_match.group(1),
                        func_call, module_code, repetitions, selected_file_path
                    )
                
                return all_results
    
    # Ejecución normal si no es un directorio
    for rep in range(repetitions):
        result = execute_perf_measurement(module_code, func_call, selected_file_path)
        all_results.append(("Single execution", [result]))
    
    return all_results

def check_if_function_expects_image(func_name: str, module_code: str) -> bool:
    """Determina si la función espera una imagen como primer parámetro analizando el código"""
    # Buscar la definición de la función en el código
    func_pattern = rf"def {func_name}\((.*?)\):"
    match = re.search(func_pattern, module_code, re.DOTALL)
    
    if not match:
        return True  # Por defecto asumir que espera una imagen cargada
    
    params = match.group(1)
    # Verificar si el primer parámetro tiene pistas de ser una imagen
    first_param = params.split(',')[0].strip()
    
    # Pistas de que espera una imagen cargada (array numpy)
    image_param_hints = ['hologram', 'image', 'img', 'array', 'inp', 'input', 'U', 'data', 'frame', 'matriz', 'matrix', 'amp', 'phase', 'amplitude']
    # Pistas de que espera una ruta
    path_param_hints = ['path', 'file', 'filename', 'folder', 'directory', 'archivo', 'ruta', 'dir', 'carpeta']
    
    first_param_lower = first_param.lower()
    
    # Si el parámetro sugiere que es una ruta
    if any(hint in first_param_lower for hint in path_param_hints):
        return False
    
    # Si el parámetro sugiere que es una imagen cargada
    if any(hint in first_param_lower for hint in image_param_hints):
        return True
    
    # Analizar el cuerpo de la función para ver cómo se usa el primer parámetro
    func_body_pattern = rf"def {func_name}\(.*?\):(.*?)(?=def |\Z)"
    body_match = re.search(func_body_pattern, module_code, re.DOTALL)
    
    if body_match:
        func_body = body_match.group(1)
        # Si el parámetro se usa en operaciones numpy, probablemente es una imagen
        numpy_ops = ['np.', 'np.abs', 'np.angle', 'np.real', 'np.imag', 'plt.', 'np.array', 'np.mean', 'np.std', '.size', '.shape', 'np.load', 'Image.open']
        for op in numpy_ops:
            if op in func_body and first_param in func_body:
                return True
    
    return True  # Por defecto asumir que espera una imagen cargada

def convert_windows_path_to_wsl(windows_path):
    """Convierte una ruta de Windows a una ruta de WSL"""
    if not windows_path:
        return windows_path
    
    # Normalizar la ruta
    windows_path = windows_path.replace('\\', '/')
    
    # Convertir rutas de unidad de Windows a WSL
    m = re.match(r'^([A-Za-z]):/(.*)', windows_path)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        return f"/mnt/{drive}/{rest}"
    
    return windows_path

def process_images_in_folder(folder_path: str, param_name: str, original_param_value: str,
                           func_call: str, module_code: str, repetitions: int, file_path: str) -> list:
    """Procesa todas las imágenes en una carpeta y sus subcarpetas"""
    all_results = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.tiff', '.npy']
    
    # Extraer el nombre de la función para verificar qué espera
    func_name_match = re.search(r'^(\w+)\(', func_call)
    if func_name_match:
        func_name = func_name_match.group(1)
        expects_loaded_image = check_if_function_expects_image(func_name, module_code)
    else:
        expects_loaded_image = True
    
    # Recursivamente buscar archivos en TODAS las subcarpetas
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file.lower())[1]
            if file_ext in image_extensions:
                file_path_full = os.path.join(root, file)
                # Normalizar la ruta para usar barras diagonales
                file_path_full = file_path_full.replace('\\', '/')
                relative_path = os.path.relpath(file_path_full, folder_path).replace('\\', '/')
                
                file_results = []
                for rep in range(repetitions):
                    if expects_loaded_image:
                        # La función espera una imagen cargada - modificar la llamada
                        file_func_call = func_call.replace(
                            f"{param_name}={original_param_value}", 
                            f"{param_name}=np.array(Image.open(r'{file_path_full}').convert('L'), dtype=np.float32)"
                        )
                    else:
                        # La función espera una ruta - pasar la ruta directamente
                        file_func_call = func_call.replace(
                            f"{param_name}={original_param_value}", 
                            f"{param_name}=r'{file_path_full}'"
                        )
                    
                    result = execute_perf_measurement(module_code, file_func_call, file_path)
                    file_results.append(result)
                
                all_results.append((relative_path, file_results))
    
    return all_results

def process_subfolders(folder_path: str, param_name: str, original_param_value: str,
                     func_call: str, module_code: str, repetitions: int, file_path: str) -> list:
    """Procesa todas las subcarpetas"""
    all_results = []
    
    # Obtener todas las subcarpetas
    subfolders = [f for f in os.listdir(folder_path) 
                 if os.path.isdir(os.path.join(folder_path, f))]
    
    for folder in subfolders:
        subfolder_path = os.path.join(folder_path, folder)
        # Normalizar la ruta para usar barras diagonales
        subfolder_path = subfolder_path.replace('\\', '/')
        
        folder_results = []
        
        for rep in range(repetitions):
            # Reemplazar el parámetro con la ruta de la subcarpeta (correctamente escapada)
            folder_func_call = func_call.replace(
                f"{param_name}={original_param_value}", 
                f"{param_name}=r'{subfolder_path}'"
            )
            
            result = execute_perf_measurement(module_code, folder_func_call, file_path)
            folder_results.append(result)
        
        all_results.append((folder, folder_results))
    
    return all_results

def execute_perf_measurement(module_code: str, func_call: str, file_path: str):
    """Ejecuta la medición de performance replicando el código en un archivo temporal en /tmp de WSL"""
    file_dir = os.path.dirname(file_path) if file_path else os.getcwd()
    file_dir = file_dir.replace('\\', '/')
    file_dir_wsl = convert_windows_path_to_wsl(file_dir)

    # NO convertir rutas aquí - ya se hizo en run_analysis
    # Solo asegurarnos de que las rutas de Windows se conviertan a WSL
    modified_func_call = func_call
    
    # Convertir cualquier ruta de Windows restante a formato WSL
    # Esto es para rutas que no fueron convertidas a imágenes cargadas
    path_matches = re.findall(r'(\w+)=r\'([^\']+)\'', func_call)
    
    for param_name, path_value in path_matches:
        path_wsl = convert_windows_path_to_wsl(path_value)
        modified_func_call = modified_func_call.replace(
            f"{param_name}=r'{path_value}'",
            f"{param_name}=r'{path_wsl}'"
        )

    code_lines = [
        "# -*- coding: utf-8 -*-",
        "import sys, os",
        "import numpy as np",
        "from PIL import Image",
        f"sys.path.insert(0, r'{file_dir_wsl}')",
        "",
        module_code,
        "",
        "if __name__ == '__main__':",
        f"    result = {modified_func_call}"
    ]
    temp_code = "\n".join(code_lines)

    # --- Guardar directamente en /tmp dentro de WSL ---
    wsl_tmp_path = f"/tmp/temp_perf_{os.getpid()}.py"
    
    subprocess.run(["wsl", "bash", "-c", f"cat > {shlex.quote(wsl_tmp_path)}"], 
                   input=temp_code, text=True, encoding='utf-8')

    # --- Ejecutar con perf ---
    if platform.system() == "Windows":
        bash_command = (
            f"cd /tmp && "
            f"env -i PATH=/usr/lib/linux-tools-6.8.0-79:/usr/lib/linux-tools-6.8.0-79-generic:/usr/bin:/bin "
            f"perf stat -e cycles,instructions,cache-references,cache-misses python3 {shlex.quote(wsl_tmp_path)} 2>&1"
        )
        cmd = ["wsl", "bash", "-c", bash_command]
    else:
        cmd = [
            "perf", "stat",
            "-e", "cycles,instructions,cache-references,cache-misses",
            "python3", wsl_tmp_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # --- Borrar el archivo temporal en WSL ---
    if platform.system() == "Windows":
        subprocess.run(["wsl", "rm", "-f", wsl_tmp_path])
    else:
        os.remove(wsl_tmp_path)

    return (result.stdout or "") + "\n" + (result.stderr or "")

def remove_main_block(code: str) -> str:
    """
    Elimina el bloque if __name__ == "__main__" del código,
    pero mantiene todas las funciones y definiciones.
    """
    lines = code.split('\n')
    processed_lines = []
    in_main_block = False
    main_indentation = 0
    
    for line in lines:
        # Detectar el inicio del bloque main
        if re.match(r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', line):
            in_main_block = True
            main_indentation = len(line) - len(line.lstrip())
            continue
        
        # Si estamos en el bloque main, verificar si hemos salido
        if in_main_block:
            current_indentation = len(line) - len(line.lstrip())
            # Si la indentación actual es menor o igual a la del bloque main, hemos salido
            if current_indentation <= main_indentation and line.strip() != '':
                in_main_block = False
            else:
                continue  # Saltar líneas dentro del bloque main
        
        # Si no estamos en el bloque main, agregar la línea
        if not in_main_block:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

# --- Funciones para análisis de bytecodes ---
# --- Reemplazo robusto para análisis de bytecodes ---
import traceback

def build_bytecode_line_map(func_or_code):
    """
    Devuelve Counter {lineno: cantidad_de_opcodes} para la función o code object dado.
    Filtra pseudo-opcodes como CACHE/EXTENDED_ARG.
    """
    try:
        code_obj = func_or_code.__code__ if hasattr(func_or_code, "__code__") else func_or_code
        byte_map = collections.Counter()
        last_line = None
        for instr in dis.get_instructions(code_obj):
            if instr.opname in {"CACHE", "EXTENDED_ARG"}:
                continue
            if instr.starts_line is not None:
                last_line = instr.starts_line
            if last_line is not None:
                byte_map[last_line] += 1
        return byte_map
    except Exception as e:
        # si falla, devolver empty counter para no romper el flujo
        print(f"[build_bytecode_line_map] error: {e}")
        return collections.Counter()


def estimate_bytecodes_by_function(entry_func, *args, **kwargs):
    """
    Ejecuta entry_func(*args, **kwargs) midiendo:
      - Veces que se ejecuta cada línea (dinámico).
      - Estimación de bytecodes ejecutados por función = sum(hits_linea * opcodes_linea).
    Asegura incluir entry_func en el mapeo aunque algo falle con el filename.
    """
    target_filename = os.path.abspath(getattr(entry_func.__code__, "co_filename", ""))
    # intentar obtener el módulo que contiene la función
    module = inspect.getmodule(entry_func) or sys.modules.get(entry_func.__module__)
    func_static = {}

    # 1) recolectar funciones definidas en el mismo archivo (si podemos)
    if module is not None:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            try:
                codefile = os.path.abspath(obj.__code__.co_filename)
                if codefile == target_filename:
                    func_static[obj.__code__] = (obj.__name__, build_bytecode_line_map(obj))
            except Exception:
                continue

    # 2) asegurar que entry_func esté en func_static (por si co_filename no coincidía)
    try:
        if entry_func.__code__ not in func_static:
            func_static[entry_func.__code__] = (entry_func.__name__, build_bytecode_line_map(entry_func))
    except Exception:
        # en caso extremo, insertar un fallback con nombre genérico
        try:
            func_static[entry_func.__code__] = (getattr(entry_func, "__name__", "entry_func"), build_bytecode_line_map(entry_func))
        except Exception:
            pass

    line_hits = collections.Counter()
    exception_trace = None

    def tracer(frame, event, arg):
        # solo interesan eventos 'line'
        if event == "line":
            code_obj = frame.f_code
            if code_obj in func_static:
                line_hits[(code_obj, frame.f_lineno)] += 1
        return tracer

    # activar tracer
    sys.settrace(tracer)
    try:
        result = entry_func(*args, **kwargs)
    except Exception:
        # capturar traceback para escribirlo al fichero después
        exception_trace = traceback.format_exc()
        result = None
    finally:
        sys.settrace(None)

    # construir totales
    totals = collections.defaultdict(lambda: {"lines_executed": 0, "estimated_bytecodes": 0})
    per_line_detail = collections.defaultdict(list)

    for (code_obj, lineno), hits in line_hits.items():
        func_name, byte_map = func_static.get(code_obj, ("<unknown>", collections.Counter()))
        opcodes_in_line = byte_map.get(lineno, 0)
        est = hits * opcodes_in_line
        totals[func_name]["lines_executed"] += hits
        totals[func_name]["estimated_bytecodes"] += est
        per_line_detail[func_name].append((lineno, hits, opcodes_in_line, est))

    for fn in per_line_detail:
        per_line_detail[fn].sort(key=lambda t: t[0])

    return result, totals, per_line_detail, exception_trace


def run_bytecode_analysis(func, func_args, image_folder, output_file):
    """
    Ejecuta el análisis de bytecodes para todas las imágenes en una carpeta.
    - func: función Python (callable) que recibe primero la imagen (numpy array) y luego func_args (lista).
    - func_args: lista con parámetros adicionales (p.ej. [param1, param2, ...])
    - image_folder: carpeta raíz a recorrer recursivamente
    - output_file: ruta del archivo donde escribir resultados
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy']
    image_count = 0
    processed_count = 0

    # asegurarse de que func_args sea lista
    func_args = list(func_args) if func_args is not None else []

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ANÁLISIS DE BYTECODES - RESULTADOS\n")
        f.write("=" * 60 + "\n\n")
        f.flush()
        try:
            if not os.path.isdir(image_folder):
                f.write(f"ERROR: La carpeta de imágenes no existe: {image_folder}\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                return False

            for subdir, _, files in os.walk(image_folder):
                for fname in files:
                    if not any(fname.lower().endswith(ext) for ext in image_extensions):
                        continue
                    image_count += 1
                    img_path = os.path.join(subdir, fname)
                    f.write(f"\nProcesando imagen {image_count}: {img_path}\n")
                    f.write("-" * 50 + "\n")
                    f.flush()

                    try:
                        # soporte .npy (arrays guardados) y formato imagen
                        if fname.lower().endswith('.npy'):
                            hologram = np.load(img_path)
                        else:
                            with Image.open(img_path) as img:
                                # convertir a una sola banda (grayscale) para consistencia
                                img_l = img.convert('L')
                                hologram = np.array(img_l, dtype=np.float32)

                        # preparar args: la función espera primer parametro = imagen
                        args = [hologram] + func_args

                        # ejecutar análisis (tracer)
                        result, totals, per_line_detail, exc_trace = estimate_bytecodes_by_function(func, *args)

                        if exc_trace:
                            f.write("ERROR durante la ejecución de la función:\n")
                            f.write(exc_trace + "\n")
                        else:
                            if totals:
                                for fn, data in totals.items():
                                    f.write(f"Función: {fn}\n")
                                    f.write(f"  Líneas ejecutadas (suma hits): {data['lines_executed']}\n")
                                    f.write(f"  Bytecodes estimados ejecutados: {data['estimated_bytecodes']}\n")
                                    if per_line_detail.get(fn):
                                        f.write("  Detalle por línea:\n")
                                        for (lineno, hits, opcodes, est) in per_line_detail[fn]:
                                            f.write(f"    L{lineno}: hits={hits} opcodes={opcodes} total={est}\n")
                            else:
                                f.write("No se registraron datos de ejecución (posible: la función fue muy vectorizada o no hay líneas Python ejecutadas dentro del archivo).\n")

                        processed_count += 1

                    except Exception as e_img:
                        f.write(f"ERROR al procesar imagen {img_path}: {str(e_img)}\n")
                        f.write(traceback.format_exc() + "\n")
                    finally:
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass

            # resumen
            f.write("\nRESUMEN:\n")
            f.write(f"Imágenes encontradas: {image_count}\n")
            f.write(f"Imágenes procesadas exitosamente: {processed_count}\n")
            f.write(f"Imágenes con error: {image_count - processed_count}\n")
            f.write("\nANÁLISIS COMPLETADO\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

            return True

        except Exception as e:
            f.write("ERROR FATAL en run_bytecode_analysis:\n")
            f.write(traceback.format_exc() + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
            return False

# --- Nueva interfaz Tkinter ---
class PerformanceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Performance")
        self.root.geometry("900x700")
        
        self.function_module = None
        self.function_info = None
        self.function_args = {}
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.setup_ui()
        
    def setup_ui(self):
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestaña 1: Análisis de performance
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Análisis")
        self.setup_analysis_tab()
        
        # Pestaña 2: Visualización de resultados
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Visualización")
        self.setup_visualization_tab()
        
        # Pestaña 3: Análisis de Bytecodes
        self.bytecode_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bytecode_frame, text="Bytecodes")
        self.setup_bytecode_tab()
        
    def setup_analysis_tab(self):
        # Frame principal de configuración
        main_frame = ttk.Frame(self.analysis_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurar grid weights
        main_frame.columnconfigure(1, weight=1)
        
        # Selección de archivo Python
        ttk.Label(main_frame, text="Archivo Python:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(main_frame, textvariable=self.file_path, width=50)
        file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Buscar", command=self.browse_python_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Selección de función
        ttk.Label(main_frame, text="Función:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.function_var = tk.StringVar()
        function_combo = ttk.Combobox(main_frame, textvariable=self.function_var, state="readonly")
        function_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        function_combo.bind('<<ComboboxSelected>>', self.on_function_selected)
        self.function_combo = function_combo
        
        # Frame para parámetros
        self.param_frame = ttk.LabelFrame(main_frame, text="Parámetros de la función", padding="10")
        self.param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.param_frame.columnconfigure(1, weight=1)
        
        # Parámetros comunes
        ttk.Label(main_frame, text="Repeticiones:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.repetitions_var = tk.IntVar(value=5)
        ttk.Spinbox(main_frame, from_=1, to=20, textvariable=self.repetitions_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Archivo de salida:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.output_file_var = tk.StringVar(value="resultados_perf_shpc.txt")
        ttk.Entry(main_frame, textvariable=self.output_file_var, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Buscar", command=self.browse_output_file).grid(row=4, column=2, padx=5, pady=5)
        
        # Botón de ejecución
        self.run_button = ttk.Button(main_frame, text="Ejecutar Análisis", command=self.run_analysis, state="disabled")
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Área de log
        ttk.Label(main_frame, text="Log de ejecución:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.log_text = tk.Text(main_frame, height=15, width=80)
        self.log_text.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Scrollbar para el log
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=7, column=3, sticky=(tk.N, tk.S), pady=5)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
    def setup_visualization_tab(self):
        """Configurar la pestaña de visualización de resultados"""
        main_frame = ttk.Frame(self.visualization_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Frame para controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        
        # Botón para cargar archivo
        ttk.Button(control_frame, text="Cargar Archivo de Resultados", 
                  command=self.load_results_file).pack(side=tk.LEFT, padx=5)
        
        # Botón para generar gráficos
        ttk.Button(control_frame, text="Generar Gráficos", 
                  command=self.generate_plots).pack(side=tk.LEFT, padx=5)
        
        # Label para mostrar archivo cargado
        self.file_label = ttk.Label(control_frame, text="Ningún archivo cargado")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Frame para gráficos
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill='both', expand=True)
        
    def setup_bytecode_tab(self):
        """Configurar la pestaña de análisis de bytecodes"""
        main_frame = ttk.Frame(self.bytecode_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurar grid weights
        main_frame.columnconfigure(1, weight=1)
        
        # Selección de archivo Python
        ttk.Label(main_frame, text="Archivo Python:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.bytecode_file_path = tk.StringVar()
        file_entry = ttk.Entry(main_frame, textvariable=self.bytecode_file_path, width=50)
        file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Buscar", command=self.browse_bytecode_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Selección de función
        ttk.Label(main_frame, text="Función:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bytecode_function_var = tk.StringVar()
        function_combo = ttk.Combobox(main_frame, textvariable=self.bytecode_function_var, state="readonly")
        function_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        function_combo.bind('<<ComboboxSelected>>', self.on_bytecode_function_selected)
        self.bytecode_function_combo = function_combo
        
        # Frame para parámetros de bytecode
        self.bytecode_param_frame = ttk.LabelFrame(main_frame, text="Parámetros de la función", padding="10")
        self.bytecode_param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.bytecode_param_frame.columnconfigure(1, weight=1)
        
        # Carpeta de imágenes
        ttk.Label(main_frame, text="Carpeta de imágenes:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.bytecode_image_folder = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.bytecode_image_folder, width=50).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Buscar", command=lambda: self.browse_folder(self.bytecode_image_folder)).grid(row=3, column=2, padx=5, pady=5)
        
        # Archivo de salida
        ttk.Label(main_frame, text="Archivo de salida:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.bytecode_output_file = tk.StringVar(value="resultados_bytecodes.txt")
        ttk.Entry(main_frame, textvariable=self.bytecode_output_file, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Buscar", command=lambda: self.browse_output_file(self.bytecode_output_file)).grid(row=4, column=2, padx=5, pady=5)
        
        # Botón de ejecución
        self.run_bytecode_button = ttk.Button(main_frame, text="Ejecutar Análisis de Bytecodes", 
                                            command=self.run_bytecode_analysis, state="disabled")
        self.run_bytecode_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Área de log
        ttk.Label(main_frame, text="Log de ejecución:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.bytecode_log_text = tk.Text(main_frame, height=15, width=80)
        self.bytecode_log_text.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Scrollbar para el log
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.bytecode_log_text.yview)
        scrollbar.grid(row=7, column=3, sticky=(tk.N, tk.S), pady=5)
        self.bytecode_log_text.configure(yscrollcommand=scrollbar.set)
        
    def browse_bytecode_file(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo Python",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filepath:
            self.bytecode_file_path.set(filepath)
            self.load_bytecode_functions(filepath)
            
    def load_bytecode_functions(self, filepath):
        try:
            spec = importlib.util.spec_from_file_location("module.name", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.bytecode_module = module
            
            # Obtener todas las funciones del módulo
            functions = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    functions.append(name)
            
            self.bytecode_function_combo['values'] = functions
            if functions:
                self.bytecode_function_combo.set(functions[0])
                self.on_bytecode_function_selected()
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
            
    def on_bytecode_function_selected(self, event=None):
        if not self.bytecode_module:
            return
            
        function_name = self.bytecode_function_var.get()
        try:
            func = getattr(self.bytecode_module, function_name)
            sig = inspect.signature(func)
            
            # Limpiar frame de parámetros
            for widget in self.bytecode_param_frame.winfo_children():
                widget.destroy()
                
            self.bytecode_function_args = {}
            row = 0
            
            # Header
            headers = ["Parámetro", "Valor"]
            for col, header in enumerate(headers):
                ttk.Label(self.bytecode_param_frame, text=header, font=('Arial', 9, 'bold')).grid(
                    row=row, column=col, padx=2, pady=3)
            row += 1
            
            # Obtener lista de parámetros (omitir el primero que es la imagen)
            params = list(sig.parameters.items())[1:]  # Saltar el primer parámetro (imagen)
            
            for param_name, param in params:
                # Nombre del parámetro
                ttk.Label(self.bytecode_param_frame, text=param_name, font=('Arial', 9)).grid(
                    row=row, column=0, sticky=tk.W, padx=2, pady=2)
                
                # Entrada de valor
                var = tk.StringVar(value=str(param.default) if param.default != param.empty else "")
                entry = ttk.Entry(self.bytecode_param_frame, textvariable=var, width=30)
                entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
                self.bytecode_function_args[param_name] = var
                
                row += 1
            
            # Ajustar pesos de columnas
            self.bytecode_param_frame.columnconfigure(1, weight=1)
            
            self.run_bytecode_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo obtener información de la función: {e}")
            
    def run_bytecode_analysis(self):
        if not self.bytecode_module or not self.bytecode_function_var.get():
            messagebox.showerror("Error", "Seleccione una función primero")
            return
            
        if not self.bytecode_image_folder.get():
            messagebox.showerror("Error", "Seleccione una carpeta de imágenes")
            return
            
        function_name = self.bytecode_function_var.get()
        func = getattr(self.bytecode_module, function_name)
        
        # Obtener argumentos
        args = []
        for param_name, var in self.bytecode_function_args.items():
            try:
                # Intentar evaluar el valor (para números, listas, etc.)
                value = eval(var.get())
            except:
                # Si falla, usar el string directamente
                value = var.get()
            args.append(value)
        
        # Ejecutar análisis
        try:
            self.log_to_bytecode_console(f"Iniciando análisis de bytecodes para función: {function_name}")
            self.log_to_bytecode_console(f"Parámetros: {args}")
            self.log_to_bytecode_console(f"Carpeta de imágenes: {self.bytecode_image_folder.get()}")
            
            success = run_bytecode_analysis(
                func, 
                args, 
                self.bytecode_image_folder.get(), 
                self.bytecode_output_file.get()
            )
            
            if success:
                self.log_to_bytecode_console("Análisis completado exitosamente")
                messagebox.showinfo("Éxito", "Análisis de bytecodes completado")
            else:
                self.log_to_bytecode_console("Error en el análisis")
                messagebox.showerror("Error", "Ocurrió un error durante el análisis")
                
        except Exception as e:
            self.log_to_bytecode_console(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
            
    def log_to_bytecode_console(self, message):
        self.bytecode_log_text.insert(tk.END, message + "\n")
        self.bytecode_log_text.see(tk.END)
        self.root.update_idletasks()
        
    def browse_python_file(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo Python",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filepath:
            self.file_path.set(filepath)
            self.load_functions(filepath)
            
    def load_functions(self, filepath):
        try:
            spec = importlib.util.spec_from_file_location("module.name", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.function_module = module
            
            # Obtener todas las funciones del módulo
            functions = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    functions.append(name)
            
            self.function_combo['values'] = functions
            if functions:
                self.function_combo.set(functions[0])
                self.on_function_selected()
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
            
    def on_function_selected(self, event=None):
        if not self.function_module:
            return
            
        function_name = self.function_var.get()
        try:
            func = getattr(self.function_module, function_name)
            sig = inspect.signature(func)
            
            # Limpiar frame de parámetros
            for widget in self.param_frame.winfo_children():
                widget.destroy()
                
            self.function_args = {}
            row = 0
            
            # Header
            headers = ["Parámetro", "Valor", "Tipo"]
            for col, header in enumerate(headers):
                ttk.Label(self.param_frame, text=header, font=('Arial', 9, 'bold')).grid(
                    row=row, column=col, padx=2, pady=3)
            row += 1
            
            for param_name, param in sig.parameters.items():
                # Nombre del parámetro
                ttk.Label(self.param_frame, text=param_name, font=('Arial', 9)).grid(
                    row=row, column=0, sticky=tk.W, padx=2, pady=2)
                
                # Entrada de valor
                var = tk.StringVar(value=str(param.default) if param.default != param.empty else "")
                entry = ttk.Entry(self.param_frame, textvariable=var, width=30)
                entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
                self.function_args[param_name] = var
                
                # Tipo del parámetro
                param_type = str(param.annotation) if param.annotation != param.empty else "Any"
                ttk.Label(self.param_frame, text=param_type, font=('Arial', 9)).grid(
                    row=row, column=2, sticky=tk.W, padx=2, pady=2)
                
                row += 1
            
            # Ajustar pesos de columnas
            self.param_frame.columnconfigure(1, weight=1)
            
            self.run_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo obtener información de la función: {e}")
            
    def browse_output_file(self, var=None):
        if var is None:
            var = self.output_file_var
        filepath = filedialog.asksaveasfilename(
            title="Guardar resultados como",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            var.set(filepath)
            
    def browse_folder(self, var):
        folderpath = filedialog.askdirectory(title="Seleccionar carpeta")
        if folderpath:
            var.set(folderpath)
            
    def run_analysis(self):
        if not self.function_module or not self.function_var.get():
            messagebox.showerror("Error", "Seleccione una función primero")
            return
            
        function_name = self.function_var.get()
        func_call = f"{function_name}("
        
        # Construir la llamada a la función con los parámetros
        for param_name, var in self.function_args.items():
            param_value = var.get()
            # Si el valor está vacío, usar el valor por defecto
            if not param_value:
                continue
            # Si el valor parece ser numérico, no poner comillas
            if param_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                func_call += f"{param_name}={param_value}, "
            else:
                func_call += f"{param_name}=r'{param_value}', "
        
        func_call = func_call.rstrip(', ') + ")"
        
        # Guardar la ruta del archivo para usar en la medición
        measure_function_perf.selected_file_path = self.file_path.get()
        
        # Ejecutar la medición
        try:
            self.log_to_console(f"Ejecutando: {func_call}")
            self.log_to_console(f"Repeticiones: {self.repetitions_var.get()}")
            
            results = measure_function_perf(
                f"{self.function_module.__name__}.{function_name}",
                func_call,
                self.repetitions_var.get()
            )
            
            # Guardar resultados
            with open(self.output_file_var.get(), 'w', encoding='utf-8') as f:
                for name, rep_results in results:
                    f.write(f"=== {name} ===\n")
                    for i, result in enumerate(rep_results):
                        f.write(f"--- Repetición {i+1} ---\n")
                        f.write(result + "\n")
                    f.write("\n")
            
            self.log_to_console("Análisis completado. Resultados guardados en " + self.output_file_var.get())
            messagebox.showinfo("Éxito", "Análisis completado")
            
        except Exception as e:
            self.log_to_console(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
            
    def log_to_console(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_results_file(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de resultados",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.results_file = filepath
            self.file_label.config(text=f"Archivo: {os.path.basename(filepath)}")
            
    def generate_plots(self):
        """Generar gráficos a partir del archivo de resultados"""
        if not hasattr(self, 'results_file'):
            messagebox.showerror("Error", "Primero debe cargar un archivo de resultados")
            return
            
        try:
            # Limpiar área de gráficos
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            # Crear figura de matplotlib
            fig = self.create_performance_plots(self.results_file)
            
            if fig is None:
                messagebox.showerror("Error", "No se pudieron extraer datos del archivo de resultados")
                return
                
            # Mostrar en la interfaz
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Agregar toolbar de navegación
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
            self.log_message("Gráficos generados exitosamente")
            
        except Exception as e:
            error_msg = f"Error generando gráficos: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def parse_results_file(self, filepath):
        # Implementar el parsing del archivo de resultados
        # Esto es un placeholder - necesitarás implementar el parsing real
        results = {
            'cycles': [],
            'instructions': [],
            'cache_references': [],
            'cache_misses': []
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extraer métricas usando expresiones regulares
            cycles_matches = re.findall(r'(\d+,\d+|\d+)\s+cycles', content)
            instructions_matches = re.findall(r'(\d+,\d+|\d+)\s+instructions', content)
            cache_ref_matches = re.findall(r'(\d+,\d+|\d+)\s+cache-references', content)
            cache_miss_matches = re.findall(r'(\d+,\d+|\d+)\s+cache-misses', content)
            
            # Convertir a números
            for match in cycles_matches:
                results['cycles'].append(int(match.replace(',', '')))
            for match in instructions_matches:
                results['instructions'].append(int(match.replace(',', '')))
            for match in cache_ref_matches:
                results['cache_references'].append(int(match.replace(',', '')))
            for match in cache_miss_matches:
                results['cache_misses'].append(int(match.replace(',', '')))
                
        return results
        
    def create_performance_plots(self, txt_file):
        """Crear gráficos de performance a partir de archivo TXT - Versión corregida"""
        # Orden deseado de carpetas (resoluciones)
        resol_order = [
            "128x128", "256x256", "512x512", "640x480",
            "800x600", "1024x768", "1024x1024", "1280x960","1600x1200","1920x1440", "2048x2048", "2560x1920","3840x2880", "4096x4096"
        ]

        # Calcular pixeles totales por resolución
        resol_to_pixels = {}
        for r in resol_order:
            w, h = map(int, r.split("x"))
            resol_to_pixels[r] = w * h

        # Diccionarios para almacenar los datos
        data_cycles = defaultdict(list)
        data_instructions = defaultdict(list)
        data_cache_refs = defaultdict(list)
        data_cache_misses = defaultdict(list)
        data_time = defaultdict(list)

        # Expresiones regulares CORREGIDAS - el valor va primero, luego el nombre
        patterns = {
            'cycles': r'^\s*(\d+,\d+|\d+)\s+cycles:u',
            'instructions': r'^\s*(\d+,\d+|\d+)\s+instructions:u',
            'cache_refs': r'^\s*(\d+,\d+|\d+)\s+cache-references:u', 
            'cache_misses': r'^\s*(\d+,\d+|\d+)\s+cache-misses:u',
            'time': r'^\s*([0-9]*\.[0-9]+)\s+seconds time elapsed'
        }

        def parse_perf_line(line):
            """Parse una línea de output de perf y extrae las métricas"""
            metrics = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    val_str = match.group(1).replace(',', '')  # quitar comas si las hay
                    if key == 'time':
                        metrics[key] = float(val_str)
                    else:
                        metrics[key] = float(val_str)
            return metrics

        # Leer y procesar el archivo
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_folder = None
        current_image = None
        current_rep_data = {}
        in_perf_block = False

        for line in lines:
            line = line.strip()
            
            # Detectar nueva carpeta
            if "=== CARPETA:" in line:
                folder_match = re.search(r'=== CARPETA:\s+([^\n]+)', line)
                if folder_match:
                    current_folder = folder_match.group(1)
                    current_image = None
                    in_perf_block = False
            
            # Detectar nueva imagen
            elif "--- IMAGEN:" in line:
                image_match = re.search(r'--- IMAGEN:\s+([^\n]+)', line)
                if image_match and current_folder:
                    current_image = image_match.group(1)
                    in_perf_block = False
            
            # Detectar inicio de bloque de perf (Performance counter stats)
            elif "Performance counter stats for" in line:
                in_perf_block = True
                current_rep_data = {}
            
            # Detectar fin de bloque de perf (time elapsed)
            elif in_perf_block and "seconds time elapsed" in line:
                metrics = parse_perf_line(line)
                if 'time' in metrics and current_folder:
                    data_time[current_folder].append(metrics['time'])
                in_perf_block = False
            
            # Procesar líneas de métricas dentro del bloque perf
            elif in_perf_block and current_folder:
                metrics = parse_perf_line(line)
                if metrics:
                    if 'cycles' in metrics:
                        data_cycles[current_folder].append(metrics['cycles'])
                    if 'instructions' in metrics:
                        data_instructions[current_folder].append(metrics['instructions'])
                    if 'cache_refs' in metrics:
                        data_cache_refs[current_folder].append(metrics['cache_refs'])
                    if 'cache_misses' in metrics:
                        data_cache_misses[current_folder].append(metrics['cache_misses'])
        
        # Calcular promedios por resolución
        def calculate_stats(data_dict):
            means = {}
            stds = {}
            for resol in resol_order:
                if resol in data_dict and data_dict[resol]:
                    means[resol] = np.mean(data_dict[resol])
                    stds[resol] = np.std(data_dict[resol])
            return means, stds

        cycles_means, cycles_stds = calculate_stats(data_cycles)
        instructions_means, instructions_stds = calculate_stats(data_instructions)
        cache_refs_means, cache_refs_stds = calculate_stats(data_cache_refs)
        cache_misses_means, cache_misses_stds = calculate_stats(data_cache_misses)
        time_means, time_stds = calculate_stats(data_time)

        # ... el resto del código permanece igual hasta los gráficos ...
        # Limpiar los nombres de las carpetas (quitar " ===" al final)
        def clean_folder_name(name):
            if name.endswith(' ==='):
                return name[:-4]
            return name
        
        cleaned_data_cycles = {clean_folder_name(k): v for k, v in data_cycles.items()}
        cleaned_data_instructions = {clean_folder_name(k): v for k, v in data_instructions.items()}
        cleaned_data_cache_refs = {clean_folder_name(k): v for k, v in data_cache_refs.items()}
        cleaned_data_cache_misses = {clean_folder_name(k): v for k, v in data_cache_misses.items()}
        cleaned_data_time = {clean_folder_name(k): v for k, v in data_time.items()}
        
        # Usar los datos limpios en lugar de los originales
        cycles_means, cycles_stds = calculate_stats(cleaned_data_cycles)
        instructions_means, instructions_stds = calculate_stats(cleaned_data_instructions)
        cache_refs_means, cache_refs_stds = calculate_stats(cleaned_data_cache_refs)
        cache_misses_means, cache_misses_stds = calculate_stats(cleaned_data_cache_misses)
        time_means, time_stds = calculate_stats(cleaned_data_time)
        # Calcular métricas derivadas
        ipc_means = {}
        cache_miss_ratio_means = {}
        for resol in resol_order:
            if resol in cycles_means and resol in instructions_means:
                ipc_means[resol] = instructions_means[resol] / cycles_means[resol] if cycles_means[resol] != 0 else 0
            if resol in cache_refs_means and resol in cache_misses_means:
                cache_miss_ratio_means[resol] = (cache_misses_means[resol] / cache_refs_means[resol] * 100) if cache_refs_means[resol] != 0 else 0

        # --- Preparar datos para ajuste de complejidad ---
        # Filtrar solo las resoluciones que tienen datos
        available_resolutions = [r for r in resol_order if r in time_means]
        x_fit = np.array([resol_to_pixels[r] for r in available_resolutions])
        y_time_fit = np.array([time_means[r] for r in available_resolutions])

        # Funciones de ajuste
        def f_linear(N, a, b): 
            return a*N + b

        def f_quadratic(N, a, b): 
            return a*N**2 + b

        def f_nlogn(N, a, b): 
            return a*N*np.log(N) + b

        # Ajustes de complejidad para tiempo
        popt_time_lin, pcov_time_lin = None, None
        popt_time_quad, pcov_time_quad = None, None
        popt_time_nlogn, pcov_time_nlogn = None, None

        if len(x_fit) >= 2:
            try:
                popt_time_lin, pcov_time_lin = curve_fit(f_linear, x_fit, y_time_fit)
            except:
                pass
        
        if len(x_fit) >= 2:
            try:
                popt_time_quad, pcov_time_quad = curve_fit(f_quadratic, x_fit, y_time_fit)
            except:
                pass
        
        if len(x_fit) >= 2:
            try:
                popt_time_nlogn, pcov_time_nlogn = curve_fit(f_nlogn, x_fit, y_time_fit)
            except:
                pass

        # --- Graficar resultados ---
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Tiempo de ejecución por resolución
        x = range(len(resol_order))
        y_time = [time_means.get(r, 0) for r in resol_order]
        yerr_time = [time_stds.get(r, 0) for r in resol_order]
        xtick_labels = [f"{r}\n{resol_to_pixels[r]} px" for r in resol_order]

        axes[0,0].errorbar(x, y_time, yerr=yerr_time, fmt='o-', capsize=5, color='blue')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(xtick_labels, rotation=45)
        axes[0,0].set_ylabel('Tiempo (s)')
        axes[0,0].set_title('Tiempo de ejecución por resolución')
        axes[0,0].grid(True, linestyle='--', alpha=0.6)

        # 2. Ciclos de CPU por resolución
        y_cycles = [cycles_means.get(r, 0) for r in resol_order]
        yerr_cycles = [cycles_stds.get(r, 0) for r in resol_order]

        axes[0,1].errorbar(x, y_cycles, yerr=yerr_cycles, fmt='o-', capsize=5, color='red')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(xtick_labels, rotation=45)
        axes[0,1].set_ylabel('Ciclos de CPU')
        axes[0,1].set_title('Ciclos de CPU por resolución')
        axes[0,1].grid(True, linestyle='--', alpha=0.6)

        # 3. Instrucciones por resolución
        y_instructions = [instructions_means.get(r, 0) for r in resol_order]
        yerr_instructions = [instructions_stds.get(r, 0) for r in resol_order]

        axes[0,2].errorbar(x, y_instructions, yerr=yerr_instructions, fmt='o-', capsize=5, color='green')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(xtick_labels, rotation=45)
        axes[0,2].set_ylabel('Instrucciones')
        axes[0,2].set_title('Instrucciones ejecutadas por resolución')
        axes[0,2].grid(True, linestyle='--', alpha=0.6)

        # 4. IPC (Instructions Per Cycle)
        y_ipc = [ipc_means.get(r, 0) for r in resol_order]

        axes[1,0].plot(x, y_ipc, 'o-', color='purple')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(xtick_labels, rotation=45)
        axes[1,0].set_ylabel('IPC')
        axes[1,0].set_title('Instructions Per Cycle (IPC)')
        axes[1,0].grid(True, linestyle='--', alpha=0.6)

        # 5. Cache miss ratio
        y_miss_ratio = [cache_miss_ratio_means.get(r, 0) for r in resol_order]

        axes[1,1].plot(x, y_miss_ratio, 'o-', color='orange')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(xtick_labels, rotation=45)
        axes[1,1].set_ylabel('Cache Miss Ratio (%)')
        axes[1,1].set_title('Porcentaje de fallos de cache')
        axes[1,1].grid(True, linestyle='--', alpha=0.6)

        # 6. Ajuste de complejidad para tiempo
        if len(x_fit) > 0:
            axes[1,2].scatter(x_fit, y_time_fit, label='Datos', color='black')
            xx = np.linspace(min(x_fit), max(x_fit), 500)
            
            # Solo graficar ajustes si se calcularon exitosamente
            if popt_time_lin is not None:
                axes[1,2].plot(xx, f_linear(xx, *popt_time_lin), '--', label='O(N)')
            
            if popt_time_quad is not None:
                axes[1,2].plot(xx, f_quadratic(xx, *popt_time_quad), '--', label='O(N²)')
            
            if popt_time_nlogn is not None:
                axes[1,2].plot(xx, f_nlogn(xx, *popt_time_nlogn), '--', label='O(N log N)')

            axes[1,2].set_xlabel('Número de píxeles')
            axes[1,2].set_ylabel('Tiempo (s)')
            axes[1,2].set_title('Ajuste de complejidad computacional (Tiempo)')
            axes[1,2].legend()
            axes[1,2].grid(True, linestyle='--', alpha=0.6)
        else:
            axes[1,2].text(0.5, 0.5, 'No hay datos suficientes\npara el ajuste de complejidad', 
                        ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Ajuste de complejidad computacional (Tiempo)')

        plt.tight_layout()
        return fig
        
    def on_closing(self):
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PerformanceAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()