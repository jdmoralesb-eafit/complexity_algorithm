import os
import subprocess
import platform
import re
import shlex
from PIL import Image
import numpy as np
import wx
import wx.aui
import inspect
import importlib.util
import sys
import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas, NavigationToolbar2WxAgg as NavigationToolbar
import importlib
import perf_protocol
from visualizer_v2 import (
    analyze_file, format_verdict_report, create_plots as create_plots_v2,
    METRICS,
)
from complexity_fit import ShapeFilter
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
    import matplotlib
matplotlib.use("WXAgg")
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
            result = execute_perf_measurement("", func_call, None)  # Sin imagen
            all_results.append(("Single execution", [result]))
        return all_results
    
    try:
        with open(selected_file_path, 'r', encoding='utf-8') as f:
            module_code = f.read()
    except FileNotFoundError:
        return ["Error: File not found"] * repetitions
    except Exception as e:
        return [f"Error reading file: {e}"] * repetitions
    
    all_results = []
    
    # Extraer el nombre de la función
    func_name_match = re.search(r'^(\w+)\(', func_call)
    if not func_name_match:
        # Ejecución normal SIN imagen
        for rep in range(repetitions):
            result = execute_perf_measurement(module_code, func_call, None)
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
    
    # Ejecución normal si no es un directorio - SIN imagen
    for rep in range(repetitions):
        result = execute_perf_measurement(module_code, func_call, None)
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
                    
                    result = execute_perf_measurement(module_code, file_func_call, file_path_full)
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
            
            result = execute_perf_measurement(module_code, folder_func_call, subfolder_path)
            folder_results.append(result)
        
        all_results.append((folder, folder_results))
    
    return all_results

def execute_perf_measurement(module_code: str, func_call: str, file_path: str):
    """Ejecuta la medición de performance replicando el código en un archivo temporal en /tmp de WSL"""
    file_dir = os.path.dirname(file_path) if file_path else os.getcwd()
    file_dir = file_dir.replace('\\', '/')
    file_dir_wsl = convert_windows_path_to_wsl(file_dir)

    # Convertir la ruta de la imagen a WSL
    image_path_wsl = convert_windows_path_to_wsl(file_path) if file_path else ""

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
    ]
    
    # Si hay una ruta de imagen, cargarla directamente y mostrar dimensiones
    if file_path and image_path_wsl:
        code_lines.extend([
            "    # Obtener dimensiones de la imagen",
            "    try:",
            f"        img = Image.open(r'{image_path_wsl}')",
            "        w, h = img.size",
            "        print(f'IMAGE_DIMENSIONS: {w}x{h}')",
            "    except Exception as e:",
            "        print(f'IMAGE_DIMENSIONS_ERROR: {e}')",
            f"    result = {modified_func_call}"
        ])
    else:
        # Si no hay ruta de imagen, ejecutar directamente
        code_lines.extend([
            f"    result = {modified_func_call}",
            "    print('IMAGE_DIMENSIONS: No image path provided')"
        ])

    temp_code = "\n".join(code_lines)
    return execute_perf_code(temp_code)

def execute_perf_code(temp_code: str) -> str:
    """Ejecuta un script ya construido bajo perf stat y devuelve su salida."""
    # --- Guardar directamente en /tmp dentro de WSL ---
    wsl_tmp_path = f"/tmp/temp_perf_{os.getpid()}.py"
    
    subprocess.run(["wsl", "bash", "-c", f"cat > {shlex.quote(wsl_tmp_path)}"], 
                   input=temp_code, text=True, encoding='utf-8')

    # --- Ejecutar con perf ---
    if platform.system() == "Windows":
        # Ya no necesitamos pasar argumentos porque la imagen se carga directamente
        bash_command = (
            f"cd /tmp && "
            f"perf stat -e cycles,instructions,cache-references,cache-misses python3 {shlex.quote(wsl_tmp_path)} 2>&1 | "
            f"grep -v 'WARNING: perf not found for kernel'"
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

def image_dims(path: str):
    """(ancho, alto) de una imagen o .npy; None si no se puede leer."""
    try:
        if path.lower().endswith('.npy'):
            shape = np.load(path, mmap_mode='r').shape
            return int(shape[-1]), int(shape[0])
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None

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

# --- Nueva interfaz Tkinter ---
FOLDER_TOOLTIP = "Select the folder containing the benchmark images"


def msg_error(title, text):
    wx.MessageBox(str(text), title, wx.OK | wx.ICON_ERROR)


def msg_warning(title, text):
    wx.MessageBox(str(text), title, wx.OK | wx.ICON_WARNING)


def msg_info(title, text):
    wx.MessageBox(str(text), title, wx.OK | wx.ICON_INFORMATION)


class PerformanceAnalyzerApp(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="Performance and Complexity Analyzer", size=(900, 700))

        self.function_module = None
        self.function_info = None
        self.function_args = {}
        self.function_flags = {}

        # Variables para almacenar la figura actual y datos
        self.current_figure = None
        self.current_fit_data = None
        self.current_txt_file = None

        # Inicializar lista para múltiples archivos
        self.loaded_txt_files = []  # Lista para múltiples archivos
        self.current_figures = {}   # Diccionario para figuras por archivo
        self.current_fit_data = {}  # Diccionario para datos de ajuste

        self.CreateStatusBar()
        self.setup_menus()
        self.setup_ui()
        self.Centre()

    def setup_menus(self):
        menubar = wx.MenuBar()

        file_menu = wx.Menu()
        open_item = file_menu.Append(wx.ID_OPEN, "Open Python File...\tCtrl+O")
        load_item = file_menu.Append(wx.ID_ANY, "Load Results Files...\tCtrl+L")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "Exit")
        menubar.Append(file_menu, "&File")

        analysis_menu = wx.Menu()
        self.run_item = analysis_menu.Append(wx.ID_ANY, "Run Analysis\tCtrl+R")
        self.run_item.Enable(False)
        plots_item = analysis_menu.Append(wx.ID_ANY, "Generate Plots\tCtrl+G")
        menubar.Append(analysis_menu, "&Analysis")

        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.browse_python_file, open_item)
        self.Bind(wx.EVT_MENU, self.load_results_file, load_item)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), exit_item)
        self.Bind(wx.EVT_MENU, lambda e: self.run_analysis(), self.run_item)
        self.Bind(wx.EVT_MENU, lambda e: self.generate_plots(), plots_item)

    def setup_ui(self):
        # Notebook (pestañas)
        self.notebook = wx.aui.AuiNotebook(self, style=wx.aui.AUI_NB_TOP | wx.aui.AUI_NB_TAB_MOVE)

        # Pestaña 1: Análisis de performance
        self.analysis_panel = wx.Panel(self.notebook)
        self.notebook.AddPage(self.analysis_panel, "Analysis")
        self.setup_analysis_tab()

        # Pestaña 2: Visualización de resultados
        self.visualization_panel = wx.Panel(self.notebook)
        self.notebook.AddPage(self.visualization_panel, "Visualization")
        self.setup_visualization_tab()

    def setup_analysis_tab(self):
        panel = self.analysis_panel
        outer = wx.BoxSizer(wx.VERTICAL)

        grid = wx.FlexGridSizer(cols=3, vgap=6, hgap=6)
        grid.AddGrowableCol(1)

        # Selección de archivo Python
        grid.Add(wx.StaticText(panel, label="Python File:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.file_ctrl = wx.TextCtrl(panel)
        grid.Add(self.file_ctrl, 1, wx.EXPAND)
        browse_py = wx.Button(panel, label="Browse")
        browse_py.Bind(wx.EVT_BUTTON, self.browse_python_file)
        grid.Add(browse_py)

        # Selección de función
        grid.Add(wx.StaticText(panel, label="Function:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.function_combo = wx.Choice(panel)
        self.function_combo.Bind(wx.EVT_CHOICE, self.on_function_selected)
        grid.Add(self.function_combo, 1, wx.EXPAND)
        grid.Add((0, 0))

        outer.Add(grid, 0, wx.EXPAND | wx.ALL, 10)

        # Frame para parámetros
        param_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Function Parameters")
        self.param_panel = wx.Panel(param_box.GetStaticBox())
        self.param_grid = wx.FlexGridSizer(cols=4, vgap=4, hgap=6)
        self.param_grid.AddGrowableCol(1)
        self.param_panel.SetSizer(self.param_grid)
        param_box.Add(self.param_panel, 1, wx.EXPAND | wx.ALL, 5)
        outer.Add(param_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Parámetros comunes
        common = wx.FlexGridSizer(cols=3, vgap=6, hgap=6)
        common.AddGrowableCol(1)

        common.Add(wx.StaticText(panel, label="Repetitions:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.repetitions_ctrl = wx.SpinCtrl(panel, min=1, max=20, initial=5, size=(80, -1))
        common.Add(self.repetitions_ctrl)
        common.Add((0, 0))

        common.Add(wx.StaticText(panel, label="Output file:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.output_ctrl = wx.TextCtrl(panel, value="performance.txt")
        common.Add(self.output_ctrl, 1, wx.EXPAND)
        browse_out = wx.Button(panel, label="Browse")
        browse_out.Bind(wx.EVT_BUTTON, self.browse_output_file)
        common.Add(browse_out)

        outer.Add(common, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # Botón de ejecución
        self.run_button = wx.Button(panel, label="Run Analysis")
        self.run_button.Enable(False)
        self.run_button.Bind(wx.EVT_BUTTON, lambda e: self.run_analysis())
        outer.Add(self.run_button, 0, wx.ALIGN_CENTER | wx.ALL, 15)

        # Área de log
        log_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Execution Log")
        self.log_text = wx.TextCtrl(log_box.GetStaticBox(),
                                    style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        log_box.Add(self.log_text, 1, wx.EXPAND | wx.ALL, 5)
        outer.Add(log_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(outer)

    def setup_visualization_tab(self):
        """Configurar la pestaña de visualización de resultados"""
        panel = self.visualization_panel
        outer = wx.BoxSizer(wx.VERTICAL)

        # Barra de controles
        controls = wx.BoxSizer(wx.HORIZONTAL)

        load_btn = wx.Button(panel, label="Load Results Files")
        load_btn.Bind(wx.EVT_BUTTON, self.load_results_file)
        controls.Add(load_btn, 0, wx.RIGHT, 5)

        self.generate_btn = wx.Button(panel, label="Generate Plots")
        self.generate_btn.Bind(wx.EVT_BUTTON, lambda e: self.generate_plots())
        controls.Add(self.generate_btn, 0, wx.RIGHT, 5)

        self.clear_btn = wx.Button(panel, label="Clear Files")
        self.clear_btn.Bind(wx.EVT_BUTTON, lambda e: self.clear_loaded_files())
        controls.Add(self.clear_btn, 0, wx.RIGHT, 5)

        self.download_btn = wx.Button(panel, label="Download Plot")
        self.download_btn.Bind(wx.EVT_BUTTON, lambda e: self.download_plot())
        controls.Add(self.download_btn, 0, wx.RIGHT, 10)

        # Label para mostrar archivos cargados
        self.file_label = wx.StaticText(panel, label="No files loaded")
        controls.Add(self.file_label, 1, wx.ALIGN_CENTER_VERTICAL)

        outer.Add(controls, 0, wx.EXPAND | wx.ALL, 10)

        # Splitter: gráficos arriba, log del visualizador abajo
        splitter = wx.SplitterWindow(panel, style=wx.SP_LIVE_UPDATE)
        splitter.SetMinimumPaneSize(80)
        splitter.SetSashGravity(0.75)

        self.plot_panel = wx.Panel(splitter)
        self.plot_sizer = wx.BoxSizer(wx.VERTICAL)
        self.plot_panel.SetSizer(self.plot_sizer)

        viz_log_panel = wx.Panel(splitter)
        viz_log_sizer = wx.StaticBoxSizer(wx.VERTICAL, viz_log_panel, "Log")
        self.viz_log_text = wx.TextCtrl(viz_log_sizer.GetStaticBox(),
                                        style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        viz_log_sizer.Add(self.viz_log_text, 1, wx.EXPAND | wx.ALL, 5)
        viz_log_panel.SetSizer(viz_log_sizer)

        splitter.SplitHorizontally(self.plot_panel, viz_log_panel, 430)
        outer.Add(splitter, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(outer)

        # Inicializar estado de botones
        self.update_buttons_state()

    def browse_python_file(self, event=None):
        with wx.FileDialog(self, "Select Python File",
                           wildcard="Python files (*.py)|*.py|All files (*.*)|*.*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            filepath = dlg.GetPath()
        self.file_ctrl.SetValue(filepath)
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

            self.function_combo.Set(functions)
            if functions:
                self.function_combo.SetSelection(0)
                self.on_function_selected()

        except Exception as e:
            msg_error("Error", f"Could not load the file: {e}")

    def get_function_name(self):
        return self.function_combo.GetStringSelection()

    def is_function_param(self, param_name):
        """True si el usuario marcó el parámetro como función (el parámetro de imágenes no tiene casilla)."""
        checkbox = self.function_flags.get(param_name)
        return bool(checkbox and checkbox.GetValue())

    def on_function_selected(self, event=None):
        if not self.function_module:
            return

        function_name = self.get_function_name()
        try:
            func = getattr(self.function_module, function_name)
            sig = inspect.signature(func)

            # Limpiar panel de parámetros
            self.param_grid.Clear(True)

            self.function_args = {}
            self.function_flags = {}
            panel = self.param_panel

            # Header con el nuevo orden
            for header in ["Parameter", "Value", "", "Is it a function?"]:
                label = wx.StaticText(panel, label=header)
                font = label.GetFont()
                font.SetWeight(wx.FONTWEIGHT_BOLD)
                label.SetFont(font)
                self.param_grid.Add(label, 0, wx.ALIGN_CENTER)

            # Obtener lista de parámetros
            params = list(sig.parameters.items())

            for i, (param_name, param) in enumerate(params):
                # Columna 0: Nombre del parámetro
                self.param_grid.Add(wx.StaticText(panel, label=param_name), 0, wx.ALIGN_CENTER_VERTICAL)

                # Columna 1: Entrada de valor. El parámetro de imágenes solo se llena con el botón de carpeta.
                default = str(param.default) if param.default != param.empty else ""
                is_image_param = param_name.lower() in ['folder', 'path', 'directory', 'image_path'] or i == 0
                style = wx.TE_READONLY if is_image_param else 0
                entry = wx.TextCtrl(panel, value=default, size=(250, -1), style=style)
                self.param_grid.Add(entry, 1, wx.EXPAND)
                self.function_args[param_name] = entry

                # Columna 2: Botón de explorar (solo para el parámetro de imágenes)
                if is_image_param:
                    bitmap = wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_BUTTON)
                    browse_btn = wx.BitmapButton(panel, bitmap=bitmap)
                    browse_btn.SetToolTip(FOLDER_TOOLTIP)
                    browse_btn.Bind(wx.EVT_BUTTON, lambda evt, c=entry: self.browse_folder(c))
                    self.param_grid.Add(browse_btn, 0, wx.ALIGN_CENTER)
                else:
                    self.param_grid.Add((0, 0))

                # Columna 3: Casilla de función; no aplica al parámetro de imágenes
                if is_image_param:
                    self.param_grid.Add((0, 0))
                else:
                    checkbox = wx.CheckBox(panel)
                    self.param_grid.Add(checkbox, 0, wx.ALIGN_CENTER)
                    self.function_flags[param_name] = checkbox

            self.analysis_panel.Layout()

            self.run_button.Enable(True)
            self.run_item.Enable(True)

        except Exception as e:
            msg_error("Error", f"Could not retrieve function information: {e}")

    def browse_output_file(self, event=None):
        with wx.FileDialog(self, "Save results as",
                           wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.output_ctrl.SetValue(dlg.GetPath())

    def browse_folder(self, ctrl):
        with wx.DirDialog(self, FOLDER_TOOLTIP,
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                ctrl.SetValue(dlg.GetPath())

    def run_analysis(self):
        if not verify_dependencies_once():
            msg_error("Error", "Missing dependencies in WSL. Check the console.")
            return
            
        # Construir llamada a la función
        args_dict = {}
        for param_name, ctrl in self.function_args.items():
            value = ctrl.GetValue()
            
            # Verificar si el usuario marcó este parámetro como función
            is_function = self.is_function_param(param_name)
            
            if is_function:
                # Es una función, usar el nombre directamente (sin comillas)
                args_dict[param_name] = value
            else:
                # Convertir a número/None si es posible
                try:
                    if value.lower() == 'none':
                        value = None
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, AttributeError):
                    pass  # Mantener como string
                args_dict[param_name] = value
        
        # Construir string de llamada a función CORRECTAMENTE
        func_call_parts = []
        for param_name, value in args_dict.items():
            # Si está marcado como función, no usar comillas
            if self.is_function_param(param_name):
                func_call_parts.append(f"{param_name}={value}")  # Sin comillas
            else:
                func_call_parts.append(f"{param_name}={repr(value)}")  # Con comillas si es string
        
        original_func_call = f"{self.get_function_name()}("
        original_func_call += ", ".join(func_call_parts)
        original_func_call += ")"
        
        # Mostrar en log para debugging
        self.log_message(f"Constructed call: {original_func_call}")
        
        # Obtener el PRIMER parámetro (asumiendo que es el directorio)
        first_param_name = list(args_dict.keys())[0]
        first_param_value = args_dict[first_param_name]
        
        # Verificar que el primer parámetro es un directorio válido
        if not isinstance(first_param_value, str) or not os.path.isdir(first_param_value):
            msg_error("Error", f"The first parameter '{first_param_name}' must be a valid directory.")
            return
        
        # Normalizar la ruta
        first_param_value = first_param_value.replace('\\', '/')
        
        # Obtener todas las subcarpetas del directorio
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy']
        subfolders = [f for f in os.listdir(first_param_value)
                    if os.path.isdir(os.path.join(first_param_value, f))]

        if not subfolders:
            msg_error("Error", f"The directory '{first_param_value}' does not contain subfolders.")
            return

        # Ordenar subcarpetas de menor a mayor tamaño de imagen
        folder_plan = []  # (folder, dims, images)
        for folder in subfolders:
            folder_path = os.path.join(first_param_value, folder).replace('\\', '/')
            images = sorted(
                os.path.join(folder_path, f).replace('\\', '/')
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and any(f.lower().endswith(ext) for ext in image_extensions)
            )
            dims = image_dims(images[0]) if images else None
            folder_plan.append((folder, dims, images))
        folder_plan.sort(key=lambda t: (t[1] is None, t[1][0] * t[1][1] if t[1] else 0, t[0]))
        subfolders = [t[0] for t in folder_plan]
        
        # Leer el código del archivo Python una vez
        try:
            with open(self.file_ctrl.GetValue(), 'r', encoding='utf-8') as f:
                module_code = f.read()
        except Exception as e:
            msg_error("Error", f"Could not read the Python file: {e}")
            return
        
        self.log_message(f"Starting performance analysis...")
        self.log_message(f"Function: {self.get_function_name()}")
        self.log_message(f"Repetitions: {self.repetitions_ctrl.GetValue()}")
        self.log_message(f"Base directory: {first_param_value}")
        self.log_message(f"Subfolders found: {len(subfolders)}")
        
        output_file = self.output_ctrl.GetValue()
        total_items = 0
        total_errors = 0
        n_folders = len(folder_plan)

        try:
            # Linea base de librerias (una vez): L
            self.log_message(f"Measuring libs baseline ({perf_protocol.LIBS_REPS} runs)...")
            libs_baseline_text = perf_protocol.measure_libs_baseline(
                execute_perf_code, module_code,
                convert_windows_path_to_wsl(first_param_value),
                log=self.log_message,
            )

            # Cabecera: el archivo se va completando a medida que hay resultados
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Performance analysis - {self.get_function_name()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analyzed directory: {first_param_value}\n")
                f.write(f"Original parameters: {args_dict}\n\n")
                f.write("--- BASELINE L (libs without image or call) ---\n")
                f.write(libs_baseline_text)
                f.write("\n")

            # Recorrer cada subcarpeta, de menor a mayor tamaño de imagen
            for fi, (folder, dims, folder_images) in enumerate(folder_plan, 1):
                dims_txt = f"{dims[0]}x{dims[1]}" if dims else "?"
                if not folder_images:
                    self.log_message(f"No images found in {folder}")
                    print(f"[folder {fi}/{n_folders}] {folder}: no images", flush=True)
                    continue

                self.log_message(f"Processing folder {fi}/{n_folders}: {folder} ({dims_txt})")
                print(f"\n[folder {fi}/{n_folders}] {folder} ({dims_txt}, "
                      f"{len(folder_images)} images)", flush=True)

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"=== FOLDER: {folder} ===\n")

                for ii, image_path in enumerate(folder_images, 1):
                    self.log_message(f"    Processing: {os.path.basename(image_path)}")
                    print(f"  [image {ii}/{len(folder_images)}] {os.path.basename(image_path)}", flush=True)

                    image_path_wsl = convert_windows_path_to_wsl(image_path)

                    # Protocolo v2: LI / L1 / LK por imagen (ver perf_protocol.py)
                    image_expr = f"np.array(Image.open(r'{image_path_wsl}').convert('L'), dtype=np.float32)"
                    call_template = original_func_call.replace(
                        f"{first_param_name}={repr(args_dict[first_param_name])}",
                        f"{first_param_name}=_img"
                    )
                    image_results = [perf_protocol.measure_image(
                        execute_perf_code, module_code,
                        convert_windows_path_to_wsl(os.path.dirname(image_path).replace('\\', '/')),
                        image_path_wsl, image_expr, call_template,
                        self.repetitions_ctrl.GetValue(), log=self.log_message,
                    )]

                    # Guardar de inmediato el resultado de esta imagen
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(f"--- IMAGE: {os.path.basename(image_path)} ---\n")
                        for i, result in enumerate(image_results, 1):
                            f.write(f"  Repetition {i}:\n")
                            f.write(result)
                            f.write("\n")
                            if "Error:" in result or "Traceback" in result:
                                total_errors += 1
                            total_items += 1
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write("\n")

            # Resumen estadístico
            if total_items > 0:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"=== SUMMARY  ===\n")
                    f.write(f"Total runs: {total_items}\n")
                    f.write(f"Errors found: {total_errors}\n")
                    f.write(f"Success rate: {(total_items - total_errors) / total_items * 100:.2f}%\n")

            self.log_message(f"Analysis completed. Results saved in: {output_file}")
            msg_info("Success", f"Analysis completed. Results in: {output_file}")
            
        except Exception as e:
            self.log_message(f"Error during analysis: {e}")
            msg_error("Error", f"Error during execution: {e}")
            
    def load_results_file(self, event=None):
        """Cargar archivo de resultados a la lista"""
        with wx.FileDialog(self, "Select results files",
                           wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
                           style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            filepaths = dlg.GetPaths()

        for filepath in filepaths:
            if filepath not in self.loaded_txt_files:
                self.loaded_txt_files.append(filepath)
                self.viz_log_message(f"Loaded: {os.path.basename(filepath)}")

        self.update_file_list_display()

        # Habilitar botones que requieren archivos
        self.update_buttons_state()
        self.notebook.SetSelection(1)

    def clear_loaded_files(self):
        """Limpiar lista de archivos cargados"""
        if self.loaded_txt_files:
            file_list = ", ".join([os.path.basename(f) for f in self.loaded_txt_files])
            self.loaded_txt_files.clear()
            self.update_file_list_display()

            # Limpiar también el área de gráficos
            self.plot_sizer.Clear(True)
            self.plot_panel.Layout()

            self.current_figure = None
            self.current_fit_data = {}
            self.update_buttons_state()

            self.viz_log_message(f"Cleared files: {file_list}")
        else:
            self.viz_log_message("No files to clear")

    def update_file_list_display(self):
        """Actualizar display de archivos cargados"""
        if self.loaded_txt_files:
            file_list = "\n".join([f"• {os.path.basename(f)}" for f in self.loaded_txt_files])
            self.file_label.SetLabel(f"Files loaded: {len(self.loaded_txt_files)}\n{file_list}")
        else:
            self.file_label.SetLabel("No files loaded")
        self.SetStatusText(f"{len(self.loaded_txt_files)} results file(s) loaded")
        self.visualization_panel.Layout()

    def update_buttons_state(self):
        """Actualizar estado de los botones según los archivos cargados"""
        has_files = len(self.loaded_txt_files) > 0

        # Botón de generar gráficos - solo habilitado si hay archivos
        self.generate_btn.Enable(has_files)

        # Botón de descarga - solo habilitado si hay un gráfico actual
        has_plot = hasattr(self, 'current_figure') and self.current_figure is not None
        self.download_btn.Enable(has_plot)

        # Botón de limpiar - siempre habilitado
        self.clear_btn.Enable(True)

    def show_figure(self, fig):
        """Mostrar una figura de matplotlib (con barra de navegación) en el panel de gráficos"""
        self.plot_sizer.Clear(True)
        canvas = FigureCanvas(self.plot_panel, -1, fig)
        toolbar = NavigationToolbar(canvas)
        toolbar.Realize()
        self.plot_sizer.Add(canvas, 1, wx.EXPAND)
        self.plot_sizer.Add(toolbar, 0, wx.EXPAND)
        self.plot_panel.Layout()

        # La figura siempre ocupa el espacio del panel: al cambiar el tamaño
        # (ventana, divisor del log) se reacomodan ejes, leyendas y márgenes.
        def refit(event):
            event.Skip()  # el canvas ajusta el tamaño de la figura primero
            def relayout():
                try:
                    fig.tight_layout()
                    canvas.draw_idle()
                except Exception:
                    pass  # panel demasiado chico: se reintenta en el próximo resize
            wx.CallAfter(relayout)
        canvas.Bind(wx.EVT_SIZE, refit)

        canvas.draw()

    def generate_plots(self):
        """Analizar los archivos de resultados cargados con complexity_fit y graficar"""
        if not self.loaded_txt_files:
            msg_error("Error", "You must load at least one results file first")
            return

        shape_filter = ShapeFilter.NONE  # sin filtros: se usan todos los puntos

        try:
            self.plot_sizer.Clear(True)

            analysis_results = []
            parse_errors = []
            for txt_file in self.loaded_txt_files:
                result = analyze_file(txt_file, shape_filter)
                if "parse_error" in result:
                    self.viz_log_message(f"\n=== {result['filename']}: ERROR ===")
                    self.viz_log_message(f"  {result['parse_error']}")
                    parse_errors.append(f"{result['filename']}:\n  {result['parse_error']}")
                    continue
                analysis_results.append(result)
                self.viz_log_message(f"\n=== Analyzed: {result['filename']} "
                                     f"(strategy={shape_filter.value}) ===")
                for field_name, _ in METRICS:
                    m = result["metrics"][field_name]
                    self.viz_log_message(format_verdict_report(result["filename"], m["label"], m["verdict"]))

            if parse_errors:
                msg_warning("Files without usable data", "\n\n".join(parse_errors))

            if not analysis_results:
                if not parse_errors:
                    msg_error("Error", "No valid performance data found in loaded files")
                return

            fig = create_plots_v2(analysis_results)
            self.show_figure(fig)

            self.current_figure = fig
            self.current_fit_data = analysis_results
            self.update_buttons_state()
            self.visualization_panel.Layout()
            self.viz_log_message(f"\nPlots generated for {len(analysis_results)} file(s).")

        except Exception as e:
            error_msg = f"Error generating plots: {str(e)}"
            self.viz_log_message(error_msg)
            msg_error("Error", error_msg)

    def download_plot(self):
        """Descargar la figura actual"""
        if self.current_figure is None:
            msg_warning("Warning", "No plot available to download.")
            return

        try:
            # Sugerir nombre por defecto basado en el archivo TXT
            if self.current_txt_file:
                default_name = os.path.splitext(self.current_txt_file)[0] + "_performance_plot.png"
            else:
                default_name = "performance_plot.png"

            wildcard = ("PNG files (*.png)|*.png|PDF files (*.pdf)|*.pdf|SVG files (*.svg)|*.svg|"
                        "JPEG files (*.jpg)|*.jpg|All files (*.*)|*.*")
            with wx.FileDialog(self, "Save plot", defaultFile=os.path.basename(default_name),
                               wildcard=wildcard,
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
                if dlg.ShowModal() != wx.ID_OK:
                    return
                file_path = dlg.GetPath()

            # Determinar formato basado en extensión
            format_map = {
                '.png': 'png',
                '.pdf': 'pdf',
                '.svg': 'svg',
                '.jpg': 'jpeg',
                '.jpeg': 'jpeg'
            }
            ext = os.path.splitext(file_path)[1].lower()
            file_format = format_map.get(ext, 'png')

            # Guardar con alta calidad
            self.current_figure.savefig(
                file_path,
                format=file_format,
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )

            msg_info("Success", f"Plot saved successfully as:\n{file_path}")

        except Exception as e:
            msg_error("Error", f"Error saving plot: {str(e)}")

    def viz_log_message(self, message):
        """Mensaje de log para la pestaña Visualization"""
        self.viz_log_text.AppendText(message + "\n")
        self.SetStatusText(message.strip().splitlines()[-1][:120] if message.strip() else "")
        wx.SafeYield(None, True)

    def log_message(self, message):
        """Mensaje de log para la pestaña principal"""
        self.log_text.AppendText(message + "\n")
        self.SetStatusText(message.strip()[:120])
        wx.SafeYield(None, True)


def main():
    app = wx.App(False)
    frame = PerformanceAnalyzerApp(None)
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()