import os
import subprocess
import platform
import re
import shlex
from PIL import Image
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

# --- Nueva interfaz Tkinter ---
class PerformanceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Performance and Complexity Analizer")
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
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.setup_analysis_tab()
        
        # Pestaña 2: Visualización de resultados
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Visualization")
        self.setup_visualization_tab()
        
    def setup_analysis_tab(self):
        # Frame principal de configuración
        main_frame = ttk.Frame(self.analysis_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurar grid weights
        main_frame.columnconfigure(1, weight=1)
        
        # Selección de archivo Python
        ttk.Label(main_frame, text="Python File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(main_frame, textvariable=self.file_path, width=50)
        file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_python_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Selección de función
        ttk.Label(main_frame, text="Function:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.function_var = tk.StringVar()
        function_combo = ttk.Combobox(main_frame, textvariable=self.function_var, state="readonly")
        function_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        function_combo.bind('<<ComboboxSelected>>', self.on_function_selected)
        self.function_combo = function_combo
        
        # Frame para parámetros
        self.param_frame = ttk.LabelFrame(main_frame, text="Function Parameters", padding="10")
        self.param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.param_frame.columnconfigure(1, weight=1)
        
        # Parámetros comunes
        ttk.Label(main_frame, text="Repetitions:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.repetitions_var = tk.IntVar(value=5)
        ttk.Spinbox(main_frame, from_=1, to=20, textvariable=self.repetitions_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Output file:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.output_file_var = tk.StringVar(value="performance.txt")
        ttk.Entry(main_frame, textvariable=self.output_file_var, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_file).grid(row=4, column=2, padx=5, pady=5)
        
        # Botón de ejecución
        self.run_button = ttk.Button(main_frame, text="Run Analysis", command=self.run_analysis, state="disabled")
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Área de log
        ttk.Label(main_frame, text="Execution Log:").grid(row=6, column=0, sticky=tk.W, pady=5)
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
        ttk.Button(control_frame, text="Load Results File", 
                  command=self.load_results_file).pack(side=tk.LEFT, padx=5)
        
        # Botón para generar gráficos
        ttk.Button(control_frame, text="Generate Plots", 
                  command=self.generate_plots).pack(side=tk.LEFT, padx=5)
        
        # Label para mostrar archivo cargado
        self.file_label = ttk.Label(control_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Frame para gráficos
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill='both', expand=True)
        
    def browse_python_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Python File",
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
            messagebox.showerror("Error", f"Could not load the file: {e}")
            
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
            self.function_flags = {}
            row = 0
            
            # Header con el nuevo orden
            headers = ["Parameter", "Value", "", "Is it a function?"]
            for col, header in enumerate(headers):
                ttk.Label(self.param_frame, text=header, font=('Arial', 9, 'bold')).grid(
                    row=row, column=col, padx=2, pady=3)
            row += 1
            
            # Obtener lista de parámetros
            params = list(sig.parameters.items())
            
            for i, (param_name, param) in enumerate(params):
                # Columna 0: Nombre del parámetro (PRIMERO)
                ttk.Label(self.param_frame, text=param_name, font=('Arial', 9)).grid(
                    row=row, column=0, sticky=tk.W, padx=2, pady=2)
                
                # Columna 1: Entrada de valor (SEGUNDO)
                var = tk.StringVar(value=str(param.default) if param.default != param.empty else "")
                entry = ttk.Entry(self.param_frame, textvariable=var, width=30)
                entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
                self.function_args[param_name] = var
                
                # Columna 2: Botón de explorar (TERCERO - solo para algunos)
                if param_name.lower() in ['folder', 'path', 'directory', 'image_path'] or i == 0:
                    browse_btn = ttk.Button(self.param_frame, text="📁", width=3)
                    browse_btn.grid(row=row, column=2, padx=2, pady=2)
                    browse_btn.configure(command=lambda v=var: self.browse_folder(v))
                else:
                    ttk.Label(self.param_frame, text="").grid(row=row, column=2, padx=2, pady=2)
                
                # Columna 3: Checkbox para función (CUARTO - ÚLTIMO)
                is_function_var = tk.BooleanVar(value=False)
                function_check = ttk.Checkbutton(self.param_frame, variable=is_function_var)
                function_check.grid(row=row, column=3, padx=2, pady=2)
                self.function_flags[param_name] = is_function_var
                
                row += 1
            
            # Ajustar pesos de columnas para mejor visualización
            self.param_frame.columnconfigure(1, weight=1)  # La columna de valor se expande
            
            self.run_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve function information: {e}")
            
    def browse_output_file(self, var=None):
        if var is None:
            var = self.output_file_var
        filepath = filedialog.asksaveasfilename(
            title="Save results as",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            var.set(filepath)
            
    def browse_folder(self, var):
        folderpath = filedialog.askdirectory(title="Select folder")
        if folderpath:
            var.set(folderpath)
            
    def run_analysis(self):
        if not verify_dependencies_once():
            messagebox.showerror("Error", "Missing dependencies in WSL. Check the console.")
            return
            
        # Construir llamada a la función
        args_dict = {}
        for param_name, var in self.function_args.items():
            value = var.get()
            
            # Verificar si el usuario marcó este parámetro como función
            is_function = self.function_flags[param_name].get()
            
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
            if self.function_flags[param_name].get():
                func_call_parts.append(f"{param_name}={value}")  # Sin comillas
            else:
                func_call_parts.append(f"{param_name}={repr(value)}")  # Con comillas si es string
        
        original_func_call = f"{self.function_var.get()}("
        original_func_call += ", ".join(func_call_parts)
        original_func_call += ")"
        
        # Mostrar en log para debugging
        self.log_message(f"Constructed call: {original_func_call}")
        
        # Obtener el PRIMER parámetro (asumiendo que es el directorio)
        first_param_name = list(args_dict.keys())[0]
        first_param_value = args_dict[first_param_name]
        
        # Verificar que el primer parámetro es un directorio válido
        if not isinstance(first_param_value, str) or not os.path.isdir(first_param_value):
            messagebox.showerror("Error", f"The first parameter '{first_param_name}' must be a valid directory.")
            return
        
        # Normalizar la ruta
        first_param_value = first_param_value.replace('\\', '/')
        
        # Obtener todas las subcarpetas del directorio
        subfolders = [f for f in os.listdir(first_param_value) 
                    if os.path.isdir(os.path.join(first_param_value, f))]
        
        if not subfolders:
            messagebox.showerror("Error", f"The directory '{first_param_value}' does not contain subfolders.")
            return
        
        # Leer el código del archivo Python una vez
        try:
            with open(self.file_path.get(), 'r', encoding='utf-8') as f:
                module_code = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the Python file: {e}")
            return
        
        self.log_message(f"Starting performance analysis...")
        self.log_message(f"Function: {self.function_var.get()}")
        self.log_message(f"Repetitions: {self.repetitions_var.get()}")
        self.log_message(f"Base directory: {first_param_value}")
        self.log_message(f"Subfolders found: {len(subfolders)}")
        
        all_results = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy']
        
        try:
            # Recorrer cada subcarpeta
            for folder in subfolders:
                folder_path = os.path.join(first_param_value, folder)
                folder_path = folder_path.replace('\\', '/')
                
                self.log_message(f"Processing folder: {folder}")
                
                # Recorrer cada imagen en la subcarpeta
                folder_images = []
                for file in os.listdir(folder_path):
                    file_path_full = os.path.join(folder_path, file)
                    if (os.path.isfile(file_path_full) and 
                        any(file.lower().endswith(ext) for ext in image_extensions)):
                        folder_images.append(file_path_full.replace('\\', '/'))
                
                if not folder_images:
                    self.log_message(f"  No images found in {folder}")
                    continue
                    
                self.log_message(f"  Images found: {len(folder_images)}")
                
                # Procesar cada imagen en la subcarpeta
                folder_results = []
                for image_path in folder_images:
                    self.log_message(f"    Processing: {os.path.basename(image_path)}")
                    
                    # Modificar SOLO el primer parámetro (directorio) por la imagen actual
                    # CONVERTIR la ruta de Windows a WSL para que funcione en Linux
                    image_path_wsl = convert_windows_path_to_wsl(image_path)
                    modified_func_call = original_func_call.replace(
                        f"{first_param_name}={repr(args_dict[first_param_name])}",
                        f"{first_param_name}=np.array(Image.open(r'{image_path_wsl}').convert('L'), dtype=np.float32)"
                    )
                    
                    # Ejecutar para esta imagen
                    image_results = []
                    for rep in range(self.repetitions_var.get()):
                        result = execute_perf_measurement(module_code, modified_func_call, self.file_path.get())
                        image_results.append(result)
                    
                    folder_results.append((os.path.basename(image_path), image_results))
                
                all_results.append((folder, folder_results))
            
            # Guardar resultados
            output_file = self.output_file_var.get()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Performance analysis - {self.function_var.get()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analyzed directory: {first_param_value}\n")
                f.write(f"Original parameters: {args_dict}\n\n")
                
                total_items = 0
                total_errors = 0
                
                for folder_name, folder_data in all_results:
                    f.write(f"=== FOLDER: {folder_name} ===\n")
                    
                    for image_name, image_results in folder_data:
                        f.write(f"--- IMAGE: {image_name} ---\n")
                        
                        for i, result in enumerate(image_results, 1):
                            f.write(f"  Repetition {i}:\n")
                            f.write(result)
                            f.write("\n")
                            
                            # Contar errores
                            if "Error:" in result or "Traceback" in result:
                                total_errors += 1
                            total_items += 1
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                # Resumen estadístico
                if total_items > 0:
                    f.write(f"=== SUMMARY  ===\n")
                    f.write(f"Total runs: {total_items}\n")
                    f.write(f"Errors found: {total_errors}\n")
                    f.write(f"Success rate: {(total_items - total_errors) / total_items * 100:.2f}%\n")
            
            self.log_message(f"Analysis completed. Results saved in: {output_file}")
            messagebox.showinfo("Success", f"Analysis completed. Results in: {output_file}")
            
        except Exception as e:
            self.log_message(f"Error during analysis: {e}")
            messagebox.showerror("Error", f"Error during execution: {e}")
            
    def log_to_console(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_results_file(self):
        filepath = filedialog.askopenfilename(
            title="Select results file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.results_file = filepath
            self.file_label.config(text=f"File: {os.path.basename(filepath)}")
            
            # Detectar tipo de archivo
            file_type = self.detect_file_type(filepath)
            self.log_message(f"File detected as: {file_type}")
        
    def detect_file_type(self, filepath):
        """Detecta si el archivo es de performance o de bytecodes"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
                
            # Buscar patrones característicos
            for line in first_lines:
                if "BYTECODE ANALYSIS" in line:
                    return "bytecodes"
                if "Performance counter stats" in line:
                    return "performance"
                if "cycles" in line and "instructions" in line:
                    return "performance"
                if "Function:" in line and "Executed lines" in line:
                    return "bytecodes"
                    
            # Si no encuentra patrones claros, intentar por estructura
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "Executed lines" in content and "Estimated bytecodes" in content:
                return "bytecodes"
            if "cycles" in content and "cache-misses" in content:
                return "performance"
                
            return "desconocido"
            
        except Exception as e:
            self.log_message(f"Error detecting file type: {e}")
            return "unknown"

    def generate_plots(self):
        """Generar gráficos a partir del archivo de resultados"""
        if not hasattr(self, 'results_file'):
            messagebox.showerror("Error", "You must load a results file first")
            return
            
        try:
            # Detectar tipo de archivo
            file_type = self.detect_file_type(self.results_file)
            
            # Limpiar área de gráficos
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            # Crear figura de matplotlib según el tipo
            if file_type == "performance":
                fig = self.create_performance_plots(self.results_file)
            else:
                messagebox.showerror("Error", "Unrecognized file type")
                return
                
            if fig is None:
                messagebox.showerror("Error", "Could not extract data from the results file")
                return
                
            # Mostrar en la interfaz
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Agregar toolbar de navegación
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
            self.log_message(f"{file_type} plots generated successfully")
            
        except Exception as e:
            error_msg = f"Error generating plots: {str(e)}"
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
    def log_message(self, message):
        """Mensaje de log para la pestaña principal"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def create_performance_plots(self, txt_file):
        """Crear gráficos de performance a partir de archivo TXT - Versión mejorada"""
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

        # Expresiones regulares
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
                    val_str = match.group(1).replace(',', '')
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
            
            if "=== FOLDER:" in line:
                folder_match = re.search(r'=== FOLDER:\s+([^\n]+)', line)
                if folder_match:
                    current_folder = folder_match.group(1)
                    current_image = None
                    in_perf_block = False
            
            elif "--- IMAGE:" in line:
                image_match = re.search(r'--- IMAGE:\s+([^\n]+)', line)
                if image_match and current_folder:
                    current_image = image_match.group(1)
                    in_perf_block = False
            
            elif "Performance counter stats for" in line:
                in_perf_block = True
                current_rep_data = {}
            
            elif in_perf_block and "seconds time elapsed" in line:
                metrics = parse_perf_line(line)
                if 'time' in metrics and current_folder:
                    data_time[current_folder].append(metrics['time'])
                in_perf_block = False
            
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

        # Limpiar los nombres de las carpetas
        def clean_folder_name(name):
            if name.endswith(' ==='):
                return name[:-4]
            return name
        
        cleaned_data_cycles = {clean_folder_name(k): v for k, v in data_cycles.items()}
        cleaned_data_instructions = {clean_folder_name(k): v for k, v in data_instructions.items()}
        cleaned_data_cache_refs = {clean_folder_name(k): v for k, v in data_cache_refs.items()}
        cleaned_data_cache_misses = {clean_folder_name(k): v for k, v in data_cache_misses.items()}
        cleaned_data_time = {clean_folder_name(k): v for k, v in data_time.items()}
        
        # Usar los datos limpios
        cycles_means, cycles_stds = calculate_stats(cleaned_data_cycles)
        instructions_means, instructions_stds = calculate_stats(cleaned_data_instructions)
        time_means, time_stds = calculate_stats(cleaned_data_time)

        # Determinar resoluciones disponibles
        available_resolutions = [r for r in resol_order if r in time_means]
        
        if not available_resolutions:
            available_resolutions = resol_order

        # Preparar datos para ajuste de complejidad
        x_fit = np.array([resol_to_pixels[r] for r in available_resolutions])
        y_time_fit = np.array([time_means[r] for r in available_resolutions])
        y_instructions_fit = np.array([instructions_means[r] for r in available_resolutions])

        # Funciones de ajuste básicas
        def f_linear(N, a, b): 
            return a*N + b

        def f_quadratic(N, a, b): 
            return a*N**2 + b

        def f_nlogn(N, a, b): 
            return a*N*np.log(N) + b

        def f_cubic(N, a, b):
            return a*N**3 + b

        def f_quartic(N, a, b):
            return a*N**4 + b

        # FUNCIÓN POLINÓMICA COMPLETA - a*N⁴ + b*N³ + c*N² + d*N + e
        def f_polynomial_complete(N, a, b, c, d, e):
            return a*N**4 + b*N**3 + c*N**2 + d*N + e

        # Función para generar ecuaciones legibles
        def format_polynomial_equation(coeffs):
            """Formatea ecuación polinómica completa"""
            a, b, c, d, e = coeffs
            parts = []
            
            if abs(a) > 1e-10:
                parts.append(f"({a:.2e})·N⁴")
            if abs(b) > 1e-10:
                parts.append(f"({b:.2e})·N³")
            if abs(c) > 1e-10:
                parts.append(f"({c:.2e})·N²")
            if abs(d) > 1e-10:
                parts.append(f"({d:.2e})·N")
            if abs(e) > 1e-10:
                parts.append(f"({e:.2e})")
            
            if not parts:
                return "y = 0"
            
            return "y = " + " + ".join(parts)

        def complete_polynomial_fit(x, y, threshold=0.009):
            """
            Ajuste polinómico completo: a*N⁴ + b*N³ + c*N² + d*N + e
            con eliminación jerárquica de términos no significativos
            """
            best_r2 = -np.inf
            best_fit = None
            best_params = None
            best_complexity = "Polynomial"
            
            # Probar diferentes combinaciones eliminando términos de mayor a menor orden
            term_combinations = [
                [True, True, True, True, True],   # Todos los términos: N⁴, N³, N², N, constante
                [True, True, True, True, False],  # Sin constante
                [True, True, True, False, True],  # Sin término N
                [True, True, False, True, True],  # Sin término N²
                [True, False, True, True, True],  # Sin término N³
                [False, True, True, True, True],  # Sin término N⁴
                [True, True, True, False, False], # Solo N⁴, N³, N²
                [True, True, False, False, True], # Solo N⁴, N³, constante
                [True, False, False, True, True], # Solo N⁴, N, constante
                [False, True, True, False, True], # Solo N³, N², constante
                [False, False, True, True, True], # Solo N², N, constante
            ]
            
            for terms in term_combinations:
                try:
                    # Definir función parcial según términos activos
                    def partial_poly(N, *params):
                        result = np.zeros_like(N, dtype=float)
                        param_idx = 0
                        if terms[0]:  # N⁴
                            result += params[param_idx] * N**4
                            param_idx += 1
                        if terms[1]:  # N³
                            result += params[param_idx] * N**3
                            param_idx += 1
                        if terms[2]:  # N²
                            result += params[param_idx] * N**2
                            param_idx += 1
                        if terms[3]:  # N
                            result += params[param_idx] * N
                            param_idx += 1
                        if terms[4]:  # constante
                            result += params[param_idx]
                        return result
                    
                    n_params = sum(terms)
                    if n_params == 0 or n_params >= len(x):
                        continue
                    
                    # Valores iniciales
                    p0 = [1.0] * n_params
                    
                    # Ajuste
                    popt, pcov = curve_fit(partial_poly, x, y, p0=p0, maxfev=5000)
                    
                    # Verificar si los coeficientes son significativos
                    significant = True
                    param_idx = 0
                    
                    if terms[0] and param_idx < len(popt) and abs(popt[param_idx]) < threshold:
                        significant = False
                    param_idx += 1 if terms[0] else 0
                    
                    if terms[1] and param_idx < len(popt) and abs(popt[param_idx]) < threshold:
                        significant = False
                    param_idx += 1 if terms[1] else 0
                    
                    if terms[2] and param_idx < len(popt) and abs(popt[param_idx]) < threshold:
                        significant = False
                    
                    if not significant:
                        continue
                    
                    # Calcular R²
                    y_pred = partial_poly(x, *popt)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Si es mejor que el anterior, actualizar
                    if r_squared > best_r2:
                        best_r2 = r_squared
                        best_fit = partial_poly
                        best_params = popt
                        
                except Exception:
                    continue
            
            return best_complexity, best_r2, best_fit, best_params, terms if best_fit is not None else None

        # Definir las funciones de ajuste
        fit_functions = {
            'O(N)': f_linear,
            'O(N²)': f_quadratic,
            'O(N log N)': f_nlogn,
            'O(N³)': f_cubic,
            'O(N⁴)': f_quartic,
            'Polynomial': f_polynomial_complete
        }

        # Función para calcular el mejor ajuste
        def find_best_fit(x, y, metric_name):
            """Encuentra el mejor ajuste entre diferentes funciones"""
            fits = {}
            r_squared_values = {}
            equations = {}
            
            # Probar ajuste polinómico completo
            poly_complexity, poly_r2, poly_func, poly_params, poly_terms = complete_polynomial_fit(x, y)
            
            if poly_func is not None:
                fits['Polynomial'] = (poly_params, None, poly_func(x, *poly_params))
                r_squared_values['Polynomial'] = poly_r2
                
                # Reconstruir ecuación completa
                full_params = [0, 0, 0, 0, 0]  # a, b, c, d, e
                param_idx = 0
                if poly_terms[0]:
                    full_params[0] = poly_params[param_idx]
                    param_idx += 1
                if poly_terms[1]:
                    full_params[1] = poly_params[param_idx]
                    param_idx += 1
                if poly_terms[2]:
                    full_params[2] = poly_params[param_idx]
                    param_idx += 1
                if poly_terms[3]:
                    full_params[3] = poly_params[param_idx]
                    param_idx += 1
                if poly_terms[4]:
                    full_params[4] = poly_params[param_idx]
                
                equations['Polynomial'] = format_polynomial_equation(full_params)
            
            # Probar funciones simples
            for name, func in fit_functions.items():
                if name == 'Polynomial':  # Ya probado
                    continue
                try:
                    if len(x) >= 2:
                        popt, pcov = curve_fit(func, x, y, maxfev=5000)
                        y_pred = func(x, *popt)
                        
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        fits[name] = (popt, pcov, y_pred)
                        r_squared_values[name] = r_squared
                        
                        # Generar ecuación
                        if name == 'O(N log N)':
                            equations[name] = f"y = ({popt[0]:.2e})·N·log(N) + {popt[1]:.2e}"
                        else:
                            order = 1 if name == 'O(N)' else int(name.split('O(N')[1].split(')')[0]) if 'O(N²)' not in name else 2
                            if order == 1:
                                equations[name] = f"y = ({popt[0]:.2e})·N + {popt[1]:.2e}"
                            else:
                                equations[name] = f"y = ({popt[0]:.2e})·N^{order} + {popt[1]:.2e}"
                except:
                    continue
            
            # Encontrar el mejor ajuste
            best_fit = None
            best_r2 = -1
            for name, r2 in r_squared_values.items():
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = name
            
            return best_fit, best_r2, fits, r_squared_values, equations

        # Encontrar mejores ajustes
        best_fit_time, best_r2_time, fits_time, r2_values_time, equations_time = find_best_fit(x_fit, y_time_fit, "Time")
        best_fit_instructions, best_r2_instructions, fits_instructions, r2_values_instructions, equations_instructions = find_best_fit(x_fit, y_instructions_fit, "Instructions")

        # Crear gráficos
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # Colores para los ajustes
        colors = {
            'O(N)': 'blue', 'O(N²)': 'green', 'O(N log N)': 'orange',
            'O(N³)': 'red', 'O(N⁴)': 'brown', 'Polynomial': 'magenta'
        }

        # 1. Ajuste de complejidad computacional (instrucciones)
        if len(x_fit) > 0:
            # Graficar puntos reales
            for i, (resol, x_val, y_val) in enumerate(zip(available_resolutions, x_fit, y_instructions_fit)):
                axes[0].scatter(x_val, y_val, color='purple', s=50, alpha=0.7, label='Data points' if i == 0 else "")

            xx = np.linspace(min(x_fit), max(x_fit), 500)
            
            # Graficar solo el MEJOR ajuste
            if best_fit_instructions and best_fit_instructions in fits_instructions:
                if best_fit_instructions == 'Polynomial':
                    # Para polinomio completo
                    poly_complexity, _, poly_func, poly_params, _ = complete_polynomial_fit(x_fit, y_instructions_fit)
                    y_best_fit = poly_func(xx, *poly_params)
                else:
                    popt = fits_instructions[best_fit_instructions][0]
                    y_best_fit = fit_functions[best_fit_instructions](xx, *popt)
                
                label = f'Best Fit: {best_fit_instructions} (R²={best_r2_instructions:.3f})'
                color = colors.get(best_fit_instructions, 'red')
                
                axes[0].plot(xx, y_best_fit, '-', color=color, label=label, linewidth=3)

            # Añadir ecuación
            if best_fit_instructions in equations_instructions:
                eq_text = equations_instructions[best_fit_instructions]
                axes[0].text(0.02, 0.98, f'Equation: {eq_text}', 
                            transform=axes[0].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            fontsize=9)

            axes[0].set_xlabel('Number of pixels (N)')
            axes[0].set_ylabel('Instructions')
            axes[0].set_title(f'Computational complexity fit\n(Best: {best_fit_instructions}, R²={best_r2_instructions:.3f})')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Ajuste de complejidad temporal (tiempo)
        if len(x_fit) > 0:
            # Graficar puntos reales
            for i, (resol, x_val, y_val) in enumerate(zip(available_resolutions, x_fit, y_time_fit)):
                axes[1].scatter(x_val, y_val, color='red', s=50, alpha=0.7, label='Data points' if i == 0 else "")
                
            xx = np.linspace(min(x_fit), max(x_fit), 500)
            
            # Graficar solo el MEJOR ajuste
            if best_fit_time and best_fit_time in fits_time:
                if best_fit_time == 'Polynomial':
                    # Para polinomio completo
                    poly_complexity, _, poly_func, poly_params, _ = complete_polynomial_fit(x_fit, y_time_fit)
                    y_best_fit = poly_func(xx, *poly_params)
                else:
                    popt = fits_time[best_fit_time][0]
                    y_best_fit = fit_functions[best_fit_time](xx, *popt)
                
                label = f'Best Fit: {best_fit_time} (R²={best_r2_time:.3f})'
                color = colors.get(best_fit_time, 'red')
                
                axes[1].plot(xx, y_best_fit, '-', color=color, label=label, linewidth=3)

            # Añadir ecuación
            if best_fit_time in equations_time:
                eq_text = equations_time[best_fit_time]
                axes[1].text(0.02, 0.98, f'Equation: {eq_text}', 
                            transform=axes[1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            fontsize=9)

            axes[1].set_xlabel('Number of pixels (N)')
            axes[1].set_ylabel('Time (s)')
            axes[1].set_title(f'Time complexity fit\n(Best: {best_fit_time}, R²={best_r2_time:.3f})')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

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