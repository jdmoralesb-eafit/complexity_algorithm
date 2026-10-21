"""Instala WSL2 + Ubuntu + perf desde Windows, sin abrir la terminal de Linux.

Uso (PowerShell o CMD de Windows, como ADMINISTRADOR la primera vez):
    python setup_windows.py [--distro Ubuntu] [--skip-update]

Si WSL no estaba instalado, el script lo instala y pide reiniciar; despues del
reinicio se vuelve a ejecutar el mismo comando y continua donde quedo.
Los comandos dentro de Linux corren como root (`wsl -u root`), asi que no hace
falta crear usuario ni escribir contrasenas.
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP_LINUX = os.path.join(HERE, "setup.py")


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def wsl(args, **kw):
    """Corre `wsl.exe` y devuelve (codigo, salida). wsl.exe emite UTF-16 en sus
    mensajes propios (no en los de los programas de Linux), por eso se decodifica
    aparte."""
    res = subprocess.run(["wsl"] + args, capture_output=True, **kw)
    raw = res.stdout or b""
    text = raw.decode("utf-16-le", errors="ignore") if b"\x00" in raw else raw.decode("utf-8", errors="ignore")
    return res.returncode, text.replace("\x00", "").strip()


def installed_distros():
    code, out = wsl(["-l", "-q"])
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def step(msg):
    print(f"\n[+] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Instala WSL2 + perf desde Windows.")
    ap.add_argument("--distro", default="Ubuntu")
    ap.add_argument("--skip-update", action="store_true")
    args = ap.parse_args()

    if os.name != "nt":
        sys.exit("Este script es para Windows. Dentro de Linux usa setup.py.")
    if not os.path.exists(SETUP_LINUX):
        sys.exit(f"No encuentro {SETUP_LINUX}")

    # 1) WSL instalado?
    step("Comprobando WSL")
    if not shutil.which("wsl"):
        sys.exit("wsl.exe no existe: se requiere Windows 10 2004+ o Windows 11.")

    distros = installed_distros()
    if not distros:
        if not is_admin():
            sys.exit("Falta instalar WSL. Vuelve a ejecutar este script como ADMINISTRADOR.")
        step(f"Instalando WSL2 y {args.distro} (puede tardar varios minutos)")
        code = subprocess.run(["wsl", "--install", "-d", args.distro, "--no-launch"]).returncode
        if code != 0:
            sys.exit("Fallo `wsl --install`. Revisa que la virtualizacion este activa en la BIOS.")
        distros = installed_distros()
        if not distros:
            print("\nWSL quedo instalado pero hace falta REINICIAR Windows.")
            print("Despues del reinicio vuelve a ejecutar:  python setup_windows.py")
            sys.exit(0)

    distro = args.distro if args.distro in distros else distros[0]
    print(f"Distribucion: {distro}")

    # 2) Inicializar la distro sin pedir usuario: se usa root.
    step("Inicializando la distribucion (usuario root)")
    exe = distro.lower().replace(" ", "").replace("-", "") + ".exe"
    if shutil.which(exe):
        subprocess.run([exe, "install", "--root"])  # si ya estaba lista, no hace nada
    code, _ = wsl(["-d", distro, "-u", "root", "--", "true"])
    if code != 0:
        sys.exit(f"No pude arrancar {distro}. Abrela una vez a mano y reintenta.")

    # 3) Ejecutar setup.py dentro de Linux. wslpath convierte la ruta de Windows.
    step("Compilando perf dentro de WSL (puede tardar 10-30 min)")
    code, linux_path = wsl(["-d", distro, "-u", "root", "--", "wslpath", "-a", SETUP_LINUX.replace("\\", "/")])
    if code != 0 or not linux_path:
        sys.exit("No pude convertir la ruta de setup.py a formato Linux.")
    cmd = ["wsl", "-d", distro, "-u", "root", "--cd", "~", "--",
           "bash", "-c", f"apt-get update -y && apt-get install -y python3 sudo && python3 '{linux_path}'"
           + (" --skip-update" if args.skip_update else "")]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("setup.py fallo dentro de WSL. Log en /root/perf_setup.log (wsl -u root cat ~/perf_setup.log).")

    # 4) Verificar
    step("Verificando")
    code, out = wsl(["-d", distro, "-u", "root", "--", "perf", "--version"])
    print(out if code == 0 else "perf no responde")
    sys.exit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
