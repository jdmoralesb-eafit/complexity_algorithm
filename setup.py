#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
import logging
from datetime import datetime

LOGFILE = os.path.expanduser("~/perf_setup.log")

# ==============================
# Logging configuration
# ==============================
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


def run(cmd, desc=None, check=True):
    """Run shell command with error handling and logging."""
    logging.info(f"\n[+] {desc or cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        logging.error(f"[!] Command failed: {cmd}")
        sys.exit(1)


def check_installed(pkg_name):
    """Check if a Debian package is already installed."""
    res = subprocess.run(f"dpkg -s {pkg_name}", shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Automated perf setup for WSL.")
    parser.add_argument("--skip-update", action="store_true",
                        help="Skip apt update/upgrade step.")
    parser.add_argument("--only-build-perf", action="store_true",
                        help="Only build and install perf (skip dependency setup).")
    args = parser.parse_args()

    logging.info("=== Automated Perf Environment Setup ===")
    logging.info(f"Log file: {LOGFILE}")

    start_time = datetime.now()

    try:
        if not args.skip_update and not args.only_build_perf:
            run("sudo apt update && sudo apt upgrade -y", "Updating system packages")

        if not args.only_build_perf:
            base_pkgs = ["python3", "python3-pip", "build-essential", "git"]
            missing = [p for p in base_pkgs if not check_installed(p)]
            if missing:
                run("sudo apt install -y " + " ".join(missing),
                    "Installing Python and base tools")
            else:
                logging.info("[✓] Core dependencies already installed")

            kernel_deps = ["build-essential", "flex", "bison", "libssl-dev", "libelf-dev"]
            missing = [p for p in kernel_deps if not check_installed(p)]
            if missing:
                run("sudo apt install -y " + " ".join(missing),
                    "Installing kernel build dependencies")
            else:
                logging.info("[✓] Kernel build dependencies already installed")

        if not os.path.exists("WSL2-Linux-Kernel"):
            run("git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git",
                "Cloning WSL2 kernel source")
        else:
            logging.info("[✓] Kernel source already cloned")

        os.chdir("WSL2-Linux-Kernel/tools/perf")

        perf_deps = [
            "libtraceevent-dev", "libtracefs-dev", "libunwind-dev",
            "libslang2-dev", "libperl-dev", "liblzma-dev",
            "libnuma-dev", "libcap-dev", "libdw-dev"
        ]
        if not args.only_build_perf:
            missing = [p for p in perf_deps if not check_installed(p)]
            if missing:
                run("sudo apt install -y " + " ".join(missing),
                    "Installing perf build dependencies")
            else:
                logging.info("[✓] Perf dependencies already installed")

        run("make NO_LIBTRACEEVENT=1 NO_LIBTRACEFS=1", "Building perf from source")
        run("sudo cp perf /usr/local/bin/", "Copying perf binary to /usr/local/bin")

        run("perf --version", "Checking perf version")
        run("sudo perf stat -- sleep 0.1", "Testing perf functionality")

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logging.info(f"\n✅ Setup completed successfully in {elapsed:.1f} seconds.")
        logging.info("You can now use `perf` anywhere in your environment.")
        logging.info(f"Detailed log saved at: {LOGFILE}")

    except KeyboardInterrupt:
        logging.warning("\n[!] Installation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n[!] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
