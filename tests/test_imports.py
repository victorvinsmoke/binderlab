"""Tests de importación y CLI de binderlab."""
import subprocess
import sys
from binderlab.cli import _COMMANDS, main


def test_dispatcher_conoce_los_4_subcomandos():
    assert set(_COMMANDS) == {"monitor", "corr", "dna", "score"}


def test_cli_help_and_version():
    assert main(["--help"]) == 0
    assert main([]) == 0
    assert main(["--version"]) == 0
    assert main(["-V"]) == 0


def test_import_sin_side_effects():
    """Importar el monitor en un proceso limpio no debe crear carpetas temporales."""
    code = (
        "import sys, glob, os, tempfile;"
        "sys.argv=['x'];"
        "before=set(glob.glob(os.path.join(tempfile.gettempdir(),'bindermonitor_*')));"
        "import binderlab.monitor as m;"
        "assert m.PIPELINE is None, 'PIPELINE no es None a import-time';"
        "assert m.TEMP_DIR is None, 'TEMP_DIR no es None a import-time';"
        "after=set(glob.glob(os.path.join(tempfile.gettempdir(),'bindermonitor_*')));"
        "assert before==after, 'se creo un tempdir al importar'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"Error al importar monitor: {r.stderr.strip()}"
