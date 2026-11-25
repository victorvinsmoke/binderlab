"""Self-check minimo. Falla si el empaquetado rompe algo.

Ejecutar: python -m binderlab._selfcheck
Cubre SOLO lo que el empaquetado introduce (diferido de import-time, enrutado
de subcomandos, resolucion de presets). La logica de cálculo intensivo se valida contra datos de diseño en el entorno HPC.
"""
import sys


def test_import_sin_side_effects():
    """Importar el monolito NO debe parsear sys.argv ni crear tempdir.
    Se comprueba en un interprete limpio: si se hiciera en este proceso, otro
    test podria haber poblado ya los globals y el check seria falso positivo."""
    import subprocess
    import sys as _s
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
    r = subprocess.run([_s.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"hazard de import-time: {r.stderr.strip()}"


def test_dispatcher_conoce_los_4():
    from binderlab.cli import _COMMANDS
    assert set(_COMMANDS) == {"monitor", "corr", "dna", "score"}


def test_help_no_revienta():
    from binderlab.cli import main
    assert main(["--help"]) == 0
    assert main([]) == 0


def test_preset_complexa_se_resuelve_por_nombre():
    """-c complexa y -c complexa.json deben resolver al preset del paquete y
    aportar las claves que el monolito usa para pipelines no-RF/MPNN/AF."""
    sys.argv = ["x"]
    import binderlab.monitor as m
    for arg in ("complexa", "complexa.json"):
        s = m.load_pipeline_structure(arg)
        assert s.get("extra_markers"), f"-c {arg}: no cargo extra_markers"
        assert s.get("metric_patterns"), f"-c {arg}: no cargo metric_patterns"


def test_deteccion_experimentos_complexa():
    """Con el preset, un directorio con binder_results_*.csv (y sin carpetas
    RF/MPNN/AF) debe reconocerse como experimento."""
    import tempfile
    import os
    import shutil
    sys.argv = ["x"]
    import binderlab.monitor as m
    base = tempfile.mkdtemp(prefix="selfcheck_exp_")
    try:
        exp = os.path.join(base, "PES1_1_design")
        os.makedirs(exp)
        open(os.path.join(exp, "binder_results_x_0.csv"), "w").close()
        m.PIPELINE = m.load_pipeline_structure("complexa")
        assert len(m.get_experiments(base)) == 1, "no detecto el experimento Complexa"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_base_name_desde_pdb_path():
    """Los CSV de Complexa no traen 'description': base_name se deriva de la
    carpeta padre de pdb_path. Es lo que alimenta el desplegable de hits."""
    import tempfile
    import os
    sys.argv = ["x"]
    import binderlab.monitor as m
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("id_gen,pdb_path\n")
            f.write("0,/w/configs/PES1_2_design/job_0_n_535_id_14/job_0_n_535_id_14.pdb\n")
        df = m.parse_metrics_file(path)
        assert "base_name" in df.columns, "no genero base_name desde pdb_path"
        assert df["base_name"].iloc[0] == "job_0_n_535_id_14", \
            f"base_name incorrecto: {df['base_name'].iloc[0]}"
    finally:
        os.unlink(path)


def test_core_es_puro():
    """Los modulos de core/ NO deben importar Dash: son logica pura y deben
    poder usarse desde la CLI o desde tests sin levantar nada."""
    import pathlib
    raiz = pathlib.Path(__file__).parent / "core"
    for f in raiz.glob("*.py"):
        texto = f.read_text(encoding="utf-8")
        assert "import dash" not in texto, f"{f.name} importa dash"


def test_core_scanner_complexa():
    """La cadena de escaneo funciona con el preset Complexa: detecta el
    experimento, filtra las metricas y deriva base_name desde pdb_path."""
    import tempfile
    import os
    import shutil
    from binderlab.core import scanner as S
    from binderlab.core.config import load_pipeline_structure
    base = tempfile.mkdtemp(prefix="selfcheck_scan_")
    try:
        exp = os.path.join(base, "PES1_1_design")
        os.makedirs(os.path.join(exp, "job_A"))
        with open(os.path.join(exp, "binder_results_x_0.csv"), "w") as f:
            f.write("id_gen,pdb_path\n0,/w/PES1_1_design/job_A/job_A.pdb\n")
        open(os.path.join(exp, "RAW_x_transposed.csv"), "w").close()
        pl = load_pipeline_structure("complexa")
        exps = S.get_experiments(base, pl)
        assert len(exps) == 1, "no detecto el experimento Complexa"
        mets = S.get_metric_files(exps[0], pl)
        assert all("RAW_" not in os.path.basename(m) for m in mets), "no filtro los RAW_"
        df = S.parse_metrics_file(mets[0], pl)
        assert list(df["base_name"]) == ["job_A"], "base_name mal derivado de pdb_path"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_settings_deriva_etapas():
    """Settings deriva las constantes de etapa del preset."""
    from binderlab.core.config import Settings, load_pipeline_structure
    s = Settings(master_dir="/tmp", pipeline=load_pipeline_structure("complexa"))
    assert s.stage_keys == ("complexa",), f"etapas inesperadas: {s.stage_keys}"
    s2 = Settings(master_dir="/tmp", pipeline=load_pipeline_structure(None))
    assert len(s2.stage_keys) == 3, "el esquema por defecto debe tener 3 etapas"


def test_expresiones_peligrosas_rechazadas():
    """El acceso a atributos dunder se rechaza; las expresiones normales no."""
    import pandas as pd
    from binderlab.core import search as S
    df = pd.DataFrame({"pae": [5, 12], "plddt": [90, 70]})
    _, err = S.apply_expression(df, "pae.__class__.__init__.__globals__")
    assert err is not None, "no rechazo el acceso a __globals__"
    mask, err = S.apply_expression(df, "pae < 10")
    assert err is None and list(mask) == [True, False], "rompio una expresion legitima"


def test_coercion_unificada():
    """El helper de coercion existe y lo usan ambos sitios (antes duplicado)."""
    import pandas as pd
    import inspect
    from binderlab.core import search as S
    out = S.coerce_numeric_columns(pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]}))
    assert str(out["a"].dtype) != "object", "no convirtio la columna numerica"
    # Lo que importa no es el nombre del dtype (cambia entre versiones de
    # pandas) sino que la columna de texto conserve sus valores.
    assert list(out["b"]) == ["x", "y"], "destruyo una columna de texto"
    for fn in (S.apply_expression, S.apply_computed_columns):
        assert "coerce_numeric_columns" in inspect.getsource(fn), \
            f"{fn.__name__} no usa el helper unificado"


def test_anotaciones_ciclo_completo():
    """Tags y nota conviven; cambiar tags no borra la nota."""
    import tempfile
    import shutil
    from binderlab.core import annotations as A
    d = tempfile.mkdtemp(prefix="selfcheck_ann_")
    try:
        A.set_design_tags(d, "j1", ["a", "b"])
        A.set_design_note(d, "j1", "nota")
        assert A.get_design_annotations(d, "j1") == (["a", "b"], "nota")
        A.set_design_tags(d, "j1", ["c"])
        assert A.get_design_annotations(d, "j1") == (["c"], "nota"), \
            "cambiar tags no debe tocar la nota"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_trajectory_pdb_sin_name_error():
    """Localizar la trayectoria multimodelo usa `re` y no debe fallar con NameError."""
    import tempfile
    import os
    import shutil
    from binderlab.core import pdb as P
    base = tempfile.mkdtemp(prefix="selfcheck_traj_")
    try:
        rf_dir = os.path.join(base, "RF_output", "traj")
        os.makedirs(rf_dir)
        traj_file = os.path.join(rf_dir, "design_0_pX0_traj.pdb")
        open(traj_file, "w").close()
        pl = {"rfdiffusion_dir": "RF_output", "trajectory_subdir": "traj", "trajectory_suffix": "_pX0_traj"}
        found = P.get_trajectory_pdb(base, "design_0_dldesign_1", pl)
        assert found == traj_file, f"no localizo la trayectoria esperada: {found}"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_scoring_lee_cabecera_real():
    """scoring_fisico ya no impone nombres de columna por posicion."""
    import pathlib
    src = (pathlib.Path(__file__).parent / "scoring_fisico.py").read_text(encoding="utf-8")
    # Ignorar los comentarios: uno de ellos MENCIONA el patron viejo para
    # explicar por que se cambio.
    codigo = "\n".join(l for l in src.split("\n")
                       if not l.strip().startswith("#"))
    assert "names=columnas" not in codigo, "sigue imponiendo columnas por posicion"
    assert "skiprows=1" not in codigo, "sigue saltandose la cabecera real"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} checks OK")


if __name__ == "__main__":
    _run()
