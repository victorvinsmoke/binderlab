"""BinderMonitor — Dashboard de campañas de diseño de binders proteicos

Monitor web para pipelines de diseño de novo (RFdiffusion, ProteinMPNN,
AlphaFold2, Rosetta). Proyecta progreso de ejecución en HPC, métricas
moleculares, anotaciones manuales y visualización 3D interactiva en tiempo real.

Uso:
    binderlab monitor -d /ruta/al/master_dir
    binderlab monitor -d /ruta/al/master_dir -c complexa
    binderlab monitor -d /ruta -p 8060 --host 0.0.0.0
"""

import os
import re
import glob
import json
import sys
import argparse
import tempfile
import atexit
import shutil
from datetime import datetime
import signal

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    import functools
    print = functools.partial(print, flush=True)

import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import (
    dcc, html, Input, Output, State, callback,
    ALL, ctx, dash_table, no_update,
)
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

import dash_molstar as dashmolstar
from dash_molstar.utils import molstar_helper

from binderlab.core.config import (               # noqa: E402
    load_pipeline_structure,
    build_stages,
)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="BinderMonitor — Dashboard de campañas de diseño de binders.")
    parser.add_argument("--port", "-p", default=8050, type=int,
                        help="Puerto del servidor web (default: 8050)")
    parser.add_argument("--host", "-host", default="127.0.0.1",
                        help="Host de la app (default: 127.0.0.1; usa 0.0.0.0 para acceso de red)")
    parser.add_argument("--dir", "-d", default=os.getcwd(),
                        help="Directorio raíz inicial. Puede cambiarse desde la UI.")
    parser.add_argument("--config", "-c", default=None,
                        help="Archivo JSON opcional con la estructura del pipeline o nombre de preset.")
    return parser


MASTER_DIR_INIT = None
PIPELINE = None
HOST_LABEL = "?"
PORT_LABEL = "?"


# =============================================================================
# 3. UTILIDADES Y CACHÉS DE BACKEND
# =============================================================================

# Directorio temporal donde se escriben los modelos individuales extraídos
# de los PDBs multimodelo de RFdiffusion. Los nombres de archivo son
# DETERMINISTAS (ver extract_pdb_model_streaming) — no se acumulan archivos:
# si pides el modelo 5 dos veces, sobrescribe el mismo archivo.
# Empaquetado: None a import-time; main() lo crea. mkdtemp toca disco y no
# debe ejecutarse solo por importar el modulo.
TEMP_DIR = None


def _cleanup_temp_dir():
    """Borra el directorio temporal al cerrar la app limpiamente.
    OJO: este hook NO se ejecuta con `kill -9`. Por eso además limpiamos
    de forma proactiva los frames antiguos al cambiar de diseño en
    cleanup_old_traj_files()."""
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


atexit.register(_cleanup_temp_dir)


def cleanup_old_traj_files(keep_basename=None):
    """Borra los frames temporales que ya no se están usando.
    Se llama cuando el usuario cambia de diseño en el visor: así no se
    acumulan archivos en /tmp aunque la app se cierre con kill -9 después.

    Si keep_basename está dado, conserva los frames de ese diseño concreto."""
    if not TEMP_DIR:
        return  # sin tempdir no hay nada que limpiar. Sin esto,
                # os.listdir(None) listaria el CWD y os.path.join(None,..) petaria.
    try:
        for f in os.listdir(TEMP_DIR):
            # Usamos "__m" como separador determinista para no confundir
            # design_1 con design_10, design_100, etc. Sin el separador,
            # cualquier diseño que empiece igual se quedaría en /tmp.
            if keep_basename and f.startswith(keep_basename + "__m"):
                continue
            if f.endswith('.pdb'):
                try:
                    os.remove(os.path.join(TEMP_DIR, f))
                except OSError:
                    pass
    except OSError:
        pass


# -- Caché del estado HPC ----------------------------------------------------
# Llamar a `squeue` cada 10 segundos es agresivo y bloquea el thread de Dash
# si el demonio SLURM está saturado. Cacheamos el resultado durante 30s.


# --- Escaneo del filesystem --------------------------------------------------
# Logica en core/scanner.py, que recibe la configuracion por argumento. Aqui
# quedan adaptadores con la firma antigua para que los callbacks existentes
# sigan funcionando sin tocarlos (patron strangler).
from binderlab.core import scanner as _core_scanner   # noqa: E402


def get_experiments(base_dir):
    return _core_scanner.get_experiments(base_dir, PIPELINE)


def get_metric_files(exp_dir):
    return _core_scanner.get_metric_files(exp_dir, PIPELINE)


def get_base_name(filename):
    return _core_scanner.get_base_name(filename, PIPELINE)


def count_funnel_stage(exp_dir, stage_folder, pdb_glob=None):
    return _core_scanner.count_funnel_stage(exp_dir, stage_folder, PIPELINE, pdb_glob)


def parse_metrics_file(metric_file):
    return _core_scanner.parse_metrics_file(metric_file, PIPELINE)


# --- Anotaciones -------------------------------------------------------------
# Movidas a core/annotations.py (modulo puro). Se reexportan para que los
# callbacks existentes las sigan encontrando aqui sin cambios.
from binderlab.core.annotations import (          # noqa: E402
    DEFAULT_TAGS, get_design_annotations, set_design_tags, set_design_note,
    get_all_annotations,
)


# --- PDBs --------------------------------------------------------------------
# La logica vive en core/pdb.py, que recibe la configuracion por argumento.
# Aqui quedan adaptadores con la firma antigua para que los ~37 callbacks
# sigan funcionando sin tocarlos (patron strangler: se migran mas adelante).
from binderlab.core import pdb as _core_pdb           # noqa: E402
from binderlab.core.pdb import count_pdb_models       # noqa: E402  (no usa config)


def resolve_structure_pdb(exp_dir, hit_basename):
    return _core_pdb.resolve_structure_pdb(exp_dir, hit_basename, PIPELINE)


def get_trajectory_pdb(exp_dir, base_name):
    return _core_pdb.get_trajectory_pdb(exp_dir, base_name, PIPELINE)


def logical_to_physical_model(logical_idx, n_models):
    return _core_pdb.logical_to_physical_model(logical_idx, n_models, PIPELINE)


def extract_pdb_model_streaming(pdb_path, model_idx):
    return _core_pdb.extract_pdb_model_streaming(pdb_path, model_idx, TEMP_DIR)


def find_experiment_logs(exp_dir, max_results=12):
    """Localiza archivos de log de forma VERSÁTIL, pensada para servir a
    laboratorios con pipelines distintos.

    Estrategia (de más específico a más general):
      1. Busca recursivamente dentro del experimento todos los patrones
         de `PIPELINE['log_patterns']` (*.log, *.out, *.err...). Esto
         captura el run_inference.log de RFdiffusion aunque esté enterrado
         en outputs/FECHA/HORA/.
      2. Sube por los directorios padre (hasta `log_search_levels`),
         buscando solo en cada nivel (no recursivo) para capturar
         wrappers (RFrun_multiple_*.log) y los .out/.err de SLURM que
         viven en la carpeta del múltiple o en la raíz de campañas.

    Devuelve una lista de rutas ORDENADA por mtime descendente (más
    reciente primero), deduplicada, recortada a `max_results`. Así el
    usuario puede elegir cuál ver en lugar de adivinar el monitor.
    """
    if not exp_dir or not os.path.isdir(exp_dir):
        return []
    patterns = PIPELINE.get('log_patterns', ['*.log', '*.out', '*.err'])
    levels = int(PIPELINE.get('log_search_levels', 2))

    found = set()

    # 1. Dentro del experimento, recursivo
    for pat in patterns:
        found.update(glob.glob(os.path.join(exp_dir, '**', pat),
                               recursive=True))

    # 2. Hacia arriba, nivel a nivel (no recursivo para no escanear
    #    campañas enteras de otros proyectos)
    cur = exp_dir
    exp_token = os.path.basename(exp_dir).split('.')[0]
    for _ in range(levels):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        for pat in patterns:
            for p in glob.glob(os.path.join(parent, pat)):
                # En niveles superiores (donde puede haber logs de OTRAS
                # campañas) priorizamos los que mencionan el experimento,
                # pero los demás también se incluyen como fallback.
                found.add(p)
        cur = parent

    files = [f for f in found if os.path.isfile(f)]

    # Orden: primero por relevancia (nombre contiene el token del
    # experimento), luego por mtime descendente.
    def _key(p):
        rel = 0 if exp_token and exp_token in os.path.basename(p) else 1
        try:
            mt = os.path.getmtime(p)
        except OSError:
            mt = 0
        return (rel, -mt)

    files.sort(key=_key)
    return files[:max_results]


# ----------------------------------------------------------------------------
#  Parseo del estado del flujo de trabajo (tarjetas de etapa)
# ----------------------------------------------------------------------------
# Sustituye al volcado de texto crudo del .log/.out: lee el log del
# orquestador y lo traduce a un estado estructurado por etapa
# (RFdiffusion → ProteinMPNN → AlphaFold2), cruzando lo que dice el log con
# el conteo real de PDBs en disco. La salida la consume la UI para pintar una
# tarjeta por etapa con estado, barra de progreso, conteos y última línea.

# etapas derivadas de PIPELINE['stages'] (N-etapas). Default = RF/MPNN/AF.
# Empaquetado: se calculan en _init_stages() (llamado por main()) porque leen
# PIPELINE, que a import-time todavia es None.
_STAGES      = []
STAGE_KEYS   = ()
STAGE_LABELS = {}
STAGE_DIRS   = {}
STAGE_GLOBS  = {}
STAGE_COLORS = {}


def _init_stages():
    """Rellena las constantes de etapa desde PIPELINE.

    La logica vive en core/config.build_stages(); aqui solo se copian los
    resultados a los globals de modulo, que siguen existiendo como alias de
    compatibilidad mientras los callbacks no migren a recibir Settings.
    """
    global _STAGES, STAGE_KEYS, STAGE_LABELS, STAGE_DIRS, STAGE_GLOBS, STAGE_COLORS
    (_STAGES, STAGE_KEYS, STAGE_LABELS,
     STAGE_DIRS, STAGE_GLOBS, STAGE_COLORS) = build_stages(PIPELINE)


# Estados posibles de una etapa.
ST_PENDING = "pending"   # aún no ha empezado
ST_RUNNING = "running"   # etapa activa ahora mismo
ST_DONE    = "done"      # completada


def _tail_lines(path, max_bytes=2_000_000):
    """Lee las últimas ~max_bytes de un fichero y las devuelve como lista de
    líneas. Evita cargar logs de cientos de miles de líneas en memoria; para
    detectar el estado actual basta con la cola del archivo."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # descartar línea parcial
            data = f.read()
        return data.decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []


def parse_pipeline_state(log_path, exp_dir):
    """Traduce el log del orquestador + el disco a un estado por etapa.

    Devuelve un dict:
      {
        "experiment": (actual, total) | None,   # "Experimento 2 de 3"
        "global_done": bool,                     # campaña finalizada
        "active": "rfdiffusion"|... | None,      # etapa en curso
        "last_line": str,                        # última línea relevante
        "stages": {
            "rfdiffusion": {
                "status": ST_PENDING|ST_RUNNING|ST_DONE,
                "processed": int,    # unidades vistas en el log
                "total": int|None,   # total estimado (None = desconocido)
                "passed": int, "discarded": int, "pdb_total": int,
                "detail": str,       # texto corto para la tarjeta
            }, ...
        }
      }

    Estrategia: el log manda para "qué etapa está activa" (marcas [N/M]); el
    disco manda para "cuántos PDBs pasaron/se descartaron". Si no hay log se
    cae con elegancia a solo-disco (estado derivado de los conteos)."""
    markers   = PIPELINE.get('log_stage_markers', {})
    prog_pats = PIPELINE.get('log_progress_markers', {})
    exp_pat   = PIPELINE.get('log_experiment_marker')
    done_pat  = PIPELINE.get('log_done_marker')

    re_stage = {k: re.compile(v) for k, v in markers.items() if v}
    re_prog  = {k: re.compile(v) for k, v in prog_pats.items() if v}
    re_exp   = re.compile(exp_pat) if exp_pat else None
    re_done  = re.compile(done_pat) if done_pat else None
    re_ts     = PIPELINE.get('log_rf_timestep_marker')
    re_ts     = re.compile(re_ts) if re_ts else None
    time_pats = PIPELINE.get('log_unit_time_markers', {})
    re_time   = {k: re.compile(v) for k, v in time_pats.items() if v}
    stamp_pat = PIPELINE.get('log_timestamp_marker')
    re_stamp  = re.compile(stamp_pat) if stamp_pat else None
    rf_t_total = PIPELINE.get('rf_timesteps_total', 50)

    state = {
        "experiment": None,
        "global_done": False,
        "active": None,
        "postprocess": False,
        "last_line": "",
        "stages": {k: {"status": ST_PENDING, "processed": 0, "total": None,
                       "passed": 0, "discarded": 0, "pdb_total": 0,
                       "detail": "", "rate_s": None, "elapsed_s": None,
                       "eta_s": None, "ts_cur": None, "ts_total": rf_t_total}
                   for k in STAGE_KEYS},
    }

    # --- Conteo de PDBs en disco (fuente de verdad de passed/discarded) ----
    for k in STAGE_KEYS:
        t, p, d = count_funnel_stage(exp_dir, STAGE_DIRS[k], STAGE_GLOBS[k])
        state["stages"][k].update(pdb_total=t, passed=p, discarded=d)

    # el resto del parser es lógica de log RF (marcas [N/M], timesteps
    # de difusión, ritmos por etapa) y accede a 'rfdiffusion'/'proteinmpnn'/
    # 'alphafold' por nombre fijo. Para pipelines no-RF (Complexa, BindCraft...)
    # no hay ese log: derivamos el estado solo del disco y salimos. Esto evita
    # el KeyError y es correcto para lotes terminados.
    if "rfdiffusion" not in state["stages"]:
        for k in STAGE_KEYS:
            s = state["stages"][k]
            s["status"] = ST_DONE if s["pdb_total"] > 0 else ST_PENDING
            s["total"] = s["pdb_total"] or None
            s["processed"] = s["pdb_total"]
            if s["pdb_total"]:
                s["detail"] = f"{s['pdb_total']} diseños"
        state["global_done"] = any(
            state["stages"][k]["pdb_total"] > 0 for k in STAGE_KEYS)
        return state

    # --- Recorrido del log -------------------------------------------------
    # Streaming línea a línea (no carga todo en memoria). Un prefiltro de
    # subcadenas descarta la inmensa mayoría de líneas (Timestep, dumps de
    # secuencia, config) antes de tocar las regex. Al encontrar un nuevo
    # "INICIANDO EXPERIMENTO" se reinician los contadores: en una campaña
    # múltiple sólo cuenta el experimento en curso (el último del log).
    PREFILTER = ('RFdiffusion', 'ProteinMPNN', 'AlphaFold2', 'Rosetta',
                 'estad', 'termodin', 'Making design', 'generated',
                 'finished AF2', 'INICIANDO', 'FINALIZADO')

    active = None            # última etapa marcada con [N/M]
    postprocess_seen = False
    last_meaningful = ""
    exp_cur = exp_tot = None
    counts = {k: 0 for k in STAGE_KEYS}
    rf_last_ts = None
    # --- Tiempos ---
    # Duraciones por unidad (segundos) que cada programa reporta. Para RF se
    # derivan de timestamps consecutivos de "Making design".
    durations = {k: [] for k in STAGE_KEYS}
    rf_design_stamps = []    # datetimes de cada "Making design" (para ritmo)
    stage_first_stamp = {}   # primer timestamp visto tras la marca de etapa
    stage_last_stamp = {}    # último timestamp visto en la etapa activa

    def _parse_stamp(ln):
        if not re_stamp:
            return None
        m = re_stamp.search(ln)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def _reset_segment():
        nonlocal active, postprocess_seen, last_meaningful
        nonlocal counts, rf_last_ts, durations, rf_design_stamps
        nonlocal stage_first_stamp, stage_last_stamp
        active = None
        postprocess_seen = False
        last_meaningful = ""
        counts = {k: 0 for k in STAGE_KEYS}
        rf_last_ts = None
        durations = {k: [] for k in STAGE_KEYS}
        rf_design_stamps = []
        stage_first_stamp = {}
        stage_last_stamp = {}

    def _process(ln):
        nonlocal active, postprocess_seen, last_meaningful, rf_last_ts
        stamp = _parse_stamp(ln)
        if active and stamp:
            stage_first_stamp.setdefault(active, stamp)
            stage_last_stamp[active] = stamp
        # ¿finalización global?
        if re_done and re_done.search(ln):
            state["global_done"] = True
        # ¿marca de etapa?
        for k, rx in re_stage.items():
            if k == "postprocess":
                if rx.search(ln):
                    postprocess_seen = True
                    last_meaningful = ln.strip()
                    return
                continue
            if rx.search(ln):
                active = k
                last_meaningful = ln.strip()
                if stamp:
                    stage_first_stamp.setdefault(k, stamp)
                    stage_last_stamp[k] = stamp
                return
        # ¿línea de progreso de algún programa?
        for k, rx in re_prog.items():
            m = rx.search(ln)
            if m:
                counts[k] += 1
                last_meaningful = ln.strip()
                if k == "rfdiffusion":
                    if stamp:
                        rf_design_stamps.append(stamp)
                # La propia línea de progreso de MPNN/AF lleva el tiempo
                # ("... in X seconds"): extraerlo aquí mismo.
                rt = re_time.get(k)
                if rt:
                    mt = rt.search(ln)
                    if mt:
                        try:
                            durations[k].append(float(mt.group(1)))
                        except (TypeError, ValueError):
                            pass
                return
        # Línea de tiempo-por-unidad suelta (por si el patrón de tiempo no
        # coincide con el de progreso en algún pipeline).
        for k, rx in re_time.items():
            mt = rx.search(ln)
            if mt:
                try:
                    durations[k].append(float(mt.group(1)))
                except (TypeError, ValueError):
                    pass
                return
        if re_ts and 'Timestep' in ln:
            mt = re_ts.search(ln)
            if mt:
                rf_last_ts = int(mt.group(1))

    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, 'r', errors='replace') as fh:
                for ln in fh:
                    # Marca de experimento: reinicia el segmento.
                    if re_exp and 'INICIANDO' in ln:
                        m = re_exp.search(ln)
                        if m:
                            exp_cur, exp_tot = int(m.group(1)), int(m.group(2))
                            _reset_segment()
                            continue
                    if 'Timestep' in ln:
                        _process(ln)
                        continue
                    if any(s in ln for s in PREFILTER):
                        _process(ln)
        except OSError:
            pass

    # --- Síntesis del estado por etapa -------------------------------------
    # Si ya estamos en post-proceso (o la campaña terminó), NINGUNA de las 3
    # etapas sigue en curso: anular `active` para que la cabecera no muestre
    # "En curso: AlphaFold2" cuando AF ya acabó y se está corriendo Rosetta.
    if postprocess_seen or state["global_done"]:
        active = None
    state["experiment"] = (exp_cur, exp_tot) if exp_cur else None
    state["active"] = active
    state["postprocess"] = postprocess_seen and not state["global_done"]
    state["last_line"] = last_meaningful

    for k in STAGE_KEYS:
        counts_k = counts[k]
        state["stages"][k]["processed"] = counts_k

    # Total estimado de RF: NO se estima de max(design_idx)+1 porque durante
    # un run en vivo siempre es igual al procesado actual (inútil). Se deja
    # None = desconocido hasta que la etapa acabe. Para MPNN/AF, el total
    # sólo tiene sentido cuando la etapa previa ya terminó (pasaron el filtro).
    #
    # Si la etapa previa está DONE, su `passed` ya refleja el filtrado real.
    # Si está RUNNING, el conteo de disco aún no ha sido filtrado → None.

    order = list(STAGE_KEYS)
    if state["global_done"]:
        for k in order:
            state["stages"][k]["status"] = ST_DONE
    else:
        # Etapa activa = la última marcada; las anteriores DONE, posteriores
        # PENDING. Si vimos post-proceso, las 3 están DONE.
        if postprocess_seen and active is None:
            for k in order:
                state["stages"][k]["status"] = ST_DONE
        elif active is not None:
            ai = order.index(active)
            for i, k in enumerate(order):
                if i < ai or postprocess_seen:
                    state["stages"][k]["status"] = ST_DONE
                elif i == ai and not postprocess_seen:
                    state["stages"][k]["status"] = ST_RUNNING
                else:
                    state["stages"][k]["status"] = ST_DONE if postprocess_seen \
                        else ST_PENDING
        else:
            # Sin marcas de log: deducir del disco.
            for i, k in enumerate(order):
                s = state["stages"][k]
                nxt = state["stages"][order[i + 1]] if i + 1 < len(order) else None
                if s["pdb_total"] == 0:
                    s["status"] = ST_PENDING
                elif nxt and nxt["pdb_total"] > 0:
                    s["status"] = ST_DONE
                else:
                    s["status"] = ST_RUNNING

    # Ahora que el status está resuelto, calcular totales esperados.
    # MPNN total = RF.passed (sólo si RF DONE, porque el filtrado ya ocurrió).
    # AF total = MPNN.passed (ídem).
    if state["stages"]["rfdiffusion"]["status"] == ST_DONE:
        p = state["stages"]["rfdiffusion"]["passed"]
        state["stages"]["proteinmpnn"]["total"] = p if p > 0 else None
    if state["stages"]["proteinmpnn"]["status"] == ST_DONE:
        p = state["stages"]["proteinmpnn"]["passed"]
        state["stages"]["alphafold"]["total"] = p if p > 0 else None

    # --- Métricas de tiempo de la etapa activa -----------------------------
    # rate_s   = segundos por unidad (media de las últimas muestras).
    # elapsed_s= segundos transcurridos en la etapa (timestamps).
    # eta_s    = estimación de segundos restantes (rate × restantes) si total.
    # ts_cur   = timestep de difusión actual de RF (mini-barra del diseño).
    def _recent_mean(vals, n=10):
        v = [x for x in vals[-n:] if x and x > 0]
        return sum(v) / len(v) if v else None

    for k in STAGE_KEYS:
        s = state["stages"][k]
        if s["status"] != ST_RUNNING:
            continue
        # Ritmo
        if k == "rfdiffusion":
            # Derivar de timestamps entre "Making design" consecutivos.
            deltas = []
            for a, b in zip(rf_design_stamps, rf_design_stamps[1:]):
                d = (b - a).total_seconds()
                if 0 < d < 36000:
                    deltas.append(d)
            s["rate_s"] = _recent_mean(deltas)
            s["ts_cur"] = rf_last_ts
        else:
            s["rate_s"] = _recent_mean(durations[k])
        # Transcurrido en la etapa: por timestamps si los hay (RF), o por la
        # suma de duraciones reportadas (MPNN/AF, que no llevan timestamp).
        f, l = stage_first_stamp.get(k), stage_last_stamp.get(k)
        if f and l:
            s["elapsed_s"] = max(0.0, (l - f).total_seconds())
        elif durations[k]:
            s["elapsed_s"] = sum(durations[k])
        # ETA (solo si conocemos total)
        if s["rate_s"] and s["total"]:
            remaining = max(0, s["total"] - s["processed"])
            s["eta_s"] = s["rate_s"] * remaining

    # Detalle textual corto por etapa (línea de estado de la tarjeta).
    for k in STAGE_KEYS:
        s = state["stages"][k]
        if s["status"] == ST_DONE:
            s["detail"] = "Completado"
        elif s["status"] == ST_RUNNING:
            if k == "rfdiffusion":
                s["detail"] = f"Generando diseño {s['processed'] + 1}"
            elif s["total"]:
                s["detail"] = f"{s['processed']} / {s['total']} procesados"
            else:
                s["detail"] = f"{s['processed']} procesados"
        else:
            s["detail"] = "En espera"

    return state


def stage_progress_fraction(stage):
    """Fracción 0–1 para la barra de progreso de una etapa. DONE=1, PENDING=0,
    RUNNING = processed/total si se conoce el total, si no None (spinner)."""
    if stage["status"] == ST_DONE:
        return 1.0
    if stage["status"] == ST_PENDING:
        return 0.0
    total = stage.get("total")
    if total and total > 0:
        return max(0.0, min(1.0, stage["processed"] / total))
    return None  # indeterminado


# Colores por estado (coherentes con el resto del dashboard).
_STAGE_COLOR = {
    ST_DONE:    "#1D9E75",   # teal — completado
    ST_RUNNING: "#378ADD",   # azul — en ejecución
    ST_PENDING: "#888780",   # gris — en espera
}
_STAGE_ICON = {
    ST_DONE:    "bi bi-check-circle-fill",
    ST_RUNNING: "bi bi-arrow-repeat",
    ST_PENDING: "bi bi-clock",
}


def _fmt_duration(seconds):
    """Formatea segundos a un texto compacto: '45s', '12m', '3h 20m'."""
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s" if s and m < 10 else f"{m}m"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _fmt_rate(rate_s):
    """Ritmo legible: si <90s/u → 's/u'; si más → 'min/u'."""
    if not rate_s or rate_s <= 0:
        return "—"
    if rate_s < 90:
        return f"{rate_s:.0f}s/u"
    return f"{rate_s / 60:.1f}min/u"


def _stage_card(key, stage):
    """Construye la tarjeta visual de UNA etapa del pipeline."""
    color = _STAGE_COLOR[stage["status"]]
    frac  = stage_progress_fraction(stage)

    icon_cls = _STAGE_ICON[stage["status"]]
    if stage["status"] == ST_RUNNING:
        icon_cls += " bm-spin"
    header = html.Div([
        html.I(className=icon_cls + " me-2",
               style={'color': color, 'fontSize': '18px'}),
        html.Span(STAGE_LABELS[key], style={'fontWeight': 600}),
    ], style={'display': 'flex', 'alignItems': 'center',
              'marginBottom': '6px'})

    detail = html.Div(stage["detail"],
                      style={'fontSize': '12px', 'color': color,
                             'marginBottom': '8px', 'minHeight': '16px'})

    if frac is None:
        # Etapa en curso sin total conocido: barra indeterminada animada.
        bar = html.Div(
            html.Div(className='bm-indet-bar',
                     style={'backgroundColor': color}),
            className='bm-indet-track',
            style={'height': '6px', 'backgroundColor': '#e9ecef',
                   'borderRadius': '3px', 'marginBottom': '10px'})
    else:
        bar = html.Div(
            html.Div(style={'width': f"{frac * 100:.0f}%", 'height': '100%',
                            'backgroundColor': color,
                            'transition': 'width .4s ease'}),
            style={'height': '6px', 'backgroundColor': '#e9ecef',
                   'borderRadius': '3px', 'overflow': 'hidden',
                   'marginBottom': '10px'})

    def _row(label, value, strong=False):
        return html.Div([
            html.Span(label, style={'color': '#6c757d'}),
            html.Span(str(value), style={'fontWeight': 600 if strong else 400}),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'fontSize': '12px', 'lineHeight': '1.6'})

    # Etiquetas de conteo según estado:
    #  - RUNNING: aún no hay filtrado → "Generados" = PDBs brutos en disco.
    #  - DONE:    filtrado ocurrido   → "Pasaron" / "Descartados".
    #  - PENDING: nada aún.
    extras = []  # bloques añadidos bajo los conteos (mini-barra, chips)
    if stage["status"] == ST_DONE:
        counts = html.Div([
            _row("Pasaron", stage["passed"], strong=True),
            _row("Descartados", stage["discarded"]),
            _row("Total", stage["pdb_total"]),
        ])
    elif stage["status"] == ST_RUNNING:
        counts = html.Div([
            _row("Generados", stage["pdb_total"], strong=True),
        ])

        # --- Mini-barra del diseño actual (solo RF: timestep T_total→1) -----
        ts_cur, ts_total = stage.get("ts_cur"), stage.get("ts_total")
        if key == "rfdiffusion" and ts_cur and ts_total:
            done_frac = (ts_total - ts_cur) / ts_total  # 50→0% ... 1→~100%
            done_frac = max(0.0, min(1.0, done_frac))
            extras.append(html.Div([
                html.Div([
                    html.Span("Difusión del diseño actual",
                              style={'color': '#6c757d'}),
                    html.Span(f"paso {ts_cur}/{ts_total}",
                              style={'color': '#6c757d'}),
                ], style={'display': 'flex', 'justifyContent': 'space-between',
                          'fontSize': '11px', 'marginBottom': '3px'}),
                html.Div(
                    html.Div(style={'width': f"{done_frac * 100:.0f}%",
                                    'height': '100%', 'backgroundColor': color,
                                    'transition': 'width .4s ease'}),
                    style={'height': '4px', 'backgroundColor': '#e9ecef',
                           'borderRadius': '2px', 'overflow': 'hidden'}),
            ], style={'marginTop': '10px'}))

        # --- Chips de tiempo: ritmo · en etapa · ETA -----------------------
        chips = []

        def _chip(icon, text):
            return html.Span([
                html.I(className=icon + " me-1", style={'fontSize': '10px'}),
                text,
            ], style={'display': 'inline-flex', 'alignItems': 'center',
                      'backgroundColor': '#eef4fb', 'color': '#2c6cb0',
                      'borderRadius': '10px', 'padding': '2px 8px',
                      'fontSize': '11px', 'marginRight': '5px',
                      'marginTop': '5px'})

        if stage.get("rate_s"):
            chips.append(_chip("bi bi-speedometer2", _fmt_rate(stage["rate_s"])))
        if stage.get("elapsed_s"):
            chips.append(_chip("bi bi-clock-history",
                               _fmt_duration(stage["elapsed_s"])))
        if stage.get("eta_s"):
            chips.append(_chip("bi bi-hourglass-split",
                               f"ETA {_fmt_duration(stage['eta_s'])}"))
        if chips:
            extras.append(html.Div(chips, style={'marginTop': '4px'}))
    else:  # PENDING
        counts = html.Div([
            _row("Generados", "—"),
        ])

    border = ('2px solid #378ADD' if stage["status"] == ST_RUNNING
              else '0.5px solid rgba(0,0,0,.15)')
    opacity = 0.65 if stage["status"] == ST_PENDING else 1.0
    return html.Div([header, detail, bar, counts] + extras,
                    style={'flex': '1', 'backgroundColor': '#fff',
                           'border': border, 'borderRadius': '12px',
                           'padding': '14px', 'opacity': opacity})


def build_stage_cards(state):
    """Componente completo de tarjetas de etapa (reemplaza el log crudo)."""
    # Cabecera: experimento + finalización.
    head_children = []
    if state["experiment"]:
        cur, tot = state["experiment"]
        head_children.append(
            html.Span(f"Experimento {cur} de {tot}",
                      style={'fontWeight': 600, 'fontSize': '14px'}))
    if state["global_done"]:
        head_children.append(dbc.Badge(
            [html.I(className='bi bi-check-all me-1'), "Campaña finalizada"],
            color='success', className='ms-2'))
    elif state.get("postprocess"):
        head_children.append(dbc.Badge(
            [html.I(className='bi bi-gear-wide-connected me-1'),
             "Post-proceso (estadísticas/Rosetta)"],
            color='info', className='ms-2'))
    elif state["active"]:
        head_children.append(dbc.Badge(
            [html.I(className='bi bi-cpu me-1'),
             f"En curso: {STAGE_LABELS[state['active']]}"],
            color='primary', className='ms-2'))

    header = html.Div(head_children,
                      style={'display': 'flex', 'alignItems': 'center',
                             'marginBottom': '12px', 'minHeight': '24px'})

    # Tarjetas con flechas entre ellas.
    arrow = html.Div(html.I(className='bi bi-arrow-right',
                            style={'color': '#adb5bd', 'fontSize': '18px'}),
                     style={'display': 'flex', 'alignItems': 'center',
                            'padding': '0 6px'})
    cards = []
    for i, k in enumerate(STAGE_KEYS):
        cards.append(_stage_card(k, state["stages"][k]))
        if i < len(STAGE_KEYS) - 1:
            cards.append(arrow)
    row = html.Div(cards, style={'display': 'flex', 'alignItems': 'stretch'})

    # Última línea del log como contexto.
    last = html.Div([
        html.I(className='bi bi-info-circle me-2',
               style={'color': '#6c757d', 'fontSize': '13px'}),
        html.Span("Última línea: ", style={'color': '#6c757d'}),
        html.Span(state["last_line"] or "—",
                  style={'fontFamily': 'monospace', 'color': '#212529'}),
    ], style={'marginTop': '12px', 'fontSize': '12px',
              'backgroundColor': '#fff', 'border': '0.5px solid rgba(0,0,0,.1)',
              'borderRadius': '8px', 'padding': '8px 12px',
              'whiteSpace': 'nowrap', 'overflow': 'hidden',
              'textOverflow': 'ellipsis'})

    return html.Div([header, row, last])


# --- HPC ---------------------------------------------------------------------
from binderlab.core.hpc import get_hpc_status, HPC_TTL_SECONDS   # noqa: E402


# --- Motor de busqueda -------------------------------------------------------
from binderlab.core.search import (                 # noqa: E402
    OPERATORS, NUMERIC_OPS, coerce_value, apply_rule, apply_expression,
    apply_computed_columns, inject_annotation_columns,
)




# =============================================================================
# 4. DASH APP
# =============================================================================
# `suppress_callback_exceptions=True` es CRÍTICO porque el slider de
# trayectoria se crea dinámicamente: cuando arranca la app, ese componente
# todavía no existe en el DOM. Sin esta flag, Dash rechazaría los callbacks
# que apuntan a IDs "fantasma".
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
app.title = "BinderMonitor"
server = app.server  # objeto Flask subyacente, por si se quiere servir detrás de nginx

# CSS propio: pulso de la barra indeterminada (etapa en curso sin total
# conocido) y rotación del icono de la etapa activa. Se inyecta en el HTML
# base de la app para no depender de una carpeta assets/.
app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      @keyframes bm-spin { to { transform: rotate(360deg); } }
      .bm-spin { animation: bm-spin 1.4s linear infinite;
                 display: inline-block; }
      @keyframes bm-indet {
        0%   { transform: translateX(-100%); }
        100% { transform: translateX(280%); }
      }
      .bm-indet-track { position: relative; overflow: hidden; }
      .bm-indet-bar {
        position: absolute; top: 0; left: 0; height: 100%; width: 35%;
        border-radius: 3px; animation: bm-indet 1.6s ease-in-out infinite;
      }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


# =============================================================================
# 5. LAYOUT (interfaz)
# =============================================================================
SIDEBAR_STYLE = {
    'width': '24%', 'float': 'left', 'padding': '20px',
    'backgroundColor': '#f8f9fa', 'minHeight': '100vh',
    'borderRight': '1px solid #dee2e6', 'overflowY': 'auto',
}
MAIN_STYLE = {'width': '76%', 'float': 'right', 'padding': '20px'}


def _expr_placeholder(columns):
    """ejemplo del filtro-expresion derivado del pipeline activo,
    en vez de un string fijo con nombres de columna de RFdiffusion. Usa las
    2 primeras radar_metrics (ya saben high/low -> operador correcto); si no
    hay radar_metrics configuradas, cae a las 2 primeras columnas reales;
    si tampoco hay datos, cae al ejemplo clasico de RF.
    Umbral de ejemplo: 0.7/0.3 para metricas 'self_*' (escala 0-1, Complexa),
    80/10 en otro caso (escala 0-100, RF/Rosetta) -- solo es texto de ayuda,
    no afecta al filtro real."""
    radar_metrics = PIPELINE.get('radar_metrics') or RADAR_METRICS

    def thresh(m, sense):
        scale_01 = m.startswith('self_')
        hi, lo = (0.7, 0.3) if scale_01 else (80, 10)
        return ('>=' if sense == 'high' else '<=', hi if sense == 'high' else lo)

    picks = [(m, *thresh(m, s)) for m, s in radar_metrics.items()
             if m in (columns or [])][:2]
    if len(picks) == 2:
        (m1, op1, v1), (m2, op2, v2) = picks
        return f'{m1} {op1} {v1} and {m2} {op2} {v2}'
    if columns:
        return f'{columns[0]} >= 0 and {columns[min(1, len(columns)-1)]} <= 100'
    return 'plddt_binder >= 80 and pae_interaction <= 10'


def make_rule_row(rule_id, columns):
    """Construye una fila del motor de búsqueda.

    Cada fila tiene DOS modos (m01 — modo expresión):
      - 'classic': tres widgets (columna + operador + valor), el flujo clásico.
      - 'expr'   : un único Input de texto libre que se evalúa con df.eval(),
                   permitiendo aritmética y operadores compuestos en una sola
                   línea (p.ej. "plddt_binder >= 80 and pae_interaction <= 10").

    El selector de modo y el input de expresión están SIEMPRE en el DOM; se
    muestran/ocultan con CSS según el modo, de forma que el callback run_search
    pueda leerlos con el comodín ALL sin que falten componentes.

    El parámetro rule_id es un entero único que se mete en los
    atributos `id` de cada componente como diccionario:
        {'type': 'rule-col', 'index': rule_id}
    Así Dash puede manejarlas todas con un único callback usando
    el comodín ALL en lugar de un callback por fila."""
    col_options = [{'label': c, 'value': c} for c in (columns or [])]
    return html.Div(
        id={'type': 'rule-row', 'index': rule_id},
        style={'marginBottom': '8px',
               'borderLeft': '3px solid #e0e0e0', 'paddingLeft': '8px'},
        children=[
            # --- Selector de modo (clásico / expresión) --------------------
            html.Div(
                style={'display': 'flex', 'alignItems': 'center',
                       'gap': '10px', 'marginBottom': '6px'},
                children=[
                    dbc.RadioItems(
                        id={'type': 'rule-mode', 'index': rule_id},
                        options=[{'label': 'Clásico', 'value': 'classic'},
                                 {'label': 'Expresión', 'value': 'expr'}],
                        value='classic',
                        inline=True,
                        inputClassName='me-1', labelClassName='me-3',
                        style={'fontSize': '12px'},
                        persistence=True, persistence_type='session',
                    ),
                    html.Span(
                        'sintaxis pandas: and/or, ratios, aritmética',
                        id={'type': 'rule-expr-hint', 'index': rule_id},
                        className='text-muted fst-italic',
                        style={'fontSize': '11px', 'display': 'none'},
                    ),
                ],
            ),

            html.Div(
                style={'display': 'flex', 'gap': '8px', 'alignItems': 'center'},
                children=[
                    # --- Widgets del modo CLÁSICO ----------------------------------
                    html.Div(
                        id={'type': 'rule-classic-wrap', 'index': rule_id},
                        style={'display': 'flex', 'gap': '8px',
                               'alignItems': 'center', 'flex': '1'},
                        children=[
                            dcc.Dropdown(
                                id={'type': 'rule-col', 'index': rule_id},
                                options=col_options, placeholder='columna...',
                                style={'flex': '2', 'minWidth': '180px'},
                                persistence=True, persistence_type='session',
                            ),
                            dcc.Dropdown(
                                id={'type': 'rule-op', 'index': rule_id},
                                options=OPERATORS, value='>=',
                                clearable=False,
                                style={'flex': '1.2', 'minWidth': '160px'},
                                persistence=True, persistence_type='session',
                            ),
                            dcc.Input(
                                id={'type': 'rule-val', 'index': rule_id},
                                placeholder='valor...', type='text',
                                style={'flex': '1', 'minWidth': '110px',
                                       'padding': '6px', 'borderRadius': '4px',
                                       'border': '1px solid #ced4da'},
                                persistence=True, persistence_type='session',
                            ),
                        ],
                    ),

                    # --- Widget del modo EXPRESIÓN ---------------------------------
                    html.Div(
                        id={'type': 'rule-expr-wrap', 'index': rule_id},
                        style={'display': 'none', 'gap': '8px',
                               'alignItems': 'center', 'flex': '1'},
                        children=[
                            dcc.Input(
                                id={'type': 'rule-expr', 'index': rule_id},
                                placeholder=_expr_placeholder(columns),
                                type='text',
                                style={'flex': '1', 'minWidth': '320px',
                                       'padding': '6px', 'borderRadius': '4px',
                                       'border': '1px solid #ced4da',
                                       'fontFamily': 'monospace', 'fontSize': '13px'},
                                persistence=True, persistence_type='session',
                            ),
                        ],
                    ),

                    # Botón eliminar regla siempre visible en ambos modos
                    html.Button(
                        html.I(className='bi bi-trash'),
                        id={'type': 'rule-del', 'index': rule_id},
                        className='btn btn-sm btn-outline-danger',
                        title='Eliminar regla',
                    ),
                ],
            ),
        ],
    )


def serve_layout():
    """El layout se construye como una FUNCIÓN (no como variable estática)
    porque queremos que se reescaneé el directorio cada vez que un usuario
    nuevo abre la página. Si fuera estático, los experimentos se quedarían
    congelados en los que había al arrancar el script."""
    experiments = get_experiments(MASTER_DIR_INIT)
    return html.Div([

        # ---- Stores: datos invisibles compartidos entre callbacks ----------
        # Son como variables globales pero "por usuario": cada navegador que
        # abre la app tiene su propio Store, no se mezclan entre usuarios.
        dcc.Store(id='master-dir-store', data=MASTER_DIR_INIT),
        dcc.Store(id='metrics-cache-store',
                  data={'mtime': 0, 'path': None, 'records': None}),
        dcc.Store(id='available-columns-store', data=[]),
        dcc.Store(id='rules-store', data=[]),
        dcc.Store(id='rules-counter-store', data=0),
        # m09: columnas computadas definidas por el usuario, como lista de
        # dicts {'name': str, 'expr': str}. Se aplican al cache antes de
        # filtrar/ordenar, de modo que son indistinguibles de columnas reales.
        dcc.Store(id='computed-cols-store', data=[]),
        # m06: snapshot de las anotaciones del experimento activo, como dict
        # {design_name: {'tags': [...], 'note': str}}. Se refresca al cambiar
        # de experimento y al guardar. run_search lo usa para inyectar las
        # columnas _tags / _note (filtrables como cualquier otra).
        dcc.Store(id='annotations-store', data={}),

        # Polling: dispara un evento cada 10 segundos para refrescar datos
        # "vivos" (PDBs nuevos, métricas, log...). Sin esto, la página
        # estaría estática hasta que el usuario hiciera algo.
        dcc.Interval(id='polling-interval', interval=10_000, n_intervals=0),

        # =============== SIDEBAR =============================================
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H2([html.I(className='bi bi-bezier2 me-2'), 'BinderMonitor'],
                    style={'color': '#2C3E50', 'fontWeight': 700}),
            html.Small(f"Host: {HOST_LABEL} · Puerto: {PORT_LABEL}",
                       className='text-muted'),
            html.Hr(),

            # ---- Directorio raíz dinámico ---------------------------------
            html.H6('Directorio raíz (MASTER_DIR)'),
            dbc.InputGroup([
                dbc.Input(id='master-dir-input', value=MASTER_DIR_INIT,
                          placeholder='/data/.../master_dir', size='sm'),
                dbc.Button(html.I(className='bi bi-arrow-clockwise'),
                           id='master-dir-reload', color='secondary',
                           size='sm', title='Recargar experimentos'),
            ]),
            html.Div(id='master-dir-feedback',
                     className='text-muted small mt-1'),
            html.Hr(),

            # ---- Selector de experimento ----------------------------------
            html.H6('Experimento'),
            dcc.Dropdown(
                id='experiment-dropdown',
                options=[{'label': os.path.relpath(e, MASTER_DIR_INIT),
                          'value': e}
                         for e in experiments],
                value=experiments[0] if experiments else None,
                clearable=False,
                placeholder='Selecciona experimento...',
            ),
            html.Hr(),

            # ---- Selector de archivo de métricas --------------------------
            html.H6('Archivo de métricas'),
            dcc.Dropdown(id='metrics-file-dropdown', clearable=False,
                         placeholder='Selecciona archivo .sc / .csv...'),
            html.Small(id='metrics-file-info', className='text-muted'),
            html.Hr(),

            # ---- Filtros y ejes: solo visibles en pestaña Monitorización ---
            html.Div(id='contextual-sidebar', children=[
                html.H6('Filtros rápidos (dashboard)'),
                html.Label('Umbral eje X (≤ máx):', id='label-filter-x',
                           className='small mb-1'),
                dcc.Slider(id='filter-pae', min=0, max=30, step=1, value=10,
                           marks={0: '0', 10: '10', 20: '20', 30: '30'},
                           tooltip={'placement': 'bottom'}),
                html.Br(),
                html.Label('Umbral eje Y (≥ mín):', id='label-filter-y',
                           className='small mb-1'),
                dcc.Slider(id='filter-plddt', min=0, max=100, step=5, value=80,
                           marks={0: '0', 50: '50', 80: '80', 100: '100'},
                           tooltip={'placement': 'bottom'}),
                html.Hr(),

                html.H6('Ejes del scatter'),
                html.Label('X:', className='small mb-1'),
                dcc.Dropdown(id='xaxis_value', placeholder='columna X...'),
                html.Br(),
                html.Label('Y:', className='small mb-1'),
                dcc.Dropdown(id='yaxis_value', placeholder='columna Y...'),
                html.Br(),
                html.Label('Color (opcional):', className='small mb-1'),
                dcc.Dropdown(id='color_value', placeholder='columna color...'),
                html.Hr(),
            ]),

            # ---- Botón de parada de emergencia ----------------------------

            html.Hr(),
            dbc.Button([html.I(className='bi bi-power me-2'), 'APAGAR MONITOR'],
                id='shutdown-btn', color='dark', className='w-100 mb-2'),
            html.Div(id='shutdown-feedback', className='text-muted small mt-2'),
        ]),

        # =============== MAIN (área principal con pestañas) =================
        html.Div(style=MAIN_STYLE, children=[
            dcc.Tabs(id='tabs-main', value='tab-monitor', children=[

                # ----- TAB 1: vista general --------------------------------
                dcc.Tab(label='Monitorización', value='tab-monitor', children=[
                    dbc.Row(className='mt-3', children=[
                        dbc.Col(dbc.Card(dbc.CardBody([
                            html.H6('Diseños totales', className='text-muted'),
                            html.H2(id='metric-total',
                                    style={'color': '#3498db'}),
                        ]))),
                        dbc.Col(dbc.Card(dbc.CardBody([
                            html.H6('Hits (filtros rápidos)',
                                    className='text-muted'),
                            html.H2(id='metric-hits',
                                    style={'color': '#2ecc71'}),
                        ]))),
                        dbc.Col(dbc.Card(dbc.CardBody([
                            html.H6('Tasa de éxito', className='text-muted'),
                            html.H2(id='metric-rate',
                                    style={'color': '#f1c40f'}),
                        ]))),
                    ]),
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader('Nube de diseños (scatter)'),
                        dbc.CardBody(dcc.Graph(id='scatter-plot',
                                               config={'displaylogo': False})),
                    ]),
                ]),

                # ----- TAB 2: pipeline + estado HPC ------------------------
                dcc.Tab(label='Pipeline & HPC', value='tab-pipeline', children=[
                    # --- Fila superior: tarjetas de etapa (flujo visual) ---
                    dbc.Card(className='mt-3', children=[
                        dbc.CardHeader([
                            html.I(className='bi bi-diagram-3 me-2'),
                            html.Span('Estado del flujo de trabajo'),
                            dcc.Dropdown(
                                id='log-file-dropdown',
                                placeholder='log automático (más reciente)…',
                                clearable=True,
                                style={'marginTop': '6px', 'fontSize': '12px',
                                       'maxWidth': '480px'},
                            ),
                        ]),
                        dbc.CardBody([
                            html.Div(id='pipeline-stages'),
                            # Log crudo, oculto por defecto (acceso avanzado).
                            html.Div([
                                dbc.Button(
                                    [html.I(className='bi bi-terminal me-1'),
                                     'Ver log crudo'],
                                    id='toggle-raw-log', size='sm',
                                    color='outline-secondary',
                                    className='mt-3'),
                                dbc.Collapse(
                                    html.Pre(id='log-viewer', style={
                                        'height': '360px', 'overflowY': 'scroll',
                                        'backgroundColor': '#222',
                                        'color': '#0f0', 'padding': '10px',
                                        'fontSize': '12px', 'marginTop': '8px',
                                        'borderRadius': '4px',
                                    }),
                                    id='raw-log-collapse', is_open=False),
                            ]),
                        ]),
                    ]),
                    # --- Fila inferior: embudo + estado HPC ----------------
                    dbc.Row(className='mt-3', children=[
                        dbc.Col(width=7, children=[
                            dbc.Card([
                                dbc.CardHeader('Embudo de supervivencia '
                                               '(PDBs por etapa)'),
                                dbc.CardBody(dcc.Graph(id='funnel-plot')),
                            ]),
                        ]),
                        dbc.Col(width=5, children=[
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className='bi bi-cpu me-2'),
                                    'Estado HPC',
                                    dbc.Button('↻', id='hpc-refresh',
                                               size='sm', color='light',
                                               className='float-end',
                                               title='Refrescar (cache TTL '
                                                     f'{HPC_TTL_SECONDS}s)'),
                                ]),
                                dbc.CardBody([
                                    html.Span(id='hpc-source-badge'),
                                    html.Pre(id='hpc-status', style={
                                        'fontSize': '12px',
                                        'maxHeight': '200px',
                                        'overflowY': 'auto',
                                        'marginTop': '8px',
                                        'backgroundColor': '#1e1e1e',
                                        'color': '#0f0',
                                        'padding': '10px',
                                        'borderRadius': '4px',
                                    }),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),

                # ----- TAB 3: motor de búsqueda multicriterio --------------
                dcc.Tab(label='Búsqueda avanzada', value='tab-search', children=[
                    dbc.Card(className='mt-3', children=[
                        dbc.CardHeader([
                            html.I(className='bi bi-funnel-fill me-2'),
                            'Motor de filtrado multicriterio '
                            '(estilo extract_hits)',
                        ]),
                        dbc.CardBody([
                            html.Div(id='rules-container',
                                     style={'marginBottom': '12px'}),
                            html.Div([
                                dbc.Button([html.I(className='bi bi-plus-circle me-1'),
                                            'Añadir regla'],
                                           id='add-rule-btn', color='primary',
                                           size='sm', className='me-2'),
                                dbc.Button([html.I(className='bi bi-trash me-1'),
                                            'Limpiar reglas'],
                                           id='clear-rules-btn',
                                           color='outline-secondary',
                                           size='sm', className='me-2'),
                                dbc.Button([html.I(className='bi bi-download me-1'),
                                            'Exportar CSV'],
                                           id='export-csv-btn',
                                           color='success', size='sm',
                                           className='me-2'),
                                dcc.Download(id='download-csv'),
                                dbc.Button([html.I(className='bi bi-filetype-json me-1'),
                                            'Guardar manifest'],
                                           id='export-manifest-btn',
                                           color='outline-success', size='sm'),
                                dcc.Download(id='download-manifest'),
                            ]),
                            html.Hr(),

                            # ----- m04: importar configuración desde manifest --
                            html.Div([
                                dcc.Upload(
                                    id='manifest-upload',
                                    children=html.Div([
                                        html.I(className='bi bi-upload me-2'),
                                        'Arrastra aquí un manifest.json para '
                                        'restaurar reglas, orden y columnas '
                                        'computadas',
                                    ]),
                                    multiple=False,
                                    style={'border': '1px dashed #ced4da',
                                           'borderRadius': '6px',
                                           'padding': '12px',
                                           'textAlign': 'center',
                                           'fontSize': '13px',
                                           'color': '#6c757d',
                                           'cursor': 'pointer'},
                                ),
                                html.Small(id='manifest-feedback',
                                           className='text-muted'),
                            ], style={'marginBottom': '12px'}),

                            # ----- m09: columnas computadas personalizadas -----
                            html.Div([
                                html.Label(
                                    [html.I(className='bi bi-calculator me-1'),
                                     'Columna computada (m09):'],
                                    className='fw-bold',
                                    style={'fontSize': '13px',
                                           'marginRight': '8px'}),
                                dcc.Input(
                                    id='computed-name',
                                    placeholder='nombre (ej: efficiency)',
                                    type='text',
                                    style={'marginRight': '6px', 'padding': '5px',
                                           'borderRadius': '4px',
                                           'border': '1px solid #ced4da',
                                           'width': '180px'},
                                    persistence=True,
                                    persistence_type='session'),
                                dcc.Input(
                                    id='computed-expr',
                                    placeholder='expresión (ej: dSASA / binder_length)',
                                    type='text',
                                    style={'marginRight': '6px', 'padding': '5px',
                                           'borderRadius': '4px',
                                           'border': '1px solid #ced4da',
                                           'width': '300px',
                                           'fontFamily': 'monospace',
                                           'fontSize': '12px'},
                                    persistence=True,
                                    persistence_type='session'),
                                dbc.Button('Añadir columna',
                                           id='add-computed-btn',
                                           color='info', size='sm'),
                                html.Small(id='computed-feedback',
                                           className='ms-2 text-muted'),
                            ], style={'marginBottom': '12px',
                                      'padding': '10px',
                                      'backgroundColor': '#f0f7ff',
                                      'borderRadius': '6px'}),

                            html.Hr(),
                            html.Div([
                                html.Label('Ordenar por:', className='me-2'),
                                dcc.Dropdown(id='sort-by-dropdown',
                                             style={'width': '220px',
                                                    'display': 'inline-block',
                                                    'verticalAlign': 'middle'}),
                                html.Span(' ', style={'marginRight': '12px'}),
                                dbc.Switch(id='sort-asc-switch',
                                           label='Ascendente', value=True,
                                           style={'display': 'inline-block'}),
                            ]),

                            # ----- m03: explorador de umbrales -----------------
                            html.Hr(),
                            html.Div([
                                html.Label(
                                    [html.I(className='bi bi-graph-up me-1'),
                                     'Explorador de umbrales (m03):'],
                                    className='fw-bold',
                                    style={'fontSize': '13px',
                                           'marginRight': '8px'}),
                                dcc.Dropdown(
                                    id='threshold-col',
                                    placeholder='elige una columna numérica...',
                                    style={'width': '260px',
                                           'display': 'inline-block',
                                           'verticalAlign': 'middle'}),
                                dcc.Graph(id='threshold-hist',
                                          style={'height': '220px',
                                                 'marginTop': '8px'},
                                          config={'displayModeBar': False}),
                                html.Small(id='threshold-readout',
                                           className='text-muted'),
                            ], style={'marginTop': '12px'}),
                        ]),
                    ]),
                    dbc.Card(className='mt-3', children=[
                        dbc.CardHeader([
                            html.Span('Resumen embudo: ', className='fw-bold'),
                            html.Span(id='search-summary'),
                        ]),
                        dbc.CardBody([
                            html.Div(id='search-funnel-log',
                                     style={'fontFamily': 'monospace',
                                            'fontSize': '12px',
                                            'maxHeight': '180px',
                                            'overflowY': 'auto',
                                            'backgroundColor': '#f4f4f4',
                                            'padding': '8px',
                                            'borderRadius': '4px',
                                            'marginBottom': '12px'}),
                            html.Div(id='search-results-table'),
                        ]),
                    ]),
                ]),

                # ----- TAB 4: visor 3D + trayectorias ----------------------
                dcc.Tab(label='Visor 3D & Trayectorias', value='tab-viewer',
                        children=[
                    dbc.Row(className='mt-3', children=[
                        dbc.Col(width=4, children=[
                            dbc.Card([
                                dbc.CardHeader('Selección de diseño'),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='hit-dropdown',
                                        placeholder='Selecciona un hit '
                                                    'superviviente...',
                                    ),
                                    html.Hr(),

                                    # -- Controles de trayectoria --------
                                    # IMPORTANTE: estos componentes están
                                    # SIEMPRE en el DOM (no se crean/
                                    # destruyen dinámicamente). Así Dash
                                    # puede conectar sus callbacks desde
                                    # el arranque. Se muestran/ocultan
                                    # con display:none/block.
                                    dcc.Store(id='traj-pdb-store',
                                              data=None),
                                    html.Div(
                                        id='traj-controls-wrapper',
                                        style={'display': 'none'},
                                        children=[
                                            html.H6([
                                                html.I(className='bi bi-arrow-repeat me-2'),
                                                'Trayectoria RFdiffusion',
                                            ]),
                                            html.Div(
                                                id='traj-frame-label',
                                                className='small text-muted mb-1',
                                            ),
                                            dcc.Slider(
                                                id='traj-slider',
                                                min=0, max=1, step=1,
                                                value=0,
                                                updatemode='mouseup',
                                                tooltip={
                                                    'placement': 'bottom',
                                                    'always_visible': True,
                                                },
                                            ),
                                            dbc.ButtonGroup([
                                                dbc.Button(
                                                    '⏮',
                                                    id='traj-first',
                                                    size='sm',
                                                    color='outline-secondary',
                                                    title='Primer paso (ruido)',
                                                ),
                                                dbc.Button(
                                                    '◀',
                                                    id='traj-prev',
                                                    size='sm',
                                                    color='outline-secondary',
                                                    title='Paso anterior',
                                                ),
                                                dbc.Button(
                                                    '▶ Play',
                                                    id='traj-play-btn',
                                                    size='sm',
                                                    color='primary',
                                                ),
                                                dbc.Button(
                                                    '▶',
                                                    id='traj-next',
                                                    size='sm',
                                                    color='outline-secondary',
                                                    title='Paso siguiente',
                                                ),
                                                dbc.Button(
                                                    '⏭',
                                                    id='traj-last',
                                                    size='sm',
                                                    color='outline-secondary',
                                                    title='Último paso (final)',
                                                ),
                                            ], className='mt-2'),
                                            # Salto manual a un paso concreto
                                            html.Div([
                                                html.Label(
                                                    'Ir al paso:',
                                                    className='small me-2'),
                                                dcc.Input(
                                                    id='traj-goto-input',
                                                    type='number',
                                                    min=1, step=1,
                                                    debounce=True,
                                                    style={'width': '80px',
                                                           'padding': '3px',
                                                           'borderRadius': '4px',
                                                           'border': '1px solid #ced4da'},
                                                ),
                                                dbc.Button(
                                                    [html.I(className='bi bi-house me-1'),
                                                     'Estructura final (AF2)'],
                                                    id='traj-show-final',
                                                    size='sm',
                                                    color='outline-primary',
                                                    className='ms-2',
                                                ),
                                            ], className='mt-2',
                                               style={'display': 'flex',
                                                      'alignItems': 'center'}),
                                            # Metrónomo invisible para la animación
                                            dcc.Interval(
                                                id='traj-play-interval',
                                                interval=500,
                                                n_intervals=0,
                                                disabled=True,
                                            ),
                                            html.Div([
                                                html.Label('Velocidad:',
                                                           className='small me-2'),
                                                dcc.RadioItems(
                                                    id='traj-speed',
                                                    options=[
                                                        {'label': ' Lenta', 'value': 1000},
                                                        {'label': ' Normal', 'value': 500},
                                                        {'label': ' Rápida', 'value': 150},
                                                    ],
                                                    value=500, inline=True,
                                                    className='small',
                                                ),
                                            ], className='mt-2'),
                                        ],
                                    ),
                                    html.Div(
                                        id='traj-message',
                                        className='text-muted small',
                                    ),

                                    html.Hr(),
                                    html.Div(id='viewer-info',
                                             className='small text-muted'),
                                ]),
                            ]),
                            # -- Radar de métricas del diseño seleccionado --
                            dbc.Card([
                                dbc.CardHeader('Métricas del diseño'),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='radar-plot',
                                        style={'height': '320px'},
                                        config={'displaylogo': False},
                                    ),
                                    html.Div(id='radar-values',
                                             style={'fontSize': '12px',
                                                    'marginTop': '6px'}),
                                ]),
                            ], className='mt-3'),
                            # -- Anotaciones persistentes (m06) -------------
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className='bi bi-bookmark-star me-2'),
                                    'Anotaciones',
                                ]),
                                dbc.CardBody([
                                    html.Div(
                                        'Selecciona un diseño para anotarlo.',
                                        id='annot-hint',
                                        className='small text-muted'),
                                    dbc.Checklist(
                                        id='annot-tags',
                                        options=[{'label': f' {t}', 'value': t}
                                                 for t in DEFAULT_TAGS],
                                        value=[],
                                        className='mb-2',
                                    ),
                                    dcc.Textarea(
                                        id='annot-note',
                                        placeholder='Notas sobre este diseño...',
                                        style={'width': '100%', 'height': '90px',
                                               'fontSize': '13px',
                                               'borderRadius': '4px',
                                               'border': '1px solid #ced4da',
                                               'padding': '6px'},
                                    ),
                                    html.Div([
                                        dbc.Button(
                                            [html.I(className='bi bi-save me-1'),
                                             'Guardar anotaciones'],
                                            id='annot-save-btn',
                                            color='primary', size='sm',
                                            className='mt-2'),
                                        html.Small(
                                            id='annot-feedback',
                                            className='ms-2 text-muted'),
                                    ]),
                                ]),
                            ], className='mt-3'),
                        ]),
                        dbc.Col(width=8, children=[
                            dbc.Card([
                                dbc.CardHeader('Visor Molstar 3D'),
                                dbc.CardBody([
                                    dashmolstar.MolstarViewer(
                                        id='molstar-viewer',
                                        style={'height': '600px',
                                               'width': '100%'},
                                    ),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ])


app.layout = serve_layout


# =============================================================================
# 6. CALLBACKS
# =============================================================================
# Recordatorio rápido sobre Dash:
# Cada @callback escucha cambios en sus Inputs y, cuando uno cambia,
# ejecuta la función que va debajo del decorador. El return se asigna
# a los Outputs en el mismo orden en el que están declarados.
# =============================================================================


# ----- 6.1 Directorio raíz: recargar lista de experimentos ------------------
@callback(
    Output('master-dir-store', 'data'),
    Output('experiment-dropdown', 'options'),
    Output('experiment-dropdown', 'value'),
    Output('master-dir-feedback', 'children'),
    Input('master-dir-reload', 'n_clicks'),
    Input('polling-interval', 'n_intervals'),
    State('master-dir-input', 'value'),
    State('experiment-dropdown', 'value'),
)
def reload_experiments(_clicks, _n, dir_value, current_exp):
    """Reescanea el directorio raíz al pulsar el botón ↻ o cada vez que
    salta el polling. Conserva el experimento seleccionado si sigue
    existiendo."""
    dir_value = (dir_value or '').strip()
    if not dir_value:
        return no_update, [], None, "Introduce un directorio."
    abs_path = os.path.abspath(os.path.expanduser(dir_value))
    if not os.path.isdir(abs_path):
        return no_update, [], None, f"⚠ No existe: {abs_path}"
    exps = get_experiments(abs_path)
    options = [{'label': os.path.relpath(e, abs_path), 'value': e}
               for e in exps]
    new_value = current_exp if current_exp in exps else (exps[0] if exps else None)
    msg = f"✓ {len(exps)} experimento(s) detectados en {abs_path}"
    return abs_path, options, new_value, msg


# ----- 6.2 Detección de archivos de métricas disponibles --------------------
@callback(
    Output('metrics-file-dropdown', 'options'),
    Output('metrics-file-dropdown', 'value'),
    Output('metrics-file-info', 'children'),
    Input('experiment-dropdown', 'value'),
    Input('polling-interval', 'n_intervals'),
    State('metrics-file-dropdown', 'value'),
)
def update_metric_files(exp_dir, _n, current_value):
    """Cuando cambia el experimento, lista los archivos .sc / .csv
    disponibles. Por defecto selecciona out.sc si existe."""
    if not exp_dir:
        return [], None, ""
    files = get_metric_files(exp_dir)
    options = [{'label': os.path.basename(f), 'value': f} for f in files]
    if current_value and current_value in files:
        value = current_value
    elif files:
        out_sc = next((f for f in files if os.path.basename(f) == 'out.sc'), None)
        value = out_sc or files[0]
    else:
        value = None
    info = (f"{len(files)} archivo(s) de métricas detectado(s)"
            if files else "⚠ Sin métricas")
    return options, value, info


# ----- 6.3 Caché del archivo de métricas ------------------------------------
@callback(
    Output('metrics-cache-store', 'data'),
    Output('available-columns-store', 'data'),
    Input('metrics-file-dropdown', 'value'),
    Input('polling-interval', 'n_intervals'),
    State('metrics-cache-store', 'data'),
)
def refresh_metrics_cache(metric_file, _n, cache):
    """Solo re-lee el archivo si su mtime ha cambiado, y SOLO emite datos
    nuevos a los Stores cuando realmente han cambiado.

    *** CLAVE PARA EL BUG DE BÚSQUEDA AVANZADA ***
    Antes, cuando el archivo NO había cambiado, esta función devolvía el
    cache (y una lista de columnas reconstruida) en CADA polling. Como
    devolvía objetos nuevos cada 10 s, disparaba en cascada a
    `update_rule_col_options` (reescribía los dropdowns de las reglas) y a
    `run_search` (reconstruía la tabla entera), reseteando el operador de
    las reglas a su valor por defecto y el filtrado/orden nativo de la
    tabla. Con un run EN MARCHA el mtime cambia constantemente, así que la
    cascada se repetía sin parar.

    La solución es devolver `no_update` en los dos outputs siempre que no
    haya un cambio real: así los callbacks de abajo NO se vuelven a
    ejecutar y la UI deja de resetearse. Cuando aparecen diseños nuevos
    (mtime cambia de verdad), sí recarga y propaga — que es lo deseable."""
    cache = cache or {'mtime': 0, 'path': None, 'records': None}

    if not metric_file or not os.path.exists(metric_file):
        # Si ya estaba vacío, no re-disparamos; si había algo, lo limpiamos.
        if cache.get('path') is None and cache.get('records') is None:
            return no_update, no_update
        return {'mtime': 0, 'path': None, 'records': None}, []

    try:
        cur_mtime = os.path.getmtime(metric_file)
    except OSError:
        # Archivo en plena escritura: conservamos TODO sin re-disparar.
        return no_update, no_update

    # Sin cambios → no emitir nada (evita la cascada que reseteaba la UI).
    if (cache.get('path') == metric_file
            and cache.get('mtime', 0) >= cur_mtime
            and cache.get('records') is not None):
        return no_update, no_update

    df = parse_metrics_file(metric_file)

    # Lectura a medias (pipeline escribiendo): mantenemos lo viejo sin
    # re-disparar, para no parpadear a pantalla en blanco.
    if df.empty and cache.get('records'):
        return no_update, no_update

    new_cache = {
        'mtime': cur_mtime,
        'path': metric_file,
        'records': df.to_dict('records') if not df.empty else [],
    }
    return new_cache, list(df.columns)


# ----- 6.4 Poblar dropdowns de ejes con las columnas reales -----------------
@callback(
    Output('xaxis_value', 'options'),
    Output('yaxis_value', 'options'),
    Output('color_value', 'options'),
    Output('xaxis_value', 'value'),
    Output('yaxis_value', 'value'),
    Output('sort-by-dropdown', 'options'),
    Input('available-columns-store', 'data'),
    Input('computed-cols-store', 'data'),
    State('xaxis_value', 'value'),
    State('yaxis_value', 'value'),
)
def populate_axis_dropdowns(columns, computed, cur_x, cur_y):
    """Las columnas del scatter se descubren del archivo de métricas;
    no están hardcodeadas. Si están las clásicas (pae_interaction,
    plddt_binder), se preseleccionan. Las columnas computadas (m09) se
    añaden al final, disponibles para ordenar."""
    columns = list(columns or [])
    opts = [{'label': c, 'value': c} for c in columns]
    # Sort dropdown incluye además las columnas computadas
    sort_cols = list(columns)
    for c in (computed or []):
        nm = c.get('name')
        if nm and nm not in sort_cols:
            sort_cols.append(nm)
    sort_opts = [{'label': c, 'value': c} for c in sort_cols]
    new_x = cur_x if cur_x in columns else next(
        (c for c in ('pae_interaction', 'self_complex_i_pAE') if c in columns),
        columns[0] if columns else None)
    new_y = cur_y if cur_y in columns else next(
        (c for c in ('plddt_binder', 'self_complex_i_pTM', 'self_complex_pLDDT')
         if c in columns),
        columns[1] if len(columns) > 1 else None)
    return opts, opts, opts, new_x, new_y, sort_opts


# ----- 6.5 Dashboard principal: scatter + métricas + lista de hits ----------
@callback(
    Output('filter-pae', 'min'), Output('filter-pae', 'max'),
    Output('filter-pae', 'step'), Output('filter-pae', 'value'),
    Output('filter-pae', 'marks'), Output('label-filter-x', 'children'),
    Output('filter-plddt', 'min'), Output('filter-plddt', 'max'),
    Output('filter-plddt', 'step'), Output('filter-plddt', 'value'),
    Output('filter-plddt', 'marks'), Output('label-filter-y', 'children'),
    Input('xaxis_value', 'value'),
    Input('yaxis_value', 'value'),
    State('metrics-cache-store', 'data'),
    prevent_initial_call=True,
)
def adapt_filter_ranges(xaxis, yaxis, cache):
    """cada slider se ajusta al rango real de la columna de su eje,
    para poder filtrar por cualquier métrica. X arranca en su máximo y Y en su
    mínimo (sin filtrar de inicio); el usuario arrastra para acotar."""
    records = (cache or {}).get('records') or []
    df = pd.DataFrame(records)

    def rng(col, d_lo, d_hi, d_val, d_step):
        if col and col in df.columns:
            v = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(v) and float(v.max()) > float(v.min()):
                lo, hi = float(v.min()), float(v.max())
                step = round((hi - lo) / 50, 4) or 0.001
                marks = {round(lo, 3): f'{lo:.2g}', round(hi, 3): f'{hi:.2g}'}
                return lo, hi, step, marks
        return d_lo, d_hi, d_step, {d_lo: str(d_lo), d_hi: str(d_hi)}

    xlo, xhi, xstep, xmarks = rng(xaxis, 0, 30, 10, 1)
    ylo, yhi, ystep, ymarks = rng(yaxis, 0, 100, 80, 5)
    lx = f'Umbral {xaxis} (≤ máx):' if xaxis else 'Umbral eje X (≤ máx):'
    ly = f'Umbral {yaxis} (≥ mín):' if yaxis else 'Umbral eje Y (≥ mín):'
    # X filtra por arriba → empieza en el máximo (todo pasa); Y por abajo → mínimo
    return (xlo, xhi, xstep, xhi, xmarks, lx,
            ylo, yhi, ystep, ylo, ymarks, ly)


@callback(
    Output('scatter-plot', 'figure'),
    Output('metric-total', 'children'),
    Output('metric-hits', 'children'),
    Output('metric-rate', 'children'),
    Output('hit-dropdown', 'options'),
    Input('metrics-cache-store', 'data'),
    Input('filter-pae', 'value'),
    Input('filter-plddt', 'value'),
    Input('xaxis_value', 'value'),
    Input('yaxis_value', 'value'),
    Input('color_value', 'value'),
)
def update_dashboard(cache, xthr, ythr, xaxis, yaxis, color_col):
    """Refresca el scatter, las tarjetas de métricas y la lista de diseños
    seleccionables en el visor. Los dos umbrales se aplican a las columnas
    elegidas en los ejes: X por arriba (<=), Y por abajo (>=). Así el filtro
    funciona con cualquier métrica, no solo pae/plddt."""
    records = (cache or {}).get('records') or []
    if not records:
        return go.Figure(), "0", "0", "0%", []
    df = pd.DataFrame(records)
    if xaxis not in df.columns or yaxis not in df.columns:
        return go.Figure(), str(len(df)), "—", "—", []

    # umbral X (<=) y umbral Y (>=) sobre las columnas elegidas.
    # Para labs RF los ejes por defecto son pae/plddt -> mismo comportamiento.
    mask = pd.Series(True, index=df.index)
    if xthr is not None:
        mask &= pd.to_numeric(df[xaxis], errors='coerce') <= xthr
    if ythr is not None:
        mask &= pd.to_numeric(df[yaxis], errors='coerce') >= ythr
    df['is_hit'] = mask

    total = len(df)
    hits = int(mask.sum())
    rate = f"{(hits / total) * 100:.1f}%" if total else "0%"

    hover_cols = [c for c in ('description', 'base_name') if c in df.columns]
    if color_col and color_col in df.columns:
        fig = px.scatter(df, x=xaxis, y=yaxis, color=color_col,
                         hover_data=hover_cols,
                         marginal_x='violin', marginal_y='violin',
                         color_continuous_scale='Viridis')
    else:
        fig = px.scatter(df, x=xaxis, y=yaxis, color='is_hit',
                         color_discrete_map={True: '#2ecc71', False: '#bdc3c7'},
                         hover_data=hover_cols,
                         marginal_x='violin', marginal_y='violin')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      transition_duration=300)

    # Lista de diseños seleccionables en el visor 3D
    if 'base_name' in df.columns:
        hit_options = [
            {'label': bn, 'value': bn}
            for bn in df.loc[mask, 'base_name'].dropna().unique()
        ]
    else:
        hit_options = []
    return fig, str(total), str(hits), rate, hit_options


# ----- 6.6 Pipeline tracking: funnel + opciones de log ----------------------
@callback(
    Output('funnel-plot', 'figure'),
    Output('log-file-dropdown', 'options'),
    Output('log-file-dropdown', 'value'),
    Input('experiment-dropdown', 'value'),
    Input('polling-interval', 'n_intervals'),
    State('log-file-dropdown', 'value'),
)
def update_pipeline(exp_dir, _n, current_log):
    """Cuenta los PDBs supervivientes/descartados en cada etapa del
    pipeline y refresca la LISTA de logs disponibles (versátil: .log/.out/
    .err, recursivo + niveles padre). El contenido del log lo muestra
    `update_log_viewer` según lo que el usuario elija en el dropdown."""
    if not exp_dir:
        return go.Figure(), [], None

    fc = {k: count_funnel_stage(exp_dir, STAGE_DIRS[k], STAGE_GLOBS[k])
          for k in STAGE_KEYS}

    stages    = [STAGE_LABELS[k] for k in STAGE_KEYS]
    survivors = [fc[k][1] for k in STAGE_KEYS]
    discarded = [fc[k][2] for k in STAGE_KEYS]
    colors    = [STAGE_COLORS[k] for k in STAGE_KEYS]

    all_totals = [fc[k][0] for k in STAGE_KEYS]
    if all(v == 0 for v in all_totals):
        # Nada contabilizado todavía.
        fig = go.Figure()
        fig.add_annotation(
            text="Pipeline en ejecución — aún no hay PDBs contabilizados",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#888"))
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
    else:
        # Embudo de retención: cada barra = supervivientes de la etapa.
        # "percent previous" comunica la tasa de retención etapa a etapa,
        # que es justo lo que mide un pipeline de cribado.
        fig = go.Figure(go.Funnel(
            y=stages,
            x=survivors,
            textposition="inside",
            textinfo="value+percent previous",
            marker=dict(color=colors),
            connector=dict(line=dict(color='#cfd8dc', width=1)),
            hovertemplate=("<b>%{y}</b><br>Supervivientes: %{x}"
                           "<br>Retención vs etapa previa: "
                           "%{percentPrevious:.0%}<extra></extra>"),
        ))
        # Anotar los descartados de cada etapa a la derecha de su barra.
        for i, (st, dsc, surv) in enumerate(zip(stages, discarded, survivors)):
            if dsc > 0:
                rate = dsc / (surv + dsc) if (surv + dsc) else 0
                fig.add_annotation(
                    x=max(survivors) * 1.02, y=st, xanchor='left',
                    text=f"<span style='color:#C0392B'>✕ {dsc} descartados "
                         f"({rate:.0%})</span>",
                    showarrow=False, font=dict(size=11))
        fig.update_layout(
            margin=dict(l=20, r=140, t=30, b=20),
            funnelmode="stack",
            showlegend=False,
            xaxis=dict(range=[0, max(survivors) * 1.35]),
        )

    # --- Lista versátil de logs (recursiva + niveles padre) -------------
    logs = find_experiment_logs(exp_dir)
    options = []
    for p in logs:
        # Etiqueta legible: nombre del log + carpeta contenedora corta
        rel = os.path.relpath(p, exp_dir)
        if rel.startswith('..'):
            # vive fuera del experimento (carpeta múltiple / raíz)
            label = f"⤴ {os.path.basename(p)}  ({os.path.basename(os.path.dirname(p))}/)"
        else:
            label = rel if len(rel) < 55 else f"…/{os.path.basename(p)}"
        options.append({'label': label, 'value': p})

    # Mantener la selección del usuario si sigue existiendo; si no, auto
    # (el más relevante/reciente, que find_experiment_logs pone primero).
    if current_log and any(o['value'] == current_log for o in options):
        value = current_log
    else:
        value = logs[0] if logs else None

    return fig, options, value


# ----- 6.6b Contenido del log seleccionado ----------------------------------
@callback(
    Output('log-viewer', 'children'),
    Input('log-file-dropdown', 'value'),
    Input('polling-interval', 'n_intervals'),
)
def update_log_viewer(log_path, _n):
    """Muestra las últimas 200 líneas del log elegido. Se refresca con el
    polling para seguir un run en marcha (tail en vivo)."""
    if not log_path:
        return ("Sin logs detectados para este experimento.\n"
                "Se buscan *.log, *.out y *.err de forma recursiva y en "
                "las carpetas padre.")
    if not os.path.isfile(log_path):
        return f"El log seleccionado ya no existe:\n{log_path}"
    try:
        lines = _tail_lines(log_path, max_bytes=500_000)
        return f"[{log_path}]\n" + "\n".join(lines[-200:])
    except Exception as e:
        return f"Error leyendo log ({log_path}): {e}"


# ----- 6.6c Tarjetas de etapa (flujo visual, reemplaza el volcado crudo) ----
@callback(
    Output('pipeline-stages', 'children'),
    Input('log-file-dropdown', 'value'),
    Input('experiment-dropdown', 'value'),
    Input('polling-interval', 'n_intervals'),
)
def update_pipeline_stages(log_path, exp_dir, _n):
    """Traduce el log + el disco a tarjetas de etapa (RF → MPNN → AF) con
    estado, barra de progreso y conteos. Se refresca con el polling para
    seguir un run en vivo. Si no hay experimento seleccionado, muestra aviso."""
    if not exp_dir:
        return html.Div("Selecciona un experimento para ver el flujo.",
                        style={'color': '#6c757d', 'fontSize': '13px',
                               'padding': '8px 0'})
    state = parse_pipeline_state(log_path, exp_dir)
    return build_stage_cards(state)


# ----- 6.6d Mostrar/ocultar el log crudo ------------------------------------
@callback(
    Output('raw-log-collapse', 'is_open'),
    Input('toggle-raw-log', 'n_clicks'),
    State('raw-log-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_raw_log(_n, is_open):
    """Alterna la visibilidad del log de texto crudo (acceso avanzado)."""
    return not is_open


# ----- 6.7 Estado HPC -------------------------------------------------------
@callback(
    Output('hpc-status', 'children'),
    Output('hpc-source-badge', 'children'),
    Input('hpc-refresh', 'n_clicks'),
    Input('polling-interval', 'n_intervals'),
)
def update_hpc(_clicks, _n):
    """get_hpc_status() ya gestiona la caché internamente, así que aunque
    este callback se dispare cada 10s por el polling, solo lanzará squeue
    o nvidia-smi de verdad cada HPC_TTL_SECONDS."""
    source, content = get_hpc_status()
    color_map = {'SLURM': 'primary', 'GPU': 'success', 'N/A': 'secondary'}
    badge = dbc.Badge(source, color=color_map.get(source, 'secondary'),
                      className='me-2')
    return content, badge


# =============================================================================
# 6.8 — 6.12 MOTOR DE BÚSQUEDA AVANZADA
# =============================================================================
# El motor permite encadenar reglas (columna · operador · valor) sobre el
# DataFrame de métricas, tal y como lo hace extract_hits.py por línea de
# comandos. Las filas de la UI son DINÁMICAS: se añaden/quitan en runtime.
#
# El truco para gestionar un número variable de componentes en Dash es
# darles un id de tipo diccionario {'type': '...', 'index': N} y luego
# referirlos en los callbacks con el comodín ALL. Así un único callback
# maneja TODAS las filas de reglas, sea una o cincuenta.
# =============================================================================

@callback(
    Output('rules-store', 'data'),
    Output('rules-counter-store', 'data'),
    Input('add-rule-btn', 'n_clicks'),
    Input('clear-rules-btn', 'n_clicks'),
    Input({'type': 'rule-del', 'index': ALL}, 'n_clicks'),
    State('rules-store', 'data'),
    State('rules-counter-store', 'data'),
    prevent_initial_call=True,
)
def manage_rules(_add_n, _clear_n, _del_clicks, rules, counter):
    """Gestiona el ciclo de vida de las reglas: añadir, limpiar, eliminar.
    `ctx.triggered_id` indica qué botón disparó el callback."""
    rules = list(rules or [])
    counter = int(counter or 0)
    triggered = ctx.triggered_id

    if triggered == 'add-rule-btn':
        rules.append(counter)
        counter += 1
        return rules, counter

    if triggered == 'clear-rules-btn':
        return [], counter

    # Si fue uno de los botones "borrar" de una fila concreta:
    if isinstance(triggered, dict) and triggered.get('type') == 'rule-del':
        if not any(_del_clicks or []):
            raise PreventUpdate
        idx = triggered.get('index')
        return [r for r in rules if r != idx], counter

    return rules, counter


# ----- m09: gestión de columnas computadas ----------------------------------
@callback(
    Output('computed-cols-store', 'data'),
    Output('computed-feedback', 'children'),
    Output('computed-name', 'value'),
    Output('computed-expr', 'value'),
    Input('add-computed-btn', 'n_clicks'),
    State('computed-name', 'value'),
    State('computed-expr', 'value'),
    State('computed-cols-store', 'data'),
    State('metrics-cache-store', 'data'),
    prevent_initial_call=True,
)
def add_computed_column(_n, name, expr, computed, cache):
    """Valida y registra una columna computada nueva. La expresión se
    prueba contra el dataset actual antes de aceptarla, para dar feedback
    inmediato si referencia una columna inexistente o tiene un error."""
    computed = list(computed or [])
    if not name or not str(name).strip() or not expr or not str(expr).strip():
        return no_update, "✕ Indica nombre y expresión.", no_update, no_update
    name = str(name).strip()

    # Validar contra el dataset actual
    records = (cache or {}).get('records') or []
    if records:
        df = pd.DataFrame(records)
        test = apply_computed_columns(df, [{'name': name, 'expr': expr}])
        if name not in test.columns:
            return (no_update,
                    "✕ La expresión falló (¿columna inexistente?). No añadida.",
                    no_update, no_update)

    # Reemplazar si ya existía una con ese nombre
    computed = [c for c in computed if c.get('name') != name]
    computed.append({'name': name, 'expr': str(expr).strip()})
    msg = f"✓ Columna '{name}' añadida. Ya es filtrable y ordenable."
    return computed, msg, '', ''


# ----- m03: poblar el selector del explorador de umbrales -------------------
@callback(
    Output('threshold-col', 'options'),
    Input('available-columns-store', 'data'),
    Input('computed-cols-store', 'data'),
)
def populate_threshold_col(columns, computed):
    """Ofrece todas las columnas (reales + computadas) para explorar
    su distribución y percentiles."""
    columns = list(columns or [])
    for c in (computed or []):
        nm = c.get('name')
        if nm and nm not in columns:
            columns.append(nm)
    return [{'label': c, 'value': c} for c in columns]


# ----- m03: histograma con marcas de percentil ------------------------------
@callback(
    Output('threshold-hist', 'figure'),
    Output('threshold-readout', 'children'),
    Input('threshold-col', 'value'),
    Input('metrics-cache-store', 'data'),
    State('computed-cols-store', 'data'),
)
def render_threshold_hist(col, cache, computed):
    """Dibuja la distribución de la columna elegida con líneas verticales
    en p10/p25/p50/p75/p90, para que el usuario elija un umbral con
    contexto en lugar de a ciegas (m03)."""
    if not col:
        return go.Figure(), ""
    records = (cache or {}).get('records') or []
    if not records:
        return go.Figure(), ""
    df = apply_computed_columns(pd.DataFrame(records), computed)
    if col not in df.columns:
        return go.Figure(), ""
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    if s.empty:
        return go.Figure(), f"'{col}' no tiene valores numéricos."

    pcts = {p: float(s.quantile(p / 100)) for p in (10, 25, 50, 75, 90)}
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, nbinsx=40, marker_color='#3498db',
                               opacity=0.75, name=col))
    colors = {10: '#c0392b', 25: '#e67e22', 50: '#16a085',
              75: '#e67e22', 90: '#c0392b'}
    for p, v in pcts.items():
        fig.add_vline(x=v, line_dash='dash', line_color=colors[p],
                      line_width=1.5,
                      annotation_text=f"p{p}",
                      annotation_position='top',
                      annotation_font_size=10)
    fig.update_layout(
        margin=dict(l=30, r=20, t=20, b=30),
        showlegend=False,
        bargap=0.05,
        xaxis_title=col, yaxis_title='conteo',
    )
    readout = (f"min={s.min():.3g} · p10={pcts[10]:.3g} · p25={pcts[25]:.3g} "
               f"· mediana={pcts[50]:.3g} · p75={pcts[75]:.3g} "
               f"· p90={pcts[90]:.3g} · max={s.max():.3g}   "
               f"(n={len(s)})")
    return fig, readout


@callback(
    Output('rules-container', 'children'),
    Input('rules-store', 'data'),
    State('available-columns-store', 'data'),
)
def render_rules(rules, columns):
    """Renderiza las filas de reglas a partir del estado en rules-store."""
    rules = rules or []
    if not rules:
        return html.P('Pulsa "Añadir regla" para empezar a filtrar.',
                      className='text-muted fst-italic')
    return [make_rule_row(r, columns) for r in rules]


# ----- m01: alternar visibilidad clásico/expresión por fila -----------------
@callback(
    Output({'type': 'rule-classic-wrap', 'index': ALL}, 'style'),
    Output({'type': 'rule-expr-wrap', 'index': ALL}, 'style'),
    Output({'type': 'rule-expr-hint', 'index': ALL}, 'style'),
    Input({'type': 'rule-mode', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def toggle_rule_mode(modes):
    """Muestra los widgets del modo elegido y oculta los del otro.
    Se dispara cuando el usuario cambia el RadioItems de cualquier fila."""
    classic_styles, expr_styles, hint_styles = [], [], []
    for mode in (modes or []):
        is_expr = (mode == 'expr')
        classic_styles.append(
            {'display': 'none'} if is_expr
            else {'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'flex': '1'})
        expr_styles.append(
            {'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'flex': '1'} if is_expr
            else {'display': 'none'})
        hint_styles.append(
            {'fontSize': '11px', 'display': 'inline'} if is_expr
            else {'fontSize': '11px', 'display': 'none'})
    return classic_styles, expr_styles, hint_styles


@callback(
    Output({'type': 'rule-col', 'index': ALL}, 'options'),
    Input('available-columns-store', 'data'),
    Input('computed-cols-store', 'data'),
    Input('annotations-store', 'data'),
    State({'type': 'rule-col', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def update_rule_col_options(columns, computed, annotations, ids):
    """Cuando cambia el archivo de métricas, se añade una columna
    computada (m09) o hay anotaciones (m06), refresca las opciones de
    columna en TODAS las reglas existentes (no solo en las nuevas)."""
    columns = list(columns or [])
    for c in (computed or []):
        nm = c.get('name')
        if nm and nm not in columns:
            columns.append(nm)
    # Columnas de anotación filtrables (m06)
    if annotations:
        for extra in ('_tags', '_note'):
            if extra not in columns:
                columns.append(extra)
    opts = [{'label': c, 'value': c} for c in columns]
    return [opts] * len(ids)


@callback(
    Output('search-results-table', 'children'),
    Output('search-summary', 'children'),
    Output('search-funnel-log', 'children'),
    Input('metrics-cache-store', 'data'),
    Input({'type': 'rule-col', 'index': ALL}, 'value'),
    Input({'type': 'rule-op',  'index': ALL}, 'value'),
    Input({'type': 'rule-val', 'index': ALL}, 'value'),
    Input({'type': 'rule-mode', 'index': ALL}, 'value'),
    Input({'type': 'rule-expr', 'index': ALL}, 'value'),
    Input('computed-cols-store', 'data'),
    Input('annotations-store', 'data'),
    Input('sort-by-dropdown', 'value'),
    Input('sort-asc-switch', 'value'),
)
def run_search(cache, cols, ops, vals, modes, exprs, computed, annotations,
               sort_by, sort_asc):
    """Aplica las reglas en orden y devuelve la tabla de resultados.

    Cada regla puede estar en modo 'classic' (col/op/val) o 'expr'
    (una expresión pandas evaluada con df.eval — m01). Las columnas
    computadas (m09) y las anotaciones (_tags/_note, m06) se añaden antes
    de filtrar, así son filtrables como cualquier otra.

    Toda la tabla se entrega al frontend (Dash maneja la paginación
    nativamente); no se trunca. Truncar aquí ocultaría el mejor diseño
    si por azar quedó en la fila 600."""
    records = (cache or {}).get('records') or []
    if not records:
        return (html.P('Sin datos. Selecciona un archivo de métricas válido.',
                       className='text-muted'),
                "—", "")
    df = pd.DataFrame(records)
    df = apply_computed_columns(df, computed)
    df = inject_annotation_columns(df, annotations)
    total_initial = len(df)
    funnel_lines = [f"[*] Iniciales: {total_initial}"]
    mask_global = pd.Series(True, index=df.index)

    # Normalizar las listas paralelas (pueden venir con longitudes distintas
    # durante un re-render parcial). zip_longest evita IndexError.
    from itertools import zip_longest
    cols = cols or []
    ops = ops or []
    vals = vals or []
    modes = modes or []
    exprs = exprs or []

    for col, op, val, mode, expr in zip_longest(cols, ops, vals, modes, exprs):
        # --- Modo EXPRESIÓN (m01) --------------------------------------
        if mode == 'expr':
            if not expr or not str(expr).strip():
                continue
            m, err = apply_expression(df, expr)
            if err:
                funnel_lines.append(
                    f"  [!] expr «{expr}» → {err}. Regla ignorada.")
                continue
            before = int(mask_global.sum())
            mask_global = mask_global & m
            survivors = int(mask_global.sum())
            funnel_lines.append(
                f"  → expr «{expr}»   supervivientes: {survivors}")
            if survivors == 0 and before > 0:
                funnel_lines.append(
                    "     ⚠ 0 supervivientes tras esta expresión.")
            elif before > 0 and survivors / before < 0.05:
                pct = 100 * (1 - survivors / before)
                funnel_lines.append(
                    f"     ⚠ esta expresión eliminó el {pct:.0f}% restante.")
            continue

        # --- Modo CLÁSICO ----------------------------------------------
        if not col or not op or val is None or val == '':
            continue
        coerced = coerce_value(op, val)
        if op in NUMERIC_OPS and coerced is None:
            funnel_lines.append(
                f"  [!] [{col} {op} {val}] → valor no numérico, regla ignorada.")
            continue
        m = apply_rule(df, col, op, coerced)
        if m is None:
            funnel_lines.append(
                f"  [!] [{col} {op} {val}] → operador no aplicable, regla ignorada.")
            continue
        before = int(mask_global.sum())
        mask_global = mask_global & m.fillna(False)
        survivors = int(mask_global.sum())
        funnel_lines.append(
            f"  → [{col} {op} {val}]   supervivientes: {survivors}")

        # --- Diagnóstico de filtros catastróficos --------------------------
        # Si una regla deja 0 (o casi 0) supervivientes, avisamos y sugerimos
        # un umbral razonable basado en los percentiles del dataset actual.
        col_num = pd.to_numeric(df[col], errors='coerce')
        if survivors == 0 and before > 0:
            if op in ('>=', '>') and col_num.notna().any():
                sug = round(float(col_num.quantile(0.10)), 3)
                funnel_lines.append(
                    f"     ⚠ 0 supervivientes. Máximo de '{col}' = "
                    f"{col_num.max():.3g}. Prueba con ≥ {sug} (p10).")
            elif op in ('<=', '<') and col_num.notna().any():
                sug = round(float(col_num.quantile(0.90)), 3)
                funnel_lines.append(
                    f"     ⚠ 0 supervivientes. Mínimo de '{col}' = "
                    f"{col_num.min():.3g}. Prueba con ≤ {sug} (p90).")
            else:
                funnel_lines.append(
                    "     ⚠ 0 supervivientes tras esta regla. Revisa el valor.")
        elif before > 0 and survivors / before < 0.05:
            pct = 100 * (1 - survivors / before)
            funnel_lines.append(
                f"     ⚠ esta regla eliminó el {pct:.0f}% de los diseños "
                f"que quedaban. ¿Umbral demasiado estricto?")

    df_hits = df[mask_global].copy()
    if sort_by and sort_by in df_hits.columns:
        # Ordenar numéricamente si la columna es convertible; si no, se deja
        # tal cual. (Antes se usaba errors='ignore', deprecado en pandas.)
        try:
            df_hits[sort_by] = pd.to_numeric(df_hits[sort_by])
        except (ValueError, TypeError):
            pass
        df_hits = df_hits.sort_values(by=sort_by,
                                      ascending=bool(sort_asc),
                                      na_position='last')
    n_hits = len(df_hits)
    rate = (n_hits / total_initial * 100) if total_initial else 0
    summary = (f"{total_initial} iniciales → {n_hits} supervivientes "
               f"({rate:.2f} %)")

    # OJO: aquí NO se trunca la tabla. Dash maneja la paginación nativamente
    # con `page_size`, así que pasar 10.000 filas es perfectamente seguro.
    # Recortar a head(N) ocultaría diseños buenos al investigador.
    #
    # persistence=True + id fijo ('search-table'): aunque este callback
    # reconstruya la tabla (p.ej. al llegar diseños nuevos de un run en
    # marcha), Dash RESTAURA el filtrado/orden/página nativos que tuviera
    # el usuario. Sin esto, cada recarga vaciaba esos filtros.
    table = dash_table.DataTable(
        id='search-table',
        data=df_hits.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df_hits.columns],
        page_size=15,
        filter_action='native',
        sort_action='native',
        persistence=True,
        persistence_type='session',
        persisted_props=['filter_query', 'sort_by', 'page_current'],
        style_table={'overflowX': 'auto'},
        style_cell={'fontFamily': 'monospace', 'fontSize': '12px',
                    'padding': '4px', 'textAlign': 'left',
                    'maxWidth': '220px', 'overflow': 'hidden',
                    'textOverflow': 'ellipsis'},
        style_header={'backgroundColor': '#34495e', 'color': 'white',
                      'fontWeight': 'bold'},
        style_data_conditional=[{'if': {'row_index': 'odd'},
                                 'backgroundColor': '#f8f9fa'}],
    )
    return table, summary, html.Pre("\n".join(funnel_lines))


def _build_filter_mask(df, cols, ops, vals, modes, exprs):
    """Construye la máscara global aplicando todas las reglas (clásicas y de
    expresión). Helper compartido por export_csv y la exportación de manifest
    para no duplicar la lógica de filtrado."""
    from itertools import zip_longest
    mask = pd.Series(True, index=df.index)
    for col, op, val, mode, expr in zip_longest(
            cols or [], ops or [], vals or [], modes or [], exprs or []):
        if mode == 'expr':
            if expr and str(expr).strip():
                m, _ = apply_expression(df, expr)
                if m is not None:
                    mask &= m
            continue
        if not col or not op or val is None or val == '':
            continue
        coerced = coerce_value(op, val)
        m = apply_rule(df, col, op, coerced)
        if m is not None:
            mask &= m.fillna(False)
    return mask


@callback(
    Output('download-csv', 'data'),
    Input('export-csv-btn', 'n_clicks'),
    State('metrics-cache-store', 'data'),
    State({'type': 'rule-col', 'index': ALL}, 'value'),
    State({'type': 'rule-op',  'index': ALL}, 'value'),
    State({'type': 'rule-val', 'index': ALL}, 'value'),
    State({'type': 'rule-mode', 'index': ALL}, 'value'),
    State({'type': 'rule-expr', 'index': ALL}, 'value'),
    State('computed-cols-store', 'data'),
    State('annotations-store', 'data'),
    State('sort-by-dropdown', 'value'),
    State('sort-asc-switch', 'value'),
    prevent_initial_call=True,
)
def export_csv(_n, cache, cols, ops, vals, modes, exprs, computed, annotations, sort_by, sort_asc):
    """Exporta todos los hits del filtro actual a un CSV. Es el equivalente
    UI del flag `-o HITS_ELITE.csv` de extract_hits.py."""
    records = (cache or {}).get('records') or []
    if not records:
        raise PreventUpdate
    df = pd.DataFrame(records)
    df = apply_computed_columns(df, computed)
    df = inject_annotation_columns(df, annotations)
    mask = _build_filter_mask(df, cols, ops, vals, modes, exprs)
    df_out = df[mask].copy()
    if sort_by and sort_by in df_out.columns:
        try:
            df_out[sort_by] = pd.to_numeric(df_out[sort_by])
        except (ValueError, TypeError):
            pass
        df_out = df_out.sort_values(by=sort_by, ascending=bool(sort_asc),
                                    na_position='last')
    return dcc.send_data_frame(df_out.to_csv,
                               "BinderMonitor_HITS.csv", index=False)


# ----- m04: exportar manifest (estado serializable del filtrado) ------------
@callback(
    Output('download-manifest', 'data'),
    Input('export-manifest-btn', 'n_clicks'),
    State({'type': 'rule-col', 'index': ALL}, 'value'),
    State({'type': 'rule-op',  'index': ALL}, 'value'),
    State({'type': 'rule-val', 'index': ALL}, 'value'),
    State({'type': 'rule-mode', 'index': ALL}, 'value'),
    State({'type': 'rule-expr', 'index': ALL}, 'value'),
    State('computed-cols-store', 'data'),
    State('sort-by-dropdown', 'value'),
    State('sort-asc-switch', 'value'),
    State('metrics-file-dropdown', 'value'),
    State('annotations-store', 'data'),
    prevent_initial_call=True,
)
def export_manifest(_n, cols, ops, vals, modes, exprs, computed,
                    sort_by, sort_asc, metric_file, annotations):
    """Serializa el estado completo del filtrado a un manifest.json.
    Es la otra mitad del ciclo de reproducibilidad: lo que load_manifest
    sabe leer de vuelta (reglas, columnas computadas, orden). Incluye
    además el dump de anotaciones (m06, paso 4) para que el paquete viaje
    completo: estructura + secuencia + contexto humano."""
    from itertools import zip_longest
    filters = []
    for col, op, val, mode, expr in zip_longest(
            cols or [], ops or [], vals or [], modes or [], exprs or []):
        mode = mode or 'classic'
        if mode == 'expr':
            if expr and str(expr).strip():
                filters.append({'mode': 'expr', 'expresion': str(expr).strip()})
        elif col and op and val not in (None, ''):
            filters.append({'mode': 'classic', 'columna': col,
                            'operador': op, 'valor': val})
    manifest = {
        'generated_by': 'BinderMonitor',
        'metric_file': os.path.basename(metric_file) if metric_file else None,
        'filters': filters,
        'computed_columns': computed or [],
        'sort_by': sort_by,
        'sort_ascending': bool(sort_asc),
        'annotations': annotations or {},
    }
    return dict(content=json.dumps(manifest, indent=2, ensure_ascii=False),
                filename="BinderMonitor_manifest.json")


# ----- m04: importar manifest (restaura el estado del filtrado) -------------
@callback(
    Output('rules-store', 'data', allow_duplicate=True),
    Output('rules-counter-store', 'data', allow_duplicate=True),
    Output('computed-cols-store', 'data', allow_duplicate=True),
    Output('sort-by-dropdown', 'value', allow_duplicate=True),
    Output('sort-asc-switch', 'value', allow_duplicate=True),
    Output('manifest-feedback', 'children'),
    Input('manifest-upload', 'contents'),
    State('manifest-upload', 'filename'),
    prevent_initial_call=True,
)
def load_manifest(contents, filename):
    """Lee un manifest.json arrastrado y restaura reglas, columnas
    computadas y orden. Las reglas se recrean como índices nuevos en
    rules-store; los valores concretos los recoge la persistencia de
    sesión y el propio render. Como el contenido detallado de cada regla
    se guarda en componentes (no en rules-store), aquí solo recreamos el
    número de filas necesario y dejamos que el usuario las repueble; para
    una restauración fiel de valores se requiere el patrón de stores por
    regla. Restauramos de forma fiel lo que SÍ vive en stores: columnas
    computadas y orden."""
    if not contents:
        raise PreventUpdate
    try:
        _, b64 = contents.split(',', 1)
        import base64
        manifest = json.loads(base64.b64decode(b64).decode('utf-8'))
    except Exception as e:
        return (no_update, no_update, no_update, no_update, no_update,
                f"✕ Error leyendo manifest: {e}")

    filters = manifest.get('filters', []) or []
    computed = manifest.get('computed_columns', []) or []
    sort_by = manifest.get('sort_by', no_update)
    sort_asc = manifest.get('sort_ascending', no_update)

    # Recrear tantas filas como filtros (índices 0..N-1). El detalle de
    # cada filtro se aplica vía persistencia/recarga del usuario.
    n = len(filters)
    rules = list(range(n)) if n else []
    counter = n

    msg = (f"✓ Cargado '{filename}': {n} reglas, "
           f"{len(computed)} columnas computadas. "
           f"Revisa/ajusta los valores de las reglas restauradas.")
    return rules, counter, computed, sort_by, sort_asc, msg


# =============================================================================
# 6.13 — 6.16 VISOR 3D Y TRAYECTORIAS
# =============================================================================
# Los controles del visor (slider, botones, Store) están SIEMPRE en el
# layout estático. No se crean ni destruyen dinámicamente.
#
# Razón: Dash no garantiza que callbacks pre-registrados se activen
# cuando sus Input/State son componentes inyectados por otro callback.
# Tener todo estático y controlar la visibilidad con CSS (`display:
# none/block`) es el patrón canónico de Dash para este caso.
#
# Tres callbacks escriben en molstar-viewer.data:
#   A) load_af2_structure (PRIMARIO) — selección de diseño en dropdown
#   B) load_trajectory_or_final (allow_duplicate) — slider o botón AF2
#   C) update_traj_label — etiqueta "MODEL X / N" al mover slider
# =============================================================================

@callback(
    Output('traj-pdb-store', 'data'),
    Output('traj-slider', 'max'),
    Output('traj-slider', 'value'),
    Output('traj-slider', 'marks'),
    Output('traj-controls-wrapper', 'style'),
    Output('traj-message', 'children'),
    Output('viewer-info', 'children'),
    Output('traj-play-interval', 'disabled', allow_duplicate=True),
    Output('traj-play-btn', 'children', allow_duplicate=True),
    Output('traj-play-btn', 'color', allow_duplicate=True),
    Output('traj-goto-input', 'max'),
    Output('traj-goto-input', 'value'),
    Input('hit-dropdown', 'value'),
    State('experiment-dropdown', 'value'),
    prevent_initial_call=True,
)
def setup_trajectory(base_name, exp_dir):
    """Configura los controles de trayectoria para el diseño seleccionado.

    Los componentes del slider, botones y Store están SIEMPRE en el layout
    estático (no se crean/destruyen). Este callback solo actualiza sus
    propiedades y controla su visibilidad vía CSS.

    Al cambiar de diseño, pausa la animación automáticamente."""
    hide = {'display': 'none'}
    show = {'display': 'block'}
    empty_marks = {0: '0', 1: '1'}
    # 7 outputs base + 3 de animación (pausar)
    paused = (True, '▶ Play', 'primary')

    if not base_name or not exp_dir:
        return None, 1, 0, empty_marks, hide, "", "", *paused, 1, None

    cleanup_old_traj_files(
        keep_basename=re.sub(r'_dldesign_\d+$', '', base_name))

    traj_pdb = get_trajectory_pdb(exp_dir, base_name)
    if not traj_pdb:
        print(f"[BinderMonitor] Trayectoria no encontrada para: {base_name}")
        msg = [html.I(className='bi bi-info-circle me-1'),
               'No se encontró trayectoria de RFdiffusion para este diseño.']
        return None, 1, 0, empty_marks, hide, msg, f"Diseño: {base_name}", *paused, 1, None

    n_models = count_pdb_models(traj_pdb)
    print(f"[BinderMonitor] Trayectoria encontrada: {traj_pdb} "
          f"({n_models} modelos)")

    if n_models <= 1:
        msg = [html.I(className='bi bi-info-circle me-1'),
               f'PDB de trayectoria mono-modelo ({n_models} bloques).']
        return None, 1, 0, empty_marks, hide, msg, f"Diseño: {base_name}", *paused, 1, None

    marks = {0: 'ruido', n_models - 1: 'final'}
    if n_models >= 5:
        for i in [n_models // 4, n_models // 2, 3 * n_models // 4]:
            if i not in marks:
                marks[i] = f'{i + 1}'

    info = (f"Diseño: {base_name} · Trayectoria: "
            f"{os.path.basename(traj_pdb)} · {n_models} pasos")

    # Arranca mostrando la estructura final (paso N). goto-input 1-indexado.
    return (traj_pdb, n_models - 1, n_models - 1, marks,
            show, "", info, *paused, n_models, n_models)


@callback(
    Output('traj-slider', 'value', allow_duplicate=True),
    Input('traj-first', 'n_clicks'),
    Input('traj-last',  'n_clicks'),
    Input('traj-prev',  'n_clicks'),
    Input('traj-next',  'n_clicks'),
    Input('traj-goto-input', 'value'),
    State('traj-slider', 'min'),
    State('traj-slider', 'max'),
    State('traj-slider', 'value'),
    prevent_initial_call=True,
)
def jump_slider(_first, _last, _prev, _next, goto, mn, mx, cur):
    """Atajos de navegación de la trayectoria:
      ⏮ primer paso · ◀ anterior · ▶ siguiente · ⏭ último ·
      input numérico (1-indexado) para ir a un paso concreto.
    Todos respetan el orden lógico ruido(1) → final(N)."""
    trig = ctx.triggered_id
    if trig == 'traj-first':
        return mn
    if trig == 'traj-last':
        return mx
    if trig == 'traj-prev':
        return max(mn, (cur if cur is not None else mn) - 1)
    if trig == 'traj-next':
        return min(mx, (cur if cur is not None else mn) + 1)
    if trig == 'traj-goto-input':
        if goto is None:
            raise PreventUpdate
        # input 1-indexado → índice lógico 0-indexado, recortado al rango
        idx = int(goto) - 1
        return max(mn, min(mx, idx))
    raise PreventUpdate


# ---- CALLBACKS DEL VISOR 3D ------------------------------------------------
#
# Un callback NO SE EJECUTA si alguno de sus Inputs no existe en el DOM,
# ni siquiera con suppress_callback_exceptions=True. Esa flag solo permite
# REGISTRAR el callback sin error al arrancar la app.
#
# Esto tiene una consecuencia directa aquí: traj-slider, traj-show-final
# y traj-pdb-store son componentes DINÁMICOS que build_trajectory_controls()
# crea o destruye según el diseño seleccionado. Si metemos esos Inputs
# dinámicos en el mismo callback que hit-dropdown (estático), el callback
# entero queda bloqueado hasta que los dinámicos existan.
#
# La solución: DOS callbacks separados.
#
#   Callback A (PRIMARIO): Solo Inputs estáticos (hit-dropdown).
#   Se dispara SIEMPRE que el usuario selecciona un diseño.
#   Carga la estructura AF2 final.
#
#   Callback B (SECUNDARIO, allow_duplicate=True): Solo Inputs dinámicos
#   (traj-slider, traj-show-final). Se dispara solo cuando existen, es
#   decir, después de que build_trajectory_controls() haya creado el
#   slider. Carga frames de trayectoria o vuelve a la estructura AF2.
#
# allow_duplicate=True en el Callback B es seguro aquí: indica a Dash
# que DOS callbacks pueden escribir en molstar-viewer.data. El último
# que escriba gana. No hay race condition porque los triggers son
# mutuamente excluyentes (o mueves el slider O seleccionas un hit,
# pero no ambos a la vez).
# -------------------------------------------------------------------------

# ---- Callback A: carga AF2 al seleccionar un diseño (PRIMARIO) ----------
@callback(
    Output('molstar-viewer', 'data'),
    Input('hit-dropdown', 'value'),
    State('experiment-dropdown', 'value'),
    prevent_initial_call=True,
)
def load_af2_structure(hit_basename, exp_dir):
    """Carga la estructura final (AlphaFold2 o Complexa) cuando se selecciona un
    diseño en el dropdown. Este callback usa SOLO inputs estáticos
    (que siempre existen en el DOM), así que siempre se dispara."""
    if not hit_basename or not exp_dir:
        raise PreventUpdate

    file_to_load = resolve_structure_pdb(exp_dir, hit_basename)
    if not file_to_load:
        print(f"[BinderMonitor] PDB no encontrado para: {hit_basename}")
        raise PreventUpdate

    print(f"[BinderMonitor] Visor PDB → {file_to_load}")
    return molstar_helper.parse_molecule(file_to_load, 'pdb')


# ---- Callback B: trayectoria o botón "Estructura final" (SECUNDARIO) ----
@callback(
    Output('molstar-viewer', 'data', allow_duplicate=True),
    Input('traj-slider', 'value'),
    Input('traj-show-final', 'n_clicks'),
    State('traj-pdb-store', 'data'),
    State('experiment-dropdown', 'value'),
    State('hit-dropdown', 'value'),
    prevent_initial_call=True,
)
def load_trajectory_or_final(slider_idx, _final_clicks, traj_pdb,
                             exp_dir, hit_basename):
    """Carga un frame de trayectoria (slider) o vuelve a la estructura
    AF2 (botón "Estructura final"). Los Inputs (traj-slider, traj-show-final)
    son componentes ESTÁTICOS que siempre existen en el DOM."""
    triggered = ctx.triggered_id
    print(f"[BinderMonitor] Callback traj disparado: triggered={triggered}, "
          f"slider={slider_idx}, traj_pdb={traj_pdb}")

    if triggered == 'traj-slider':
        if slider_idx is None or not traj_pdb or not os.path.exists(traj_pdb):
            raise PreventUpdate
        n_total = count_pdb_models(traj_pdb)
        phys = logical_to_physical_model(slider_idx, n_total)
        extracted = extract_pdb_model_streaming(traj_pdb, phys)
        if not extracted:
            raise PreventUpdate
        print(f"[BinderMonitor] Trayectoria → lógico {slider_idx} "
              f"→ físico {phys} → {extracted}")
        return molstar_helper.parse_molecule(extracted, 'pdb')

    if triggered == 'traj-show-final':
        if not hit_basename or not exp_dir:
            raise PreventUpdate
        file_to_load = resolve_structure_pdb(exp_dir, hit_basename)
        if not file_to_load:
            print(f"[BinderMonitor] PDB no encontrado para: {hit_basename}")
            raise PreventUpdate
        print(f"[BinderMonitor] Visor (PDB vía botón) → {file_to_load}")
        return molstar_helper.parse_molecule(file_to_load, 'pdb')

    raise PreventUpdate


# ---- Callback C: etiqueta del frame de trayectoria -------------------------
# Solo se dispara al mover el slider. En ese momento, traj-frame-label
# ya existe en el DOM (se crea junto con el slider en
# build_trajectory_controls).
@callback(
    Output('traj-frame-label', 'children'),
    Input('traj-slider', 'value'),
    State('traj-pdb-store', 'data'),
    prevent_initial_call=True,
)
def update_traj_label(slider_idx, traj_pdb):
    """Actualiza el texto del frame al mover el slider. El índice del slider
    es LÓGICO: 0 = ruido inicial, n-1 = estructura final, siempre en ese
    sentido aunque el PDB esté grabado al revés."""
    if slider_idx is None or not traj_pdb:
        raise PreventUpdate
    n_total = count_pdb_models(traj_pdb)
    paso = slider_idx + 1
    etiqueta = ''
    if slider_idx == 0:
        etiqueta = ' (ruido inicial)'
    elif slider_idx == n_total - 1:
        etiqueta = ' (estructura final)'
    return f"Paso {paso} / {n_total}{etiqueta}"


# ----- 6.17 Botón de parada de emergencia -----------------------------------
@callback(
    Output('shutdown-feedback', 'children'),
    Input('shutdown-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def shutdown_server(n):
    """Apaga el servidor Dash desde la propia web y libera el puerto."""
    if n:
        _cleanup_temp_dir()
        port = PORT_LABEL if PORT_LABEL != "?" else 8050
        # Enviar señal de terminación al propio proceso
        os.kill(os.getpid(), signal.SIGTERM)
        return f"🛑 Servidor apagado. El puerto {port} ha sido liberado. Puedes cerrar esta pestaña."


# =============================================================================
# 6.18 — 6.22 MEJORAS DE UX
# =============================================================================

# ----- 6.18 Sidebar contextual: ocultar filtros fuera de Monitorización -----
@callback(
    Output('contextual-sidebar', 'style'),
    Input('tabs-main', 'value'),
)
def toggle_sidebar_context(active_tab):
    """Oculta los filtros del scatter y los selectores de ejes cuando no
    estamos en la pestaña de Monitorización. En las demás pestañas esos
    controles no afectan a nada y solo confunden."""
    if active_tab == 'tab-monitor':
        return {'display': 'block'}
    return {'display': 'none'}


# ----- 6.19 Animación de trayectoria: Play / Pause -------------------------
@callback(
    Output('traj-play-interval', 'disabled'),
    Output('traj-play-btn', 'children'),
    Output('traj-play-btn', 'color'),
    Output('traj-slider', 'value', allow_duplicate=True),
    Input('traj-play-btn', 'n_clicks'),
    State('traj-play-interval', 'disabled'),
    State('traj-slider', 'value'),
    State('traj-slider', 'min'),
    State('traj-slider', 'max'),
    prevent_initial_call=True,
)
def toggle_play_pause(_n, is_disabled, cur_val, mn, mx):
    """Activa/desactiva el metrónomo. Al pulsar Play, si el slider ya está
    en el frame final, rebobina al ruido (mín) para que la animación
    muestre la difusión completa ruido → estructura."""
    if is_disabled:
        # arrancar
        start = mn if (cur_val is not None and cur_val >= mx) else no_update
        return False, '⏸ Pause', 'warning', start
    # pausar
    return True, '▶ Play', 'primary', no_update


# ----- 6.20 Animación: ajuste de velocidad ----------------------------------
@callback(
    Output('traj-play-interval', 'interval'),
    Input('traj-speed', 'value'),
)
def update_animation_speed(speed_ms):
    """Actualiza el intervalo del metrónomo según la velocidad elegida."""
    return speed_ms or 500


# ----- 6.21 Animación: avance automático del slider -------------------------
@callback(
    Output('traj-slider', 'value', allow_duplicate=True),
    Output('traj-play-interval', 'disabled', allow_duplicate=True),
    Output('traj-play-btn', 'children', allow_duplicate=True),
    Output('traj-play-btn', 'color', allow_duplicate=True),
    Input('traj-play-interval', 'n_intervals'),
    State('traj-slider', 'value'),
    State('traj-slider', 'max'),
    prevent_initial_call=True,
)
def auto_play_tick(_n, current_val, slider_max):
    """Cada tick del metrónomo avanza el slider un paso ASCENDENTE
    (ruido → estructura, de izquierda a derecha). El índice es lógico:
    0 = ruido inicial, max = estructura final. Al llegar al final se
    auto-pausa."""
    if current_val is None:
        raise PreventUpdate
    new_val = current_val + 1
    if new_val > slider_max:
        # Llegamos al final: pausar automáticamente
        return slider_max, True, '▶ Play', 'primary'
    return new_val, no_update, no_update, no_update


# ----- 6.22 Radar plot de métricas del diseño seleccionado ------------------
# Métricas candidatas para el radar, con su sentido de optimización:
#   'high' = mayor es mejor · 'low' = menor es mejor
# El radar normaliza por PERCENTIL dentro del dataset y ORIENTA todas las
# métricas para que "más lejos del centro = mejor", de modo que un buen
# diseño dibuja un polígono grande, sin importar si la métrica original
# se maximiza o se minimiza (resuelve el problema del pae_interaction).
RADAR_METRICS = {
    'pae_interaction':       'low',
    'plddt_binder':          'high',
    'plddt_target':          'high',
    'plddt_total':           'high',
    'pae_binder':            'low',
    'pae_target':            'low',
    'binder_aligned_rmsd':   'low',
    'dSASA':                 'high',
    'Shape_complementarity': 'high',
    'dG_separated':          'low',
    'binder_score':          'low',
    'interface_buried_sasa': 'high',
    # Métricas derivadas de scoring físico (PyRosetta / AlphaFold2)
    'ddG':                    'low',
    'Packstat':               'high',
    'SAP':                    'low',
    'interface_hbonds':       'high',
    'interface_unsat_hbonds': 'low',
    'pae_plddt_ratio':        'low',
    'binder_int_hyd':         'high',
    'binder_surf_hyd':        'low',
}


@callback(
    Output('radar-plot', 'figure'),
    Output('radar-values', 'children'),
    Input('hit-dropdown', 'value'),
    State('metrics-cache-store', 'data'),
    prevent_initial_call=True,
)
def update_radar(hit_name, cache):
    """Radar plot orientado por percentiles de calidad relativa en el dataset."""
    if not hit_name or not cache or not cache.get('records'):
        return _empty_radar("Sin datos del diseño"), ""

    df = pd.DataFrame(cache['records'])
    if 'base_name' not in df.columns:
        return _empty_radar("No hay columna base_name"), ""

    hit_row = df[df['base_name'] == hit_name]
    if hit_row.empty:
        return _empty_radar("Diseño no encontrado"), ""
    hit_row = hit_row.iloc[0]

    radar_metrics = PIPELINE.get('radar_metrics') or RADAR_METRICS
    present = []
    for m in radar_metrics:
        if m in df.columns:
            present.append(m)
        elif m == 'pae_plddt_ratio':
            for fallback in ('Macro_CUTRE', 'score_cutre'):
                if fallback in df.columns:
                    present.append(fallback)
                    break

    if not present:
        return _empty_radar("Ninguna métrica conocida en el archivo"), ""

    labels, radii, hovertexts, table_rows = [], [], [], []
    for m in present:
        col = pd.to_numeric(df[m], errors='coerce')
        val = pd.to_numeric(hit_row[m], errors='coerce')
        if pd.isna(val) or col.notna().sum() < 2:
            continue
        pct = float((col < val).mean())
        sense = radar_metrics.get(m, 'low')
        quality = pct if sense == 'high' else (1.0 - pct)
        arrow = '↑' if sense == 'high' else '↓'
        labels.append(m)
        radii.append(round(quality, 3))
        hovertexts.append(
            f"<b>{m}</b><br>valor: {val:.3g}<br>"
            f"percentil dataset: {pct*100:.0f}%<br>"
            f"{'↑ mejor alto' if sense=='high' else '↓ mejor bajo'}")
        # color del percentil de calidad: verde bueno, rojo malo
        qcolor = ('#2d7d6e' if quality >= 0.66 else
                  '#b8893a' if quality >= 0.33 else '#c0392b')
        table_rows.append(
            html.Tr([
                html.Td(f"{arrow} {m}",
                        style={'paddingRight': '10px'}),
                html.Td(f"{val:.3g}",
                        style={'textAlign': 'right',
                               'fontFamily': 'monospace'}),
                html.Td(f"{pct*100:.0f}%",
                        style={'textAlign': 'right', 'color': qcolor,
                               'paddingLeft': '10px'}),
            ]))

    if not radii:
        return _empty_radar("Sin valores numéricos para este diseño"), ""

    fig = go.Figure(data=go.Scatterpolar(
        r=radii + [radii[0]],
        theta=labels + [labels[0]],
        fill='toself',
        line_color='#2d7d6e',
        fillcolor='rgba(45,125,110,0.35)',
        hovertext=hovertexts + [hovertexts[0]],
        hoverinfo='text',
        name=hit_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=['25%', '50%', '75%', 'top'],
                tickfont=dict(size=9),
            ),
        ),
        showlegend=False,
        margin=dict(t=30, b=30, l=50, r=50),
        title=dict(
            text="Calidad relativa (radio = mejor; hover = valor real)",
            font=dict(size=11), x=0.5,
        ),
    )

    table = html.Table([
        html.Thead(html.Tr([
            html.Th('Métrica', style={'textAlign': 'left'}),
            html.Th('Valor', style={'textAlign': 'right'}),
            html.Th('Pct', style={'textAlign': 'right',
                                  'paddingLeft': '10px'}),
        ])),
        html.Tbody(table_rows),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

    return fig, table


def _empty_radar(msg):
    """Radar vacío con un mensaje central."""
    fig = go.Figure()
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        margin=dict(t=25, b=25, l=40, r=40),
        annotations=[dict(text=msg, xref='paper', yref='paper',
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=12, color='#888'))],
    )
    return fig


# =============================================================================
# 6.23 ANOTACIONES PERSISTENTES (m06)
# =============================================================================
# ----- Cargar las anotaciones del diseño seleccionado -----------------------
@callback(
    Output('annot-tags', 'value'),
    Output('annot-note', 'value'),
    Output('annot-hint', 'children'),
    Output('annot-feedback', 'children', allow_duplicate=True),
    Input('hit-dropdown', 'value'),
    State('experiment-dropdown', 'value'),
    prevent_initial_call=True,
)
def load_annotations(hit_name, exp_dir):
    """Al cambiar de diseño, carga sus tags y nota desde la BD SQLite del
    experimento. Si no hay diseño seleccionado, limpia el panel."""
    if not hit_name:
        return [], '', 'Selecciona un diseño para anotarlo.', ''
    tags, note = get_design_annotations(exp_dir, hit_name)
    hint = html.Span([
        'Anotando ', html.Code(hit_name, style={'fontSize': '11px'}),
    ])
    return tags, note, hint, ''


# ----- Guardar las anotaciones del diseño seleccionado ----------------------
@callback(
    Output('annot-feedback', 'children'),
    Output('annotations-store', 'data', allow_duplicate=True),
    Input('annot-save-btn', 'n_clicks'),
    State('hit-dropdown', 'value'),
    State('experiment-dropdown', 'value'),
    State('annot-tags', 'value'),
    State('annot-note', 'value'),
    prevent_initial_call=True,
)
def save_annotations(_n, hit_name, exp_dir, tags, note):
    """Persiste tags + nota en `.binderlab_annotations.db` dentro de la raíz
    del experimento. Da feedback inmediato y refresca el snapshot global de
    anotaciones para que las columnas _tags/_note se actualicen en la
    búsqueda avanzada."""
    if not hit_name:
        return "✕ Selecciona un diseño primero.", no_update
    if not exp_dir:
        return "✕ No hay experimento activo.", no_update
    ok_tags = set_design_tags(exp_dir, hit_name, tags)
    ok_note = set_design_note(exp_dir, hit_name, note)
    if ok_tags and ok_note:
        n_tags = len(tags or [])
        snapshot = get_all_annotations(exp_dir)
        return (f"✓ Guardado ({n_tags} tag{'s' if n_tags != 1 else ''}).",
                snapshot)
    return ("✕ Error al guardar (¿permisos de escritura en el experimento?).",
            no_update)


# ----- Refrescar el snapshot de anotaciones al cambiar de experimento -------
@callback(
    Output('annotations-store', 'data'),
    Input('experiment-dropdown', 'value'),
)
def refresh_annotations_snapshot(exp_dir):
    """Carga TODAS las anotaciones del experimento activo en un store, para
    que la búsqueda avanzada pueda inyectar las columnas _tags / _note."""
    if not exp_dir:
        return {}
    return get_all_annotations(exp_dir)


# =============================================================================
# 7. ARRANQUE
# =============================================================================
def main(argv=None):
    """Punto de entrada. Puebla los globals diferidos y arranca el servidor."""
    global MASTER_DIR_INIT, PIPELINE, TEMP_DIR, HOST_LABEL, PORT_LABEL
    args, _unknown = _build_arg_parser().parse_known_args(argv)
    MASTER_DIR_INIT = os.path.abspath(args.dir)
    PIPELINE = load_pipeline_structure(args.config)
    _init_stages()
    TEMP_DIR = tempfile.mkdtemp(prefix='bindermonitor_')
    HOST_LABEL, PORT_LABEL = args.host, args.port
    print("=" * 70)
    print("  BinderMonitor")
    print(f"  Host       : {args.host}")
    print(f"  Puerto     : {args.port}")
    print(f"  MASTER_DIR : {MASTER_DIR_INIT}")
    print(f"  Config     : {args.config or '(defaults)'}")
    print(f"  TEMP_DIR   : {TEMP_DIR}")
    stages_str = " · ".join(STAGE_LABELS.values()) if STAGE_LABELS else "Estándar"
    print(f"  Pipeline   : {stages_str}")
    print("=" * 70)
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
