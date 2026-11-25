"""Configuracion del pipeline y objeto Settings.

`Settings` agrupa lo que antes eran variables globales sueltas de monitor.py
(PIPELINE, TEMP_DIR, MASTER_DIR_INIT, host/puerto y las constantes de etapa).
Tenerlo en un objeto explicito permite que los modulos de core/ reciban su
configuracion por argumento en vez de leerla "del aire", que es lo que hace
falta para poder repartir el monolito en varios ficheros.
"""
import copy
import os
import json
from dataclasses import dataclass, field


DEFAULT_PIPELINE_STRUCTURE = {
    # --- Carpetas de salida de cada etapa del pipeline -----------------------
    "rfdiffusion_dir":  "RF_output",    # backbones generados por RFdiffusion
    "proteinmpnn_dir":  "MPNN_output",  # secuencias de ProteinMPNN
    "alphafold_dir":    "AF_output",    # predicciones de AlphaFold2

    # modelo N-etapas declarativo. None = usa RF/MPNN/AF (3 fijas).
    # Preset por -c JSON: lista de {key,label,dir,pdb_glob?,color?}. pdb_glob
    # relativo a dir cuenta PDBs por glob (p.ej. Complexa: "job_*/*_updated.pdb"
    # con dir="."). Ceiling: el parser de log en vivo sigue siendo RF; etapas no
    # RF cuentan por disco (suficiente para lotes terminados como Complexa).
    "stages": None,

    # --- Subcarpeta donde RFdiffusion guarda los PDBs de trayectoria --------
    # (relativa a rfdiffusion_dir; ej: RF_output/traj/...)
    "trajectory_subdir": "traj",

    # --- Subcarpetas de filtrado (passed/discarded) -------------------------
    # El monitor las detecta automáticamente. Si no existen, cuenta los
    # PDBs directamente en la raíz de la etapa.
    "passed_subdir":    "passed",
    "discarded_subdir": "discarded",

    # --- Sufijos de los nombres de archivo ----------------------------------
    # AlphaFold2 añade "_af2pred" al PDB final. Si lo cambian, ajustar aquí.
    "af_suffix":         "_af2pred",
    # RFdiffusion guarda la trayectoria como "<diseño>_pX0_traj.pdb"
    "trajectory_suffix": "_pX0_traj",

    # --- Búsqueda de logs (versátil para distintos labs/pipelines) ----------
    # Glob patterns de archivos de log a buscar, EN ORDEN DE PREFERENCIA.
    # Se busca recursivamente desde el experimento hacia arriba (hasta
    # `log_search_levels` niveles) y dentro del experimento (recursivo).
    # Cubre: .log de RFdiffusion (run_inference.log enterrado en
    # outputs/FECHA/HORA/), wrappers RFrun_*.log, y .out/.err de SLURM.
    "log_patterns": ["*.log", "*.out", "*.err"],
    # Cuántos niveles de directorio padre subir buscando logs (0 = solo el
    # experimento). 2 cubre: experimento → carpeta múltiple → raíz campañas.
    "log_search_levels": 2,

    # --- Orden de la trayectoria de RFdiffusion -----------------------------
    # En algunos formatos el primer MODEL del PDB es el paso final de la
    # difusión y el último MODEL es el ruido inicial (orden invertido).
    # Si True, el frame lógico 0 = "ruido" se mapea al ÚLTIMO MODEL físico,
    # de modo que el slider va siempre ruido → estructura de izq. a der.
    "trajectory_reversed": True,

    # globs que tambien marcan un experimento, para pipelines sin
    # carpetas RF/MPNN/AF (p.ej. Proteina-Complexa, que emite binder_results_*.csv).
    # Vacio por defecto = sin cambios para labs existentes. Se activa via -c JSON.
    "extra_markers": [],

    # --- Parseo de estado del flujo (tarjetas de etapa) ---------------------
    # El log del orquestador (SCRIPT_universal.sh) marca cada etapa con
    # "[N/M] ...". El monitor lee estas marcas para saber qué etapa está
    # activa, y cuenta líneas de progreso de cada programa.
    # Si tu lab usa otro orquestador, reescribe estos patrones por -c JSON.
    "log_stage_markers": {
        # Marca de inicio de cada etapa en el log del orquestador.
        # group(1) = número de iteración/run (opcional, informativo).
        "rfdiffusion": r"\[\d+/\d+\]\s.*RFdiffusion(?:\s*\(Run\s*(\d+)\))?",
        "proteinmpnn": r"\[\d+/\d+\]\s.*ProteinMPNN(?:\s*\(Run\s*(\d+)\))?",
        "alphafold":   r"\[\d+/\d+\]\s.*AlphaFold2(?:\s*\(Run\s*(\d+)\))?",
        # Post-proceso: implica que las 3 etapas anteriores acabaron.
        "postprocess": r"\[\d+/\d+\]\s.*(?:estad[íi]sticas|Rosetta|termodin)",
    },
    # Marca de experimento dentro de una campaña múltiple. group(1)=actual,
    # group(2)=total. Permite mostrar "Experimento 2 de 3".
    "log_experiment_marker": r"INICIANDO EXPERIMENTO\s+(\d+)\s+DE\s+(\d+)",
    # Marca de finalización global de la campaña.
    "log_done_marker": r"FINALIZADO EXITOSAMENTE",
    # Patrones de PROGRESO por programa: cada línea que casa cuenta como
    # "una unidad procesada".  group(1), si existe, es el índice del diseño
    # (sirve para estimar el total como max(idx)+1 cuando no se conoce).
    "log_progress_markers": {
        "rfdiffusion": r"Making design\b.*?design_(\d+)",
        "proteinmpnn": r"ProteinMPNN generated\s+\d+\s+sequences",
        "alphafold":   r"finished AF2 prediction",
    },
    # Sub-progreso de RFdiffusion dentro de un diseño: "Timestep T" cuenta
    # T → 1 (difusión inversa).  Informativo para el spinner fino.
    "log_rf_timestep_marker": r"Timestep\s+(\d+)",
    # Nº total de timesteps de difusión (T inicial). RFdiffusion arranca cada
    # diseño en este valor y baja hasta 1. Permite la mini-barra del diseño
    # actual. Por defecto 50 (config estándar del modelo Complex_base).
    "rf_timesteps_total": 50,
    # Tiempo POR UNIDAD que reporta cada programa en su propia línea de
    # progreso (segundos). group(1) = nº de segundos. RFdiffusion no lo
    # reporta directamente; su ritmo se deriva de timestamps ISO de líneas
    # "Making design" consecutivas.
    "log_unit_time_markers": {
        "proteinmpnn": r"generated\s+\d+\s+sequences\s+in\s+([\d.]+)\s+seconds",
        "alphafold":   r"finished AF2 prediction\s+in\s+([\d.]+)\s+seconds",
    },
    # Timestamp ISO al inicio de línea: [YYYY-MM-DD HH:MM:SS,mmm]. Se usa para
    # derivar el ritmo de RFdiffusion (diferencia entre diseños consecutivos)
    # y el tiempo transcurrido en la etapa activa.
    "log_timestamp_marker": r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]",
}


def load_pipeline_structure(config_path):
    """Carga la estructura del pipeline desde un JSON opcional.

    Si el archivo existe, sus claves SOBRESCRIBEN las defaults; las que
    no estén en el JSON quedan con el valor por defecto. Esto permite
    pasar un JSON parcial (p.ej. solo con "alphafold_dir") sin tener que
    repetir todas las claves.
    """
    structure = copy.deepcopy(DEFAULT_PIPELINE_STRUCTURE)
    if not config_path:
        return structure
    # Empaquetado: si no es una ruta existente, probar como nombre de preset
    # incluido en el paquete. Acepta -c complexa y -c complexa.json.
    if not os.path.exists(config_path):
        stem = config_path[:-5] if config_path.endswith('.json') else config_path
        raiz_paquete = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset = os.path.join(raiz_paquete, "presets", f"{stem}.json")
        if os.path.exists(preset):
            config_path = preset
    if not os.path.exists(config_path):
        print(f"[BinderMonitor] AVISO: config no encontrada ({config_path}). "
              f"Se usan valores por defecto.")
        return structure
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        structure.update(user_config)
        print(f"[BinderMonitor] Configuración cargada desde {config_path}")
    except Exception as e:
        print(f"[BinderMonitor] AVISO: error leyendo {config_path}: {e}. "
              f"Se usan valores por defecto.")
    return structure


def build_stages(pipeline):
    """Deriva las constantes de etapa desde el dict de configuracion.

    Devuelve (stages, keys, labels, dirs, globs, colors). Si el preset no
    define `stages`, cae al esquema clasico de 3 etapas RF/MPNN/AF.
    """
    stages = pipeline.get("stages") or [
        {"key": "rfdiffusion", "label": "RFdiffusion", "dir": pipeline["rfdiffusion_dir"], "color": "#2E86C1"},
        {"key": "proteinmpnn", "label": "ProteinMPNN", "dir": pipeline["proteinmpnn_dir"], "color": "#1D9E75"},
        {"key": "alphafold",   "label": "AlphaFold2",  "dir": pipeline["alphafold_dir"],   "color": "#8E44AD"},
    ]
    return (
        stages,
        tuple(s["key"] for s in stages),
        {s["key"]: s["label"] for s in stages},
        {s["key"]: s["dir"] for s in stages},
        {s["key"]: s.get("pdb_glob") for s in stages},
        {s["key"]: s.get("color", "#2E86C1") for s in stages},
    )


@dataclass
class Settings:
    """Configuracion viva de una sesion del monitor.

    Sustituye a los globals de modulo. Se construye una vez en el arranque y
    se pasa explicitamente a quien la necesite.
    """
    master_dir: str
    pipeline: dict
    temp_dir: str = ""
    host: str = "?"
    port: int = 0

    # Derivados de pipeline['stages'] — se rellenan en __post_init__.
    stages: list = field(default_factory=list)
    stage_keys: tuple = ()
    stage_labels: dict = field(default_factory=dict)
    stage_dirs: dict = field(default_factory=dict)
    stage_globs: dict = field(default_factory=dict)
    stage_colors: dict = field(default_factory=dict)

    def __post_init__(self):
        (self.stages, self.stage_keys, self.stage_labels,
         self.stage_dirs, self.stage_globs, self.stage_colors) = build_stages(self.pipeline)

    @classmethod
    def from_args(cls, args):
        """Construye Settings desde los argumentos ya parseados de la CLI."""
        return cls(
            master_dir=os.path.abspath(args.dir),
            pipeline=load_pipeline_structure(args.config),
            host=args.host,
            port=args.port,
        )
