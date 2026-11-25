"""Exploración del sistema de ficheros de una campaña.

Descubre experimentos, localiza sus tablas de métricas, normaliza nombres de
diseño y cuenta PDBs por etapa. La configuración (`pipeline`) llega por
argumento en vez de leerse de un global.
"""
import fnmatch
import glob
import os
import re
import pandas as pd

from binderlab.core.config import DEFAULT_PIPELINE_STRUCTURE

_SC_TAG_COLS = ('SCORE:', 'SEQUENCE:', 'SCORE_LABEL')


def clean_rosetta_sc_df(df):
    """Limpia artefactos de archivos .sc de Rosetta/AF2 (columnas decorativas y
    cabeceras repetidas al concatenar salidas de varios jobs SLURM), sin alterar
    columnas numéricas legítimas llamadas `SCORE` en archivos CSV/TSV."""
    drop_cols = [c for c in _SC_TAG_COLS if c in df.columns]
    if 'SCORE' in df.columns:
        non_na = df['SCORE'].dropna().astype(str).str.rstrip(':')
        if not non_na.empty and non_na.eq('SCORE').all():
            drop_cols.append('SCORE')
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if 'description' in df.columns:
        df = df[df['description'].astype(str) != 'description'].copy()
    for col in df.columns:
        if col in ('description', 'pdb_path', 'sequence'):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            conv = pd.to_numeric(df[col], errors='coerce')
            if conv.notna().sum() >= df[col].notna().sum():
                df[col] = conv
    return df.reset_index(drop=True)


def get_experiments(base_dir, pipeline):
    """Devuelve experimentos válidos dentro de base_dir (hasta 2 niveles)."""
    if not base_dir or not os.path.isdir(base_dir):
        return []
    pipeline = pipeline or DEFAULT_PIPELINE_STRUCTURE

    markers = (
        pipeline.get('rfdiffusion_dir', 'RF_output'),
        pipeline.get('proteinmpnn_dir', 'MPNN_output'),
        pipeline.get('alphafold_dir', 'AF_output'),
        'out.sc',
    )
    extra_markers = pipeline.get('extra_markers', [])

    def _is_experiment(path):
        if any(os.path.exists(os.path.join(path, m)) for m in markers):
            return True
        return any(glob.glob(os.path.join(path, g)) for g in extra_markers)

    def _list_subdirs(path):
        try:
            return sorted(
                os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            )
        except OSError:
            return []

    if _is_experiment(base_dir):
        return [base_dir]

    valid, non_exp_subdirs = [], []
    for sub in _list_subdirs(base_dir):
        if _is_experiment(sub):
            valid.append(sub)
        else:
            non_exp_subdirs.append(sub)

    for parent in non_exp_subdirs:
        for sub2 in _list_subdirs(parent):
            if _is_experiment(sub2):
                valid.append(sub2)

    return sorted(valid)


def get_metric_files(exp_dir, pipeline):
    """Lista los archivos de métricas legibles (.sc, .csv, .tsv) del experimento."""
    if not exp_dir or not os.path.isdir(exp_dir):
        return []
    pipeline = pipeline or DEFAULT_PIPELINE_STRUCTURE
    patterns = pipeline.get(
        'metric_patterns',
        ['out.sc', '*sorted*.sc', 'Hits_Scoring*.sc', 'Scoring*.sc', '*.csv', '*.tsv'],
    )
    exclude = pipeline.get('metric_exclude', [])
    found, seen = [], set()
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(exp_dir, pat))):
            if f in seen or not os.path.isfile(f):
                continue
            if any(fnmatch.fnmatch(os.path.basename(f), e) for e in exclude):
                continue
            seen.add(f)
            found.append(f)
    return found


def get_base_name(filename, pipeline):
    """Limpia extensión `.pdb` y sufijo de AF2 (`_af2pred`), conservando `_dldesign_X`."""
    pipeline = pipeline or DEFAULT_PIPELINE_STRUCTURE
    name = os.path.basename(str(filename))
    if name.lower().endswith('.pdb'):
        name = name[:-4]
    af_suffix = pipeline.get('af_suffix', '_af2pred')
    if af_suffix:
        name = re.sub(re.escape(af_suffix) + r'$', '', name)
    return name


def count_funnel_stage(exp_dir, stage_folder, pipeline, pdb_glob=None):
    """Cuenta cuántos PDBs de resultado hay en una etapa: (total, passed, discarded)."""
    if not exp_dir:
        return 0, 0, 0
    pipeline = pipeline or DEFAULT_PIPELINE_STRUCTURE
    folder = os.path.join(exp_dir, stage_folder)
    if not os.path.exists(folder):
        return 0, 0, 0

    if pdb_glob:
        n = len(glob.glob(os.path.join(folder, pdb_glob)))
        return n, n, 0

    passed_name = pipeline.get('passed_subdir', 'passed')
    discarded_name = pipeline.get('discarded_subdir', 'discarded')
    traj_name = pipeline.get('trajectory_subdir', 'traj')
    excluded = {passed_name, discarded_name, traj_name, *pipeline.get('non_result_subdirs', [])}

    def _pdbs(d):
        return len(glob.glob(os.path.join(d, '*.pdb'))) if os.path.isdir(d) else 0

    raw_count = _pdbs(folder)
    try:
        for d in os.listdir(folder):
            full = os.path.join(folder, d)
            if os.path.isdir(full) and d not in excluded:
                raw_count += _pdbs(full)
    except OSError:
        pass

    p_count = _pdbs(os.path.join(folder, passed_name))
    d_count = _pdbs(os.path.join(folder, discarded_name))

    if p_count == 0 and d_count == 0:
        return raw_count, raw_count, 0

    total = raw_count + p_count + d_count
    return total, raw_count + p_count, d_count


def _base_from_pdb_path(p, pipeline):
    if pd.isna(p) or not str(p).strip():
        return ""
    parent = os.path.basename(os.path.dirname(str(p)))
    return parent if parent else get_base_name(p, pipeline)


def parse_metrics_file(metric_file, pipeline):
    """Lee un archivo de métricas (.sc, .csv o .tsv) y normaliza `base_name`."""
    if not metric_file or not os.path.exists(metric_file):
        return pd.DataFrame()
    pipeline = pipeline or DEFAULT_PIPELINE_STRUCTURE
    try:
        if metric_file.endswith('.csv'):
            df = pd.read_csv(metric_file, low_memory=False)
        elif metric_file.endswith('.tsv'):
            df = pd.read_csv(metric_file, sep='\t', low_memory=False)
        else:
            df = pd.read_csv(metric_file, sep=r'\s+', header=0, engine='python')
        df = clean_rosetta_sc_df(df)
        if 'description' in df.columns:
            df['base_name'] = df['description'].apply(lambda x: get_base_name(x, pipeline))
        elif 'pdb_path' in df.columns:
            df['base_name'] = df['pdb_path'].apply(lambda p: _base_from_pdb_path(p, pipeline))
        return df
    except Exception as e:
        print(f"[BinderMonitor] Error parseando {metric_file}: {e}")
        return pd.DataFrame()
