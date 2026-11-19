"""Exploracion del sistema de ficheros de una campana.

Descubre experimentos, localiza sus tablas de metricas, normaliza nombres de
diseno y cuenta PDBs por etapa. La configuracion (`pipeline`) llega por
argumento en vez de leerse de un global, para que se pueda usar y probar sin
arrancar la app.
"""
import os
import glob
import re
import fnmatch
import pandas as pd


def get_experiments(base_dir, pipeline):
    """Devuelve experimentos válidos dentro de base_dir.

    Escanea hasta DOS niveles de profundidad para soportar tres escenarios:

      1. base_dir ES un experimento aislado (tiene RF_output, out.sc...)
         → devuelve [base_dir]

      2. base_dir es una carpeta de campañas múltiples
         (multiple_binders_midasin_260128/)
         → devuelve sus subcarpetas que sean experimentos

      3. base_dir es la raíz del proyecto (/data/.../RFdiffusion/)
         → escanea nivel 2: subcarpetas de las campañas

    Así se puede lanzar el monitor UNA sola vez con -d apuntando a la raíz
    y navegar entre todos los experimentos desde la UI, sin hacer cd.
    """
    if not base_dir or not os.path.isdir(base_dir):
        return []

    markers = (
        pipeline['rfdiffusion_dir'],
        pipeline['proteinmpnn_dir'],
        pipeline['alphafold_dir'],
        'out.sc',
    )

    def _is_experiment(path):
        if any(os.path.exists(os.path.join(path, m)) for m in markers):
            return True
        # marcadores glob para pipelines sin carpetas RF/MPNN/AF
        return any(glob.glob(os.path.join(path, g))
                   for g in pipeline.get('extra_markers', []))

    def _list_subdirs(path):
        try:
            return sorted(
                os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            )
        except (PermissionError, OSError):
            return []

    # Caso 1: el propio directorio es un experimento aislado
    if _is_experiment(base_dir):
        return [base_dir]

    # Caso 2: buscar experimentos en nivel 1 (subcarpetas directas)
    valid = []
    non_exp_subdirs = []
    for sub in _list_subdirs(base_dir):
        if _is_experiment(sub):
            valid.append(sub)
        else:
            non_exp_subdirs.append(sub)

    # Caso 3: si ningún hijo directo es experimento, buscar nivel 2
    # (subcarpetas de las campañas). Si ya encontramos algo en nivel 1,
    # también escaneamos nivel 2 de los que NO eran experimentos, para
    # no perder campañas mezcladas con experimentos sueltos.
    for parent in non_exp_subdirs:
        for sub2 in _list_subdirs(parent):
            if _is_experiment(sub2):
                valid.append(sub2)

    return sorted(valid)


def get_metric_files(exp_dir, pipeline):
    """Lista los archivos de métricas legibles dentro del experimento.
    Soporta los .sc de Rosetta/AF2 y CSVs (HITS_ELITE.csv, etc.)."""
    if not exp_dir or not os.path.isdir(exp_dir):
        return []
    patterns = pipeline.get('metric_patterns',
                ['out.sc', '*sorted*.sc', 'Hits_Scoring*.sc',
                 'Scoring*.sc', '*.csv', '*.tsv'])
    exclude = pipeline.get('metric_exclude', [])
    found, seen = [], set()
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(exp_dir, pat))):
            if f in seen or not os.path.isfile(f):
                continue
            # descarta CSV de ruido (RAW, transpuestos...) via -c JSON
            if any(fnmatch.fnmatch(os.path.basename(f), e) for e in exclude):
                continue
            seen.add(f)
            found.append(f)
    return found


def get_base_name(filename, pipeline):
    """Limpia el sufijo de AlphaFold2 (`_af2pred`) y la extensión `.pdb`,
    pero conserva el `_dldesign_X` de ProteinMPNN, porque cada variante
    de secuencia es un diseño DISTINTO y necesita su propia identidad
    (es la "clave primaria" para cruzar PDB ↔ out.sc ↔ trayectoria)."""
    name = os.path.basename(str(filename))
    name = name.replace('.pdb', '')
    af_suffix = pipeline['af_suffix']  # típicamente "_af2pred"
    name = re.sub(re.escape(af_suffix) + r'$', '', name)
    return name


def count_funnel_stage(exp_dir, stage_folder, pipeline, pdb_glob=None):
    """Cuenta cuántos PDBs de RESULTADO hay en una etapa del pipeline.

    Devuelve (total, passed, discarded).

    Reglas de conteo (importantes para no inflar cifras):
      - Solo cuentan PDBs de RESULTADO. Las carpetas auxiliares como `traj/`
        (trayectorias de difusión de RFdiffusion: un .pdb intermedio por
        diseño) NO son resultados y se EXCLUYEN siempre.
      - Si existen subcarpetas de filtro `passed/` y `discarded/`, la verdad
        del cribado vive ahí: passed = nº .pdb en passed/, discarded = nº .pdb
        en discarded/. Los .pdb sueltos en la raíz se consideran "en tránsito"
        (generados pero aún sin clasificar) y se añaden a passed.
      - Si NO hay subcarpetas de filtro, la etapa aún no se ha clasificado:
        todo .pdb de resultado cuenta como generado (passed provisional).
      - Si la etapa no existe: (0, 0, 0).

    También recorre subdirectorios de primer nivel que SÍ son de resultado
    (p.ej. AF_output/<nombre_experimento>/*.pdb), excluyendo siempre las
    carpetas auxiliares y de filtro.

    NOTA: el conteo es de archivos .pdb. RFdiffusion escribe además un .trb
    por diseño; al filtrar por extensión .pdb ya quedan fuera.
    """
    folder = os.path.join(exp_dir, stage_folder)
    if not os.path.exists(folder):
        return 0, 0, 0

    # etapa N sin carpetas passed/discarded -> cuenta por glob
    # (p.ej. Complexa job_*/*_updated.pdb). total=passed, discarded=0.
    if pdb_glob:
        n = len(glob.glob(os.path.join(folder, pdb_glob)))
        return n, n, 0

    passed_name    = pipeline['passed_subdir']
    discarded_name = pipeline['discarded_subdir']
    traj_name      = pipeline.get('trajectory_subdir', 'traj')
    # Subcarpetas que NUNCA cuentan como resultados de la etapa.
    excluded = {passed_name, discarded_name, traj_name}
    excluded.update(pipeline.get('non_result_subdirs', []))

    def _pdbs(d):
        return len(glob.glob(os.path.join(d, '*.pdb'))) if os.path.isdir(d) else 0

    # --- PDBs sueltos en la raíz de la etapa (en tránsito) -----------------
    raw_count = _pdbs(folder)

    # --- PDBs en subdirectorios de RESULTADO (excluye auxiliares/filtro) ---
    try:
        for d in os.listdir(folder):
            full = os.path.join(folder, d)
            if os.path.isdir(full) and d not in excluded:
                raw_count += _pdbs(full)
    except OSError:
        pass

    # --- Subcarpetas de filtrado (la verdad del cribado) -------------------
    p_count = _pdbs(os.path.join(folder, passed_name))
    d_count = _pdbs(os.path.join(folder, discarded_name))

    # Caso 1: sin clasificar todavía → todo lo de raíz es passed provisional.
    if p_count == 0 and d_count == 0:
        return raw_count, raw_count, 0

    # Caso 2: ya clasificado. passed/discarded mandan; lo suelto en raíz son
    # diseños aún sin clasificar (se suman a passed, optimismo).
    total = raw_count + p_count + d_count
    return total, raw_count + p_count, d_count


def parse_metrics_file(metric_file, pipeline):
    """Lee un archivo de métricas (.sc de Rosetta/AF2 o CSV) y devuelve
    un DataFrame. Detecta el formato por la extensión y limpia residuos
    típicos de Rosetta (la columna "SCORE:" repetida)."""
    if not metric_file or not os.path.exists(metric_file):
        return pd.DataFrame()
    try:
        if metric_file.endswith('.csv'):
            df = pd.read_csv(metric_file, low_memory=False)
        elif metric_file.endswith('.tsv'):
            df = pd.read_csv(metric_file, sep='\t', low_memory=False)
        else:
            # Los .sc usan whitespace variable como separador
            df = pd.read_csv(metric_file, sep=r'\s+', header=0, engine='python')
        for junk in ('SCORE:', 'SEQUENCE:'):
            if junk in df.columns:
                df = df.drop(columns=[junk])
        if 'description' in df.columns:
            df['base_name'] = df['description'].apply(
                lambda x: get_base_name(x, pipeline))
        elif 'pdb_path' in df.columns:
            # Complexa no tiene 'description'; el nombre de carpeta
            # padre en pdb_path ES el identificador unico del diseno.
            df['base_name'] = df['pdb_path'].apply(
                lambda p: os.path.basename(os.path.dirname(str(p))))
        return df
    except Exception as e:
        print(f"[BinderMonitor] Error parseando {metric_file}: {e}")
        return pd.DataFrame()
