"""Localización y manipulación de archivos PDB.

Módulo puro: la configuración llega por argumento (`pipeline`, `temp_dir`)
en vez de leerse de globals, para que se pueda usar y probar sin arrancar
la app.
"""
import glob
import os
import re


def _matches_exact_stem(filepath, prefix):
    """Evita falsos positivos de prefijo numérico (p.ej. design_1 vs design_10)."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if stem == prefix:
        return True
    if not stem.startswith(prefix):
        return False
    rest = stem[len(prefix):]
    # Debe ir seguido de '_' y NO de otro índice numérico (_0, _10, _dldesign_2...)
    if not rest.startswith('_'):
        return False
    if re.match(r'^_(?:\d+|dldesign_\d+)(?:_|$)', rest):
        return False
    return True


def get_trajectory_pdb(exp_dir, base_name, pipeline):
    """Localiza el PDB multimodelo de trayectoria asociado a un diseño.

    RFdiffusion guarda la trayectoria de cada diseño como un único archivo PDB
    con N bloques MODEL/ENDMDL (`<diseño>_pX0_traj.pdb`), sin el sufijo
    `_dldesign_X` que añade ProteinMPNN posteriormente.
    """
    if not exp_dir or not base_name:
        return None
    pipeline = pipeline or {}
    rf_dir = os.path.join(
        exp_dir,
        pipeline.get('rfdiffusion_dir', 'RF_output'),
        pipeline.get('trajectory_subdir', 'traj'),
    )
    if not os.path.isdir(rf_dir):
        return None

    rf_base = re.sub(r'_dldesign_\d+$', '', base_name)
    suffix = pipeline.get('trajectory_suffix', '_pX0_traj')
    exact = os.path.join(rf_dir, f"{rf_base}{suffix}.pdb")
    if os.path.isfile(exact):
        return exact

    for path in sorted(glob.glob(os.path.join(rf_dir, f"{rf_base}_*traj*.pdb"))):
        if os.path.isfile(path) and _matches_exact_stem(path, rf_base):
            return path
    return None


def resolve_structure_pdb(exp_dir, hit_basename, pipeline):
    """Localiza el archivo PDB de estructura final para un diseño.

    Soporta:
      - Complexa: <exp>/<hit_basename>/<hit_basename>_updated.pdb (o .pdb)
      - AF2 estándar: <alphafold_dir>/[passed/|''|discarded/]<hit_basename><af_suffix>.pdb
      - Fallback por sufijos variables sin colisionar con índices numéricos vecinos
    Devuelve la ruta del PDB o None si no se encuentra.
    """
    if not exp_dir or not hit_basename:
        return None
    pipeline = pipeline or {}

    complexa_updated = os.path.join(exp_dir, hit_basename, f"{hit_basename}_updated.pdb")
    if os.path.isfile(complexa_updated):
        return complexa_updated
    if hit_basename.startswith("job_"):
        complexa_plain = os.path.join(exp_dir, hit_basename, f"{hit_basename}.pdb")
        if os.path.isfile(complexa_plain):
            return complexa_plain

    af_root = os.path.join(exp_dir, pipeline.get('alphafold_dir', 'AF_output'))
    af_suffix = pipeline.get('af_suffix', '_af2pred')
    passed_subdir = pipeline.get('passed_subdir', 'passed')
    discarded_subdir = pipeline.get('discarded_subdir', 'discarded')

    search_dirs = [
        os.path.join(af_root, passed_subdir),
        af_root,
        os.path.join(af_root, discarded_subdir),
    ]

    for d in search_dirs:
        for name in (f"{hit_basename}{af_suffix}.pdb", f"{hit_basename}.pdb"):
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand

    for d in search_dirs:
        for path in sorted(glob.glob(os.path.join(d, f"{hit_basename}_*.pdb"))):
            if os.path.isfile(path) and _matches_exact_stem(path, hit_basename):
                return path

    return None


def count_pdb_models(pdb_path):
    """Cuenta los bloques de un PDB multimodelo (con o sin etiqueta MODEL)."""
    if not pdb_path or not os.path.exists(pdb_path):
        return 0
    n_model = n_endmdl = 0
    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('MODEL'):
                    n_model += 1
                elif line.startswith('ENDMDL'):
                    n_endmdl += 1
    except OSError:
        return 0
    return n_model if n_model > 0 else n_endmdl


def logical_to_physical_model(logical_idx, n_models, pipeline):
    """Mapea el índice lógico del slider (0 = ruido, n-1 = final) al índice
    físico del bloque MODEL dentro del PDB."""
    pipeline = pipeline or {}
    logical_idx = max(0, min(int(logical_idx), max(0, n_models - 1)))
    if pipeline.get('trajectory_reversed', False) and n_models > 1:
        return (n_models - 1) - logical_idx
    return logical_idx


def extract_pdb_model_streaming(pdb_path, model_idx, temp_dir):
    """Extrae un frame del PDB multimodelo y lo sanitiza como PDB mono-modelo."""
    if not pdb_path or not os.path.exists(pdb_path) or not temp_dir:
        return None
    model_idx = max(0, int(model_idx))

    target_lines = []
    current_frame = 0
    in_target = (model_idx == 0)

    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('ENDMDL'):
                    if in_target:
                        break
                    current_frame += 1
                    if current_frame == model_idx:
                        in_target = True
                    continue
                if line.startswith('MODEL'):
                    continue
                if in_target and line.startswith(('ATOM', 'HETATM', 'TER')):
                    target_lines.append(line)
    except OSError as e:
        print(f"[BinderMonitor] Error extrayendo frame {model_idx}: {e}")
        return None

    if not target_lines:
        return pdb_path

    base = os.path.splitext(os.path.basename(pdb_path))[0]
    out_path = os.path.join(temp_dir, f"{base}__m{model_idx:03d}.pdb")
    try:
        with open(out_path, 'w', encoding='utf-8') as g:
            g.writelines(target_lines)
            g.write('END\n')
    except OSError as e:
        print(f"[BinderMonitor] Error escribiendo frame temporal: {e}")
        return None
    return out_path
