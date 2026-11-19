"""Localizacion y manipulacion de archivos PDB.

Modulo puro: la configuracion llega por argumento (`pipeline`, `temp_dir`)
en vez de leerse de globals, para que se pueda usar y probar sin arrancar
la app.
"""
import os
import re
import glob


def get_trajectory_pdb(exp_dir, base_name, pipeline):
    """Localiza el PDB multimodelo de trayectoria asociado a un diseño.

    Recordatorio: RFdiffusion guarda la trayectoria de cada diseño como un
    ÚNICO archivo PDB con N bloques MODEL/ENDMDL dentro (no N archivos).
    El nombre tiene la forma <diseño>_pX0_traj.pdb, sin el sufijo
    _dldesign_X (ese sufijo lo añade ProteinMPNN posteriormente)."""
    rf_dir = os.path.join(exp_dir, pipeline['rfdiffusion_dir'],
                          pipeline['trajectory_subdir'])
    if not os.path.exists(rf_dir):
        return None
    # Quitamos el _dldesign_X para volver al nombre original de RFdiffusion
    rf_base = re.sub(r'_dldesign_\d+$', '', base_name)
    suffix = pipeline['trajectory_suffix']  # típicamente "_pX0_traj"
    cand = glob.glob(os.path.join(rf_dir, f"{rf_base}{suffix}.pdb"))
    if cand:
        return cand[0]
    # Fallback genérico por si el sufijo cambió
    fallback = glob.glob(os.path.join(rf_dir, f"{rf_base}*traj*.pdb"))
    return fallback[0] if fallback else None


def resolve_structure_pdb(exp_dir, hit_basename, pipeline):
    """Localiza el archivo PDB de estructura final para un diseño.

    Soporta:
      - Complexa: <exp>/<hit_basename>/<hit_basename>_updated.pdb
      - AF2 estándar: <alphafold_dir>/[passed/]<hit_basename><af_suffix>.pdb
      - Fallback por glob amplio si los sufijos varían
    Devuelve la ruta absoluta del PDB o None si no se encuentra.
    """
    if not exp_dir or not hit_basename:
        return None

    complexa_updated = os.path.join(exp_dir, hit_basename, f"{hit_basename}_updated.pdb")
    if hit_basename.startswith("job_") and os.path.exists(complexa_updated):
        return complexa_updated

    af_dir = pipeline.get('alphafold_dir', 'AF_output')
    af_root = os.path.join(exp_dir, af_dir)
    af_suffix = pipeline.get('af_suffix', '_af2pred')
    passed_subdir = pipeline.get('passed_subdir', 'passed')

    candidates = [
        os.path.join(af_root, passed_subdir, f"{hit_basename}{af_suffix}.pdb"),
        os.path.join(af_root, f"{hit_basename}{af_suffix}.pdb"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    for subdir in (passed_subdir, ''):
        matches = glob.glob(os.path.join(af_root, subdir, f"{hit_basename}*.pdb"))
        if matches:
            return matches[0]

    return None


def count_pdb_models(pdb_path):
    """Cuenta los bloques de un PDB multimodelo de forma robusta.

    Soporta los dos formatos que produce RFdiffusion:
      - Estándar (MODEL ... ENDMDL ... MODEL ... ENDMDL)
      - Solo-ENDMDL (bloques de ATOM terminados únicamente con ENDMDL,
        sin etiqueta MODEL de apertura — lo emiten algunas versiones)

    Si hay etiquetas MODEL, las cuenta; si no, cuenta los ENDMDL.
    """
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
    # MODEL tiene prioridad; si no hay, contamos ENDMDL
    return n_model if n_model > 0 else n_endmdl


def logical_to_physical_model(logical_idx, n_models, pipeline):
    """Mapea el índice LÓGICO del slider (0 = ruido inicial, n-1 = estructura
    final) al índice FÍSICO del bloque MODEL dentro del PDB.

    Algunos PDBs de trayectoria de RFdiffusion guardan los modelos en orden
    INVERSO: el primer MODEL del archivo es el paso final de difusión y el
    último MODEL es el ruido inicial. Cuando `pipeline['trajectory_reversed']`
    es True, invertimos el mapeo para que el slider siempre vaya de ruido a
    estructura de izquierda a derecha, independientemente de cómo esté
    grabado el archivo.
    """
    logical_idx = max(0, min(int(logical_idx), max(0, n_models - 1)))
    if pipeline.get('trajectory_reversed', False) and n_models > 1:
        return (n_models - 1) - logical_idx
    return logical_idx


def extract_pdb_model_streaming(pdb_path, model_idx, temp_dir):
    """Extrae UN frame del PDB multimodelo y lo sanitiza como PDB mono-modelo.

    Funciona con los dos formatos de RFdiffusion (con o sin etiqueta MODEL).
    El archivo de salida contiene SOLO líneas ATOM/HETATM/TER más END, que
    es lo que Molstar necesita para renderizar sin errores. Las etiquetas
    MODEL y ENDMDL se descartan deliberadamente: Molstar las ignora o las
    interpreta de forma inconsistente si no vienen en pares perfectos.

    El nombre de salida es determinista (`__mNNN`) — pedir el mismo frame
    dos veces sobrescribe el mismo archivo en lugar de crear uno nuevo.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return None
    model_idx = max(0, int(model_idx))

    target_lines = []
    current_frame = 0
    in_target = (model_idx == 0)  # el frame 0 empieza desde el principio

    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('ENDMDL'):
                    if in_target:
                        # Hemos recogido todo el frame que queríamos
                        break
                    current_frame += 1
                    if current_frame == model_idx:
                        in_target = True
                    continue  # nunca copiamos ENDMDL al output

                if line.startswith('MODEL'):
                    continue  # descartamos MODEL para no confundir a Molstar

                # Solo copiamos coordenadas; el resto (REMARK, CRYST1...)
                # no aporta nada al render y puede confundir a Molstar
                if in_target and (line.startswith('ATOM')
                                  or line.startswith('HETATM')
                                  or line.startswith('TER')):
                    target_lines.append(line)

    except OSError as e:
        print(f"[BinderMonitor] Error extrayendo frame {model_idx}: {e}")
        return None

    if not target_lines:
        # Fallback: si no encontramos ningún frame, devolvemos el original
        return pdb_path

    base = os.path.basename(pdb_path).replace('.pdb', '')
    out_path = os.path.join(temp_dir, f"{base}__m{model_idx:03d}.pdb")
    try:
        with open(out_path, 'w') as g:
            g.writelines(target_lines)
            g.write('END\n')  # cierre obligatorio del formato PDB para Molstar
    except OSError as e:
        print(f"[BinderMonitor] Error escribiendo frame temporal: {e}")
        return None
    return out_path
