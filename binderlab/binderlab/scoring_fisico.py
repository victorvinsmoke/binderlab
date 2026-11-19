"""Cálculo de métricas físicas y termodinámicas de interfaz en diseños AF2 con PyRosetta.

Aplica un protocolo de empaquetamiento y minimización de cadenas laterales restringido a la
interfaz (8 Å) manteniendo el backbone congelado para preservar la topología de AF2.
Calcula afinidad estimada (ddG), empaquetamiento (packstat, shape complementarity),
superficie enterrada (dSASA), puentes de hidrógeno insatisfechos (BUNs) y SAP score.

Uso:
  binderlab score --pdbdir /ruta/pdbs --hits hits.txt --out scores.tsv [opciones]
"""

import argparse
import pandas as pd
import sys
import time
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Estado por proceso worker (PyRosetta no es thread-safe).
# Se instancian en worker_init() y se reutilizan entre PDBs.
_scorefxn           = None
_interface_analyzer = None
_buns_filter        = None
_sap_metric         = None
_no_relax           = False
_binder_chain       = 1


def worker_init(no_relax_flag, binder_chain=1):
    import pyrosetta
    import pyrosetta.rosetta.core.select.residue_selector as res_sel
    import pyrosetta.rosetta.protocols.analysis as analysis
    import pyrosetta.rosetta.core.pack.guidance_scoreterms.sap as sap
    import pyrosetta.rosetta.protocols.rosetta_scripts as rosetta_scripts

    global _scorefxn, _interface_analyzer, _buns_filter, _sap_metric, _no_relax, _binder_chain

    _no_relax = no_relax_flag
    _binder_chain = int(binder_chain)

    pyrosetta.init("-mute all")

    _scorefxn = pyrosetta.get_fa_scorefxn()

    _interface_analyzer = analysis.InterfaceAnalyzerMover()
    _interface_analyzer.set_scorefunction(_scorefxn)
    _interface_analyzer.set_compute_packstat(True)
    _interface_analyzer.set_compute_interface_energy(True)
    _interface_analyzer.set_calc_dSASA(True)
    _interface_analyzer.set_calc_hbond_sasaE(True)
    _interface_analyzer.set_compute_interface_sc(True)
    _interface_analyzer.set_pack_separated(True)

    xml_string = ('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" '
                  'ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="0" '
                  'probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    _buns_filter = rosetta_scripts.XmlObjects.static_get_filter(xml_string)

    binder_sel = res_sel.ChainSelector(_binder_chain)
    _sap_metric = sap.SapScoreMetric()
    _sap_metric.set_sap_calculate_selector(binder_sel)


def interface_pack_min(pose):
    """Minimización y reempaquetamiento restringido a residuos de interfaz (8 Å)."""
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking, OperateOnResidueSubset, PreventRepackingRLT
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        NeighborhoodResidueSelector, NotResidueSelector
    )
    from pyrosetta.rosetta.protocols.minimization_packing import (
        PackRotamersMover, MinMover
    )

    start_binder = pose.conformation().chain_begin(_binder_chain)
    end_binder   = pose.conformation().chain_end(_binder_chain)

    nrs = NeighborhoodResidueSelector()
    nrs.set_focus(f'{start_binder}-{end_binder}')
    nrs.set_distance(8.0)
    nrs.set_include_focus_in_subset(True)
    interface_mask = nrs.apply(pose)

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())

    non_interface_sel = NotResidueSelector(nrs)
    prevent_op = OperateOnResidueSubset(PreventRepackingRLT(), non_interface_sel)
    tf.push_back(prevent_op)

    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(_scorefxn, packer_task)
    pack_mover.apply(pose)

    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)

    for i in range(1, pose.total_residue() + 1):
        if interface_mask[i]:
            mm.set_chi(i, True)

    min_mover = MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(_scorefxn)
    min_mover.min_type('lbfgs_armijo_nonmonotone')
    min_mover.tolerance(0.01)
    min_mover.apply(pose)

    return pose


# ==============================================================================
# FUNCIONES DE ANÁLISIS
# ==============================================================================

def get_close_residues(pose, distance=10.0):
    """Identifica residuos de interfaz entre el binder y las cadenas de la diana."""
    from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector

    NRS_binder = NeighborhoodResidueSelector()
    NRS_target = NeighborhoodResidueSelector()
    NRS_binder.set_distance(distance)
    NRS_target.set_distance(distance)

    target_chains = [c for c in range(1, pose.num_chains() + 1) if c != _binder_chain]
    start_binder = pose.conformation().chain_begin(_binder_chain)
    end_binder   = pose.conformation().chain_end(_binder_chain)

    NRS_binder.set_focus(f'{start_binder}-{end_binder}')
    neighbors_binder = NRS_binder.apply(pose)

    target_ranges = [
        f"{pose.conformation().chain_begin(c)}-{pose.conformation().chain_end(c)}"
        for c in target_chains
    ]
    NRS_target.set_focus(','.join(target_ranges))
    neighbors_target = NRS_target.apply(pose)

    close_residues_binder = [
        j for j in range(start_binder, end_binder + 1)
        if neighbors_target[j]
    ]
    close_residues_target = [
        i for c in target_chains
        for i in range(pose.conformation().chain_begin(c), pose.conformation().chain_end(c) + 1)
        if neighbors_binder[i]
    ]

    length = int(end_binder - start_binder + 1)
    return close_residues_binder, close_residues_target, length


def compute_rosetta_metrics(pose, interface):
    """Extrae métricas termodinámicas y geométricas de la interfaz."""
    _interface_analyzer.set_interface(interface)
    _interface_analyzer.apply(pose)

    data = _interface_analyzer.get_all_data()

    sc       = data.sc_value
    dsasa    = _interface_analyzer.get_interface_delta_sasa()
    packstat = _interface_analyzer.get_interface_packstat()
    hbonds   = data.interface_hbonds
    ddG      = _interface_analyzer.get_interface_dG()

    unsat_hbonds = _buns_filter.report_sm(pose)
    sap_score    = _sap_metric.calculate(pose)

    return ddG, sc, dsasa, packstat, hbonds, unsat_hbonds, sap_score


def compute_interface_hydrophobicity(pose, close_residues_binder):
    """Fracción de residuos apolares/aromáticos del binder en contacto con el target."""
    total = len(close_residues_binder)
    if total == 0:
        return 0.0
    hydrophobic = sum(
        1 for i in close_residues_binder
        if pose.residue(i).is_apolar() or pose.residue(i).name() in ['PHE', 'TRP', 'TYR']
    )
    return hydrophobic / total


def compute_surface_hydrophobicity(pose):
    """Hidrofobicidad superficial del binder aislado para evaluar riesgo de agregación."""
    from pyrosetta.rosetta.core.select.residue_selector import LayerSelector

    binder_pose = pose.split_by_chain()[_binder_chain]

    layer_sel = LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_mask = layer_sel.apply(binder_pose)

    hydrophobic = 0
    total       = 0
    for i in range(1, len(surface_mask) + 1):
        if surface_mask[i]:
            res = binder_pose.residue(i)
            if res.is_apolar() or res.name() in ['PHE', 'TRP', 'TYR']:
                hydrophobic += 1
            total += 1

    return hydrophobic / total if total > 0 else 0.0


def score_one_pdb(job):
    """Procesa un PDB y extrae métricas de interfaz."""
    from pyrosetta import pose_from_pdb

    pdb_path, clean_hit = job
    t0 = time.perf_counter()

    try:
        pose = pose_from_pdb(str(pdb_path))
        if pose.num_chains() < 2:
            print(f"ERROR: {clean_hit} tiene {pose.num_chains()} cadena(s). Saltando.")
            return None
        if _binder_chain < 1 or _binder_chain > pose.num_chains():
            print(f"ERROR: {clean_hit} tiene {pose.num_chains()} cadena(s); "
                  f"--binder_chain={_binder_chain} fuera de rango.")
            return None
    except Exception as e:
        print(f"ERROR al cargar {clean_hit}: {e}")
        return None

    if not _no_relax:
        pose = interface_pack_min(pose)

    close_res_binder, _, length = get_close_residues(pose)

    target_chains = [c for c in range(1, pose.num_chains() + 1) if c != _binder_chain]
    ch_binder = pose.pdb_info().chain(pose.conformation().chain_begin(_binder_chain))
    ch_targets = "".join(
        pose.pdb_info().chain(pose.conformation().chain_begin(c)) for c in target_chains
    )

    ddG, sc, dsasa, packstat, hbonds, unsat_hbonds, sap_score = compute_rosetta_metrics(
        pose, f'{ch_binder}_{ch_targets}'
    )

    binder_int_hyd  = compute_interface_hydrophobicity(pose, close_res_binder)
    binder_surf_hyd = compute_surface_hydrophobicity(pose)

    print(f"Procesado: {clean_hit} ({time.perf_counter() - t0:.1f}s)")

    return {
        'description':            str(clean_hit),
        'ddG':                    float(ddG),
        'Shape_complementarity':  float(sc),
        'dSASA':                  float(dsasa),
        'Packstat':               float(packstat),
        'interface_hbonds':       int(hbonds),
        'interface_unsat_hbonds': float(unsat_hbonds),
        'SAP':                    float(sap_score),
        'binder_int_hyd':         float(binder_int_hyd),
        'binder_surf_hyd':        float(binder_surf_hyd),
        'binder_length':          int(length),
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--pdbdir',   required=True,  help='Directorio con los PDBs')
    parser.add_argument('--hits',     required=True,  help='Archivo con nombres de diseños')
    parser.add_argument('--sc',       default='',     help='Archivo *_sorted.sc de AF2')
    parser.add_argument('--out',      required=True,  help='Archivo de salida (TSV)')
    parser.add_argument('--binder_chain', type=int, default=1,
                        help='Índice de cadena de Rosetta para el binder (1-indexed; por defecto: 1)')
    parser.add_argument('--no_relax', action='store_true', default=False,
                        help='Omitir Pack+Min (ddG menos fiable)')
    parser.add_argument('--workers',  type=int, default=1,
                        help='Procesos paralelos. RAM estimada: ~3 GB/worker. (por defecto: 1)')
    args = parser.parse_args()

    import importlib.util
    if importlib.util.find_spec("pyrosetta") is None:
        print("[!] ERROR: 'pyrosetta' no está instalado en este entorno de Python.")
        print("    PyRosetta requiere una licencia de RosettaCommons y no está disponible en PyPI público.")
        print("    Descárgalo e instálalo desde: https://www.pyrosetta.org/downloads")
        sys.exit(1)

    pdb_dir   = Path(args.pdbdir)
    hits_path = Path(args.hits)

    if not hits_path.exists():
        print(f"ERROR: Archivo de hits no encontrado: {args.hits}")
        sys.exit(1)

    with open(hits_path) as f:
        hit_names = [hit.strip() for hit in f if hit.strip()]

    jobs = []
    for hit in hit_names:
        clean = hit.replace('.pdb', '')
        path  = pdb_dir / f'{clean}.pdb'
        if path.exists():
            jobs.append((path, clean))
        else:
            print(f"AVISO: {path.name} no encontrado. Saltando.")

    if not jobs:
        print("ERROR: Ningún PDB encontrado. Abortando.")
        sys.exit(1)

    n_workers = min(args.workers, len(jobs))

    print(f"Diseños a evaluar : {len(jobs)}")
    print(f"Workers           : {n_workers}")
    print(f"Protocolo         : {'DESACTIVADO (--no_relax)' if args.no_relax else 'Pack+Min restringido a interfaz (8 Å)'}")

    t0 = time.perf_counter()
    rows = []

    if n_workers == 1:
        worker_init(args.no_relax, binder_chain=args.binder_chain)
        for job in jobs:
            result = score_one_pdb(job)
            if result is not None:
                rows.append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=worker_init,
            initargs=(args.no_relax, args.binder_chain)
        ) as executor:
            futures = {executor.submit(score_one_pdb, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rows.append(result)

    if not rows:
        print("ERROR: Ningún diseño procesado con éxito.")
        sys.exit(1)

    rosetta_df = pd.DataFrame(rows, columns=[
        'description', 'ddG', 'Shape_complementarity', 'dSASA',
        'Packstat', 'interface_hbonds', 'interface_unsat_hbonds',
        'SAP', 'binder_int_hyd', 'binder_surf_hyd', 'binder_length'
    ])

    if args.sc and Path(args.sc).exists():
        try:
            af2_df = pd.read_table(args.sc, sep=r'\s+')

            if 'SCORE:' in af2_df.columns:
                af2_df = af2_df.drop(columns=['SCORE:'])
            for etiqueta in ('SCORE_LABEL', 'SCORE'):
                if etiqueta in af2_df.columns:
                    af2_df = af2_df.drop(columns=[etiqueta])

            necesarias = ['description', 'pae_interaction', 'plddt_total']
            faltan = [c for c in necesarias if c not in af2_df.columns]
            if faltan:
                raise ValueError(
                    f"el .sc no tiene las columnas {faltan}. "
                    f"Encontradas: {list(af2_df.columns)}")

            af2_df = af2_df[af2_df['description'].notna()]
            af2_df['description'] = af2_df['description'].astype(str)

            af2_df['pae_plddt_ratio'] = af2_df['pae_interaction'] / (
                af2_df['plddt_total'].replace(0, float('nan')) / 100
            )

            final_df = pd.merge(rosetta_df, af2_df, on='description', how='left')
        except Exception as e:
            print(f"ERROR al procesar .sc de AF2: {e}. Guardando solo métricas de Rosetta.")
            final_df = rosetta_df
    else:
        print("AVISO: Sin archivo .sc de AF2. Guardando solo métricas de Rosetta.")
        final_df = rosetta_df

    final_df = final_df.round(3)
    final_df = final_df.sort_values('ddG', ascending=True).reset_index(drop=True)
    final_df.to_csv(args.out, sep='\t', index=False)

    elapsed = time.perf_counter() - t0
    print(f"Completado en {elapsed:.1f}s ({elapsed/len(rows):.1f}s/diseño)")
    print(f"Guardado en: {args.out}")


if __name__ == '__main__':
    main()
