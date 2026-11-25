"""Optimización de codones para síntesis de ADN (IDT gBlocks/eBlocks).

Aplica restricciones de contenido de GC global (≤60%) y local (≤65%), eliminación de homopolímeros
(≥6 nt) y prevención estricta de repetidos de 9-mers sobre frecuencias de codones del hospedador.

Uso:
    binderlab dna --single MAAAKKKEEE
    binderlab dna -i binders.fasta -o resultados
    binderlab dna -i hits.csv --seq_col sequence -o resultados
    binderlab dna -i binders.fasta -o resultados --reports

Organismos: e_coli (default), human, yeast, mouse, b_subtilis
"""

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# import perezoso. A import-time NO morimos si falta dnachisel;
# solo falla si de verdad ejecutas este subcomando.
def _load_deps():
    global dc, get_codons_table
    try:
        import dnachisel as dc
        from python_codon_tables import get_codons_table
    except ImportError:
        sys.exit("Instala dependencias: pip install dnachisel biopython python-codon-tables")


dc = None
get_codons_table = None

ORGANISMS = {
    "e_coli":     "e_coli_316407",
    "human":      "h_sapiens_9606",
    "yeast":      "s_cerevisiae_4932",
    "mouse":      "m_musculus_10090",
    "b_subtilis": "b_subtilis_1423",
}


def _min_gc(aa_seq, species):
    """
    Calcula el GC mínimo teóricamente alcanzable para esta secuencia y organismo.

    Necesario para secuencias ricas en aminoácidos con codones de alto GC
    (Ala, Gly, Pro...) donde imponer GC ≤ 60% podría ser imposible.
    En ese caso, el techo se relaja a min_gc + 4%.
    """
    try:
        table = get_codons_table(species)
        gc = sum(
            min(c.count("G") + c.count("C")
                for c in {c for c, f in table.get(aa, {}).items() if f > 0} or ["GGG"])
            for aa in aa_seq
        )
        return gc / (len(aa_seq) * 3)
    except Exception:
        return 0.40


def optimize_for_idt(aa_seq, seq_name="seq", organism="e_coli",
                     max_retries=3, generate_pdf=False, report_dir="reports_idt"):
    """
    Optimiza una secuencia de aminoácidos para síntesis en IDT.

    Intenta hasta max_retries veces con distintas semillas aleatorias por si
    el solver queda atrapado en un mínimo local. Devuelve el primer resultado
    válido o un dict con 'error' si ningún intento converge.
    """
    if dc is None or get_codons_table is None:
        _load_deps()
    aa_seq  = aa_seq.upper().strip().rstrip("*")
    if not aa_seq:
        return {"dna": None, "error": "Secuencia de aminoácidos vacía."}
    species = ORGANISMS.get(organism, organism)
    min_gc_val = _min_gc(aa_seq, species)
    bp = len(aa_seq) * 3

    # Techos de GC: 60% global y 65% local (margen bajo los umbrales IDT)
    # Se relajan si la composición de la secuencia lo requiere
    gc_max       = max(0.60, min_gc_val + 0.04)
    gc_local_max = max(0.65, min_gc_val + 0.08)

    constraints = [
        dc.EnforceTranslation(),
        dc.EnforceGCContent(mini=max(0.35, min_gc_val), maxi=gc_max),
        dc.AvoidPattern("6xA"),
        dc.AvoidPattern("6xT"),
        dc.AvoidPattern("6xG"),
        dc.AvoidPattern("6xC"),
        # Ningún 9-mer puede aparecer dos veces en la secuencia completa.
        # Más estricto que UniquifyAllKmers(k=8) como objetivo: aquí es obligatorio.
        dc.AvoidPattern(dc.RepeatedKmerPattern(n_repeats=2, kmer_size=9)),
    ]
    # Ventana de GC local solo si la secuencia es suficientemente larga
    if bp >= 100:
        constraints.append(dc.EnforceGCContent(mini=0.25, maxi=gc_local_max, window=100))
    elif bp >= 50:
        constraints.append(dc.EnforceGCContent(mini=0.25, maxi=gc_local_max, window=50))

    for attempt in range(max_retries):
        random.seed(42 + attempt * 7)

        problem = dc.DnaOptimizationProblem(
            sequence=dc.reverse_translate(aa_seq),
            constraints=constraints,
            objectives=[
                # match_codon_usage distribuye entre codones sinónimos según frecuencia,
                # lo que reduce la aparición de codones idénticos y por tanto repeticiones
                dc.CodonOptimize(species=species, method="match_codon_usage"),
                # Penaliza 8-mers repetidos restantes (los de 9+ ya están prohibidos)
                dc.UniquifyAllKmers(k=8, boost=5.0),
            ],
        )
        problem.max_random_iters = 5000

        try:
            if generate_pdf:
                rep_path = Path(report_dir)
                rep_path.mkdir(exist_ok=True, parents=True)
                problem.optimize_with_report(
                    target=str(rep_path / f"{seq_name}_optimization.pdf"))
            else:
                problem.resolve_constraints()
                problem.optimize()
        except dc.NoSolutionError:
            continue
        except Exception as e:
            return {"dna": None, "error": str(e)}

        dna = problem.sequence
        gc  = (dna.count("G") + dna.count("C")) / len(dna)
        return {
            "dna":            dna,
            "gc_pct":         round(gc * 100, 1),
            "gc_max_applied": round(gc_max * 100, 1),
            "min_gc_pct":     round(min_gc_val * 100, 1),
            "length_bp":      len(dna),
            "translation_ok": dc.translate(dna) == aa_seq,
            "error":          None,
        }

    return {
        "dna": None,
        "error": (f"Sin solución tras {max_retries} intentos. "
                  f"GC mínimo posible: {min_gc_val:.1%}. "
                  f"Prueba --max_retries mayor o revisa la secuencia."),
    }


def _read_input(path, seq_col, name_col):
    """Lee secuencias desde FASTA (.fa/.fasta/.faa), CSV o TSV."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".fa", ".fasta", ".faa"):
        entries, name, parts = [], None, []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if name:
                        entries.append((name, "".join(parts)))
                    name, parts = line[1:].split()[0], []
                elif line:
                    parts.append(line.upper())
        if name:
            entries.append((name, "".join(parts)))
        return entries
    if suffix in (".csv", ".tsv"):
        delimiter = '\t' if suffix == '.tsv' else ','
        with open(p) as f:
            return [
                (row.get(name_col, f"seq_{i}"), row.get(seq_col, "").strip().upper())
                for i, row in enumerate(csv.DictReader(f, delimiter=delimiter))
                if row.get(seq_col, "").strip()
            ]
    raise ValueError(f"Formato no reconocido: {p.suffix}. Usa .fasta, .csv o .tsv")


def main():
    parser = argparse.ArgumentParser(description="Optimizador de codones para IDT (binderlab dna)")
    parser.add_argument("-i", "--input",    help="FASTA o CSV de entrada")
    parser.add_argument("-o", "--output",   default="optimized", help="Prefijo de salida")
    parser.add_argument("--organism",       default="e_coli", choices=list(ORGANISMS))
    parser.add_argument("--seq_col",        default="sequence",    help="Columna de secuencia (CSV)")
    parser.add_argument("--name_col",       default="description", help="Columna de nombre (CSV)")
    parser.add_argument("--max_retries",    type=int, default=3,   help="Reintentos si no converge")
    parser.add_argument("--single",         metavar="AA_SEQ",      help="Optimizar una secuencia directamente")
    parser.add_argument("--reports",        action="store_true",   help="Generar PDF por secuencia (requiere dnachisel[reports])")
    args = parser.parse_args()
    _load_deps()

    print(f"[*] Organismo: {args.organism} ({ORGANISMS[args.organism]})")

    if args.single:
        r = optimize_for_idt(args.single, organism=args.organism,
                             max_retries=args.max_retries, generate_pdf=args.reports)
        if r["error"]:
            print(f"[!] {r['error']}")
        else:
            print(f"    GC {r['gc_pct']}%  |  {r['length_bp']} bp  |  traducción: {'OK' if r['translation_ok'] else 'ERROR'}")
            for i in range(0, len(r["dna"]), 60):
                print(f"    {r['dna'][i:i+60]}")
        return

    if not args.input:
        sys.exit("Especifica -i INPUT o --single AA_SEQ")

    sequences = _read_input(args.input, args.seq_col, args.name_col)
    if not sequences:
        sys.exit(f"[!] No se encontraron secuencias en {args.input}. "
                 f"Revisa el formato o las columnas --seq_col/--name_col.")
    print(f"[*] {len(sequences)} secuencias...")
    if args.reports:
        print("[*] Reportes PDF → ./reports_idt/")

    fasta_rows, csv_rows = [], []
    for i, (name, aa) in enumerate(sequences, 1):
        print(f"  [{i:>4}/{len(sequences)}] {name[:40]:<40}", end=" ", flush=True)
        r = optimize_for_idt(aa, seq_name=name, organism=args.organism,
                             max_retries=args.max_retries, generate_pdf=args.reports)
        if r["error"]:
            print(f"ERROR: {r['error'][:60]}")
        else:
            print(f"GC {r['gc_pct']}%  (máx {r['gc_max_applied']}%, mín posible {r['min_gc_pct']}%)")
            fasta_rows.append((f"{name}_opt", r["dna"]))
        csv_rows.append({"name": name, "aa_sequence": aa, **{k: r.get(k, "") for k in
            ["dna", "gc_pct", "gc_max_applied", "min_gc_pct", "length_bp", "translation_ok", "error"]}})

    if fasta_rows:
        with open(f"{args.output}.fasta", "w") as f:
            for name, seq in fasta_rows:
                f.write(f">{name}\n")
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + "\n")

    with open(f"{args.output}_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    errors = sum(1 for r in csv_rows if r.get("error"))
    print(f"\n[+] {len(fasta_rows)}/{len(sequences)} OK  "
          f"({errors} errores)  →  {args.output}.fasta  |  {args.output}_report.csv")


if __name__ == "__main__":
    main()
