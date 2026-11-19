"""Dispatcher de subcomandos. Reenvia a la main() de cada modulo, que conserva
su propio argparse intacto."""
import sys

# Import perezoso: un dep que falte en un modulo no rompe los demas.
_COMMANDS = {
    "monitor": ("binderlab.monitor",       "main"),
    "corr":    ("binderlab.correlaciones", "main"),
    "dna":     ("binderlab.aa_to_dna",     "main"),
    "score":   ("binderlab.scoring_fisico", "main"),
}

_HELP = """binderlab <subcomando> [args...]

  monitor   Dashboard BinderMonitor (Dash/HPC)
  corr      Analisis de correlaciones de metricas
  dna       Optimizador de codones AA -> DNA (IDT)
  score     Scoring fisico (Rosetta/PyRosetta)

Ejemplos:
  binderlab monitor -d /datos/campana -c complexa
  binderlab dna -i binders.csv -o out
  binderlab --help
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return 0
    cmd = argv[0]
    if cmd not in _COMMANDS:
        sys.exit(f"Subcomando desconocido: {cmd!r}. Usa --help.")
    module_name, fn_name = _COMMANDS[cmd]
    mod = __import__(module_name, fromlist=[fn_name])
    fn = getattr(mod, fn_name)
    # El subcomando ve sys.argv sin el nombre del subcomando, asi su argparse
    # funciona igual que cuando era script suelto.
    sys.argv = [f"binderlab {cmd}"] + argv[1:]
    return fn()


if __name__ == "__main__":
    sys.exit(main())
