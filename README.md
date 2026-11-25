# binderlab

Suite modular de herramientas para el diseño, cribado y monitorización de binders proteicos generados mediante flujos de trabajo basados en RFdiffusion, ProteinMPNN y AlphaFold2.

## Características

- **Monitor (Dashboard web):** Visualización reactiva en tiempo real del progreso de diseño, tablas interactivas de métricas, gráficos radar multicriterio, anotaciones persistentes (SQLite) y visualizador 3D de estructuras y trayectorias de difusión con Mol*.
- **Corr (Correlaciones):** Análisis modular de correlaciones (Spearman, Pearson, Kendall) entre métricas fisicoquímicas y reducción de dimensionalidad (heurísticas, PCA, autoencoders) sobre hotspots y residuos de interfaz.
- **DNA (Optimización de codones):** Optimización para síntesis de fragmentos génicos (IDT gBlocks/eBlocks) mediante DNA Chisel, controlando GC global/local y eliminando repeticiones.
- **Score (Scoring físico):** Evaluación termodinámica de interfaz (ddG, Shape Complementarity, dSASA, Packstat, enlaces de hidrógeno, BUNs y SAP) con PyRosetta utilizando minimización relajada en interfaz.

## Instalación

```bash
git clone https://github.com/victorvinsmoke/binderlab.git
cd binderlab

# Instalación base (dashboard BinderMonitor)
pip install .

# Extras opcionales según necesidades:
pip install ".[dna]"      # Optimización de codones (DNA Chisel)
pip install ".[corr]"     # Análisis de correlaciones (scikit-learn, seaborn)
pip install ".[corr-ae]"  # Autoencoders con TensorFlow
pip install ".[dev]"      # Herramientas de desarrollo (pytest, ruff)
```

> **Nota sobre PyRosetta (`binderlab score`):**  
> PyRosetta requiere una licencia académica o comercial propia y debe instalarse a través de los canales oficiales de RosettaCommons (no disponible en PyPI público). Más información en [https://www.pyrosetta.org](https://www.pyrosetta.org).

## Uso rápido

```bash
# 1. Iniciar el dashboard BinderMonitor
binderlab monitor -d /ruta/al/experimento -c complexa

# 2. Análisis de correlaciones de métricas
binderlab corr -i metricas.csv -c hotspots -t heuristicas pca

# 3. Optimización de codones para IDT
binderlab dna -i binders.fasta -o resultados

# 4. Evaluación física con PyRosetta
binderlab score --pdbdir pdbs/ --hits hits.txt --sc out.sc --out scored.tsv
```

## Estructura del proyecto

```text
binderlab/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── .github/workflows/    # Integración continua (GitHub Actions)
├── tests/                # Pruebas unitarias automatizadas con pytest
└── binderlab/
    ├── cli.py            # Despachador de subcomandos CLI
    ├── monitor.py        # Dashboard interactivo Dash + Mol*
    ├── correlaciones.py  # Análisis de correlaciones y reducción
    ├── aa_to_dna.py      # Optimizador de codones para síntesis IDT
    ├── scoring_fisico.py # Interfaz de scoring con PyRosetta
    ├── _selfcheck.py     # Verificación rápida de entorno
    ├── presets/          # Configuraciones predefinidas (ej. complexa.json)
    └── core/             # Lógica pura desacoplada de la interfaz gráfica
        ├── config.py     # Configuración y Settings del pipeline
        ├── scanner.py    # Escaneo y detección de experimentos
        ├── pdb.py        # Extracción y parsing de modelos PDB
        ├── search.py     # Motor de filtrado y columnas calculadas
        ├── annotations.py# Base de datos SQLite de anotaciones
        └── hpc.py        # Telemetría de SLURM y GPU
```

## Verificación y Tests

Para ejecutar la batería de pruebas:

```bash
# Pruebas con pytest
pytest

# Autocomprobación interna
python -m binderlab._selfcheck
```

## Cita
 
Si utilizas `binderlab` en tu investigación, por favor cita el repositorio:
 
```bibtex
@software{reche2026binderlab,
  author = {Reche Carrascosa, Víctor},
  title = {{binderlab: Suite modular para diseño y monitorización de binders proteicos}},
  year = {2026},
  url = {https://github.com/victorvinsmoke/binderlab}
}
```
 
También puedes consultar los metadatos completos en el archivo `CITATION.cff`.
 
## Licencia
 
Distribuido bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.

