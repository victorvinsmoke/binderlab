# Presets (`binderlab/presets/`)

JSON que describe la estructura de salida de una campaña.

```
binderlab monitor -d /ruta -c complexa        # resuelve presets/complexa.json
binderlab monitor -d /ruta -c /ruta/mio.json  # o ruta directa
```

Todas las claves son opcionales: las que no pongas quedan con el valor de
`DEFAULT_PIPELINE_STRUCTURE`. Un JSON con una sola clave vale.

## Claves que el codigo consume

| Clave             | Que hace                                              |
|-------------------|-------------------------------------------------------|
| `stages`          | Etapas del embudo: `{key,label,dir,pdb_glob,color}`. Sin esto = RF/MPNN/AF |
| `extra_markers`   | Globs que marcan un directorio como experimento       |
| `metric_patterns` | Globs de tablas de metricas a ofrecer                 |
| `metric_exclude`  | Globs a descartar (RAW_*, transpuestos, timings)      |
| `radar_metrics`   | `metrica: high|low` para el radar                     |
| `*_dir`, `*_suffix`, `log_*` | Ver `DEFAULT_PIPELINE_STRUCTURE`           |

`complexa.json` = ejemplo completo (1 etapa, deteccion por CSV).

## Preset nuevo

Crea `presets/<nombre>.json` con las claves que difieran. Nada mas.
