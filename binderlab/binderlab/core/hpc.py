"""Estado del cluster HPC (SLURM y GPUs).

Modulo puro: no depende de la configuracion del pipeline.
"""
import os
import time
import subprocess

_HPC_CACHE = {'time': 0, 'source': 'N/A', 'content': 'Cargando...'}
HPC_TTL_SECONDS = 30


def get_hpc_status():
    """Estado del HPC. Detecta SLURM (vía `squeue`) o GPUs locales (vía
    `nvidia-smi`). Cachea el resultado durante HPC_TTL_SECONDS para no
    machacar al demonio del cluster con el polling automático.

    Devuelve (fuente, contenido):
      fuente   = "SLURM" | "GPU" | "N/A"
      contenido = la salida del comando, ya formateada como string.
    """
    now = time.time()
    if now - _HPC_CACHE['time'] < HPC_TTL_SECONDS:
        return _HPC_CACHE['source'], _HPC_CACHE['content']

    user = os.environ.get('USER', '') or os.environ.get('LOGNAME', '')

    # Intento 1: SLURM
    try:
        if subprocess.run(['which', 'squeue'],
                          capture_output=True, timeout=2).returncode == 0:
            r = subprocess.run(['squeue', '-u', user] if user else ['squeue'],
                               capture_output=True, text=True, timeout=5)
            content = r.stdout.strip() or "Sin jobs en cola."
            _HPC_CACHE.update({'time': now, 'source': 'SLURM',
                               'content': content})
            return 'SLURM', content
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Intento 2: nvidia-smi local
    try:
        if subprocess.run(['which', 'nvidia-smi'],
                          capture_output=True, timeout=2).returncode == 0:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            content = r.stdout.strip() or "No hay GPUs visibles."
            _HPC_CACHE.update({'time': now, 'source': 'GPU',
                               'content': content})
            return 'GPU', content
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    msg = "Estado HPC no disponible (ni SLURM ni nvidia-smi accesibles)."
    _HPC_CACHE.update({'time': now, 'source': 'N/A', 'content': msg})
    return 'N/A', msg
