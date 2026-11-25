"""Estado del cluster HPC (SLURM y GPUs).

Modulo puro: no depende de la configuracion del pipeline.
"""
import os
import shutil
import subprocess
import time

_HPC_CACHE = {'time': 0, 'source': 'N/A', 'content': 'Cargando...'}
HPC_TTL_SECONDS = 30


def get_hpc_status():
    """Detecta SLURM (`squeue`) o GPUs locales (`nvidia-smi`) con caché TTL."""
    now = time.time()
    if now - _HPC_CACHE['time'] < HPC_TTL_SECONDS:
        return _HPC_CACHE['source'], _HPC_CACHE['content']

    user = os.environ.get('USER', '') or os.environ.get('LOGNAME', '')

    if shutil.which('squeue'):
        try:
            cmd = ['squeue', '-u', user] if user else ['squeue']
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                content = r.stdout.strip() or "Sin jobs en cola."
                _HPC_CACHE.update({'time': now, 'source': 'SLURM', 'content': content})
                return 'SLURM', content
        except (subprocess.TimeoutExpired, OSError):
            pass

    if shutil.which('nvidia-smi'):
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                content = r.stdout.strip() or "No hay GPUs visibles."
                _HPC_CACHE.update({'time': now, 'source': 'GPU', 'content': content})
                return 'GPU', content
        except (subprocess.TimeoutExpired, OSError):
            pass

    msg = "Estado HPC no disponible (ni SLURM ni nvidia-smi accesibles)."
    _HPC_CACHE.update({'time': now, 'source': 'N/A', 'content': msg})
    return 'N/A', msg
