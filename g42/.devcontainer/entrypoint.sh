#!/usr/bin/env bash
set -euo pipefail

# Always run commands inside the "workshop" env
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
source /usr/local/etc/profile.d/micromamba.sh 2>/dev/null || true

# Activate env (micromamba base image)
micromamba activate workshop

# Helpful runtime info
python -c "import sys; print('Python:', sys.version); print('Executable:', sys.executable)"
python - <<'PY'
try:
    import ESMF, xesmf
    print("ESMF:", getattr(ESMF, "__version__", "OK"))
    print("xESMF:", getattr(xesmf, "__version__", "OK"))
except Exception as e:
    print("ESMF/xESMF import check failed:", e)
PY

exec "$@"
