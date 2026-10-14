#!/usr/bin/env bash
set -euo pipefail

# The providers require authentication and/or manual license acceptance, so
# this helper prints the official sources instead of silently downloading data.

cat <<'EOF'
MorphoINR data sources

Human prefrontal cortex (Globus):
  https://app.globus.org/file-manager?origin_id=abdefb8c-731b-4907-a89d-9b5649d9967a&origin_path=%2F&two_pane=false

Human breast cancer (Zenodo):
  https://zenodo.org/records/4739739

After downloading and preprocessing, place each file at:
  data/raw/<case>/<case>_with_phikon.h5ad

See data/README.md for required AnnData keys and licensing notes.
EOF
