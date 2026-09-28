#!/bin/bash

set -e

REMOTE="ds541@login.cs.duke.edu"
REMOTE_REPO="/usr/xtmp/ds541/E_min"
SUITE="results/ppo/sweeps"
SUITE_NAME="suite_e_variants_12345"

ssh "$REMOTE" \
    "cd $REMOTE_REPO && python scripts/generate_e_variants_suite_pdf.py --suite-dir $SUITE/$SUITE_NAME"

scp "$REMOTE:$REMOTE_REPO/$SUITE/$SUITE_NAME/e_variants_suite_comparison.pdf" .