#!/bin/bash
# Generate all cross-validation ground truth from R
echo "Generating cross-validation data from R..."
Rscript scripts/generate_cross_val_data.R
echo "Done."
