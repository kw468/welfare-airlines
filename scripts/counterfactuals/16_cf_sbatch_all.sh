#!/bin/sh
"""
    Run the sh programs to run sim_cf on SLURM.
--------------------------------------------------------------------------------
change log:
    v0.0.1  Fri 25 Jun 2021
-------------------------------------------------------------------------------
notes:

--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""

for market in "BIL_SEA" "BOI_PDX" "BZN_PDX" "CHS_SEA" "CMH_SEA" "FAT_PDX" \
    "GEG_PDX" "GTF_SEA" "HLN_SEA" "ICT_SEA" "MSO_PDX" "OKC_SEA" \
    "PDX_RNO" "PDX_SBA" "PDX_STS" "SBA_SEA" "SEA_STS" "SEA_SUN" \
    "BOS_SAN" "BOS_MCI" "BOS_JAX" "AUS_BOS"
do
    sbatch runCF${market}.sh
    sleep 5
done
