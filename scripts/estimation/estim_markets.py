"""
    This script stores the markets in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
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

mkts =  [
    "BIL_SEA", "BOI_PDX", "BZN_PDX", "FAT_PDX", "GEG_PDX", "GTF_SEA",
    "HLN_SEA", "ICT_SEA", "MSO_PDX", "OKC_SEA", "CHS_SEA", "CMH_SEA",
    "PDX_RNO", "PDX_SBA", "PDX_STS", "SBA_SEA", "SEA_STS", "SEA_SUN"
]
mkts2 = ["AUS_BOS", "BOS_JAX", "BOS_SAN", "BOS_MCI"]
mkts = mkts + mkts2
mkts = sorted(mkts)