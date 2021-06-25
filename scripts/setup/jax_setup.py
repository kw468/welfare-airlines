"""
    This script stores the setup parameters for jax use in generating estimations
    for "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
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

NUM_THREADS = 21
# adjust Jax to 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)
