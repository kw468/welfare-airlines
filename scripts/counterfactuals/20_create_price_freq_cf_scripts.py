"""
    Create sh programs to run sim_pf on SLURM.
--------------------------------------------------------------------------------
change log:
    v0.0.1  Thurs 24 Dec 2020
-------------------------------------------------------------------------------
notes:
    sim_pf was renamed to price_freq_counterfactual.
--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2020 Yale University
"""

mkts = [
    "BIL_SEA", "BOI_PDX", "BZN_PDX", "CHS_SEA", "CMH_SEA", "FAT_PDX",
    "GEG_PDX", "GTF_SEA", "HLN_SEA", "ICT_SEA", "LIH_PDX", "MSO_PDX",
    "OKC_SEA", "OMA_SEA", "PDX_PSP", "PDX_RNO", "PDX_SBA", "PDX_SMF",
    "PDX_STS", "SBA_SEA", "SEA_STS", "SEA_SUN"
]
mkts2 = ["AUS_BOS", "BOS_JAX", "BOS_MCI", "BOS_PDX", "BOS_SAN", "BOS_SEA"]
mkts = mkts + mkts2
# outputs the shell script
for i in mkts:
    script = f"""#!/bin/bash
#SBATCH --job-name={i}_pf_cf
#SBATCH --output={i}_pf_cf.out
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=42
cd /gpfs/home/kw468/airlines_jmp/scripts/

module load anaconda3-2020.07-gcc-9.3.0-myrjwlf
module load cuda-11.1.0-gcc-9.3.0-qevsggz
module load py-absl-py-0.7.1-gcc-9.3.0-cdmkucu
module load py-grpcio-1.32.0-gcc-9.3.0-yrihpv2
module load py-pkgconfig-1.5.1-gcc-9.3.0-nhjbazn
module load py-termcolor-1.1.0-gcc-9.3.0-dklapo7
module load py-astunparse-1.6.3-gcc-9.3.0-u74hse7
module load py-protobuf-3.12.2-gcc-9.3.0-7f56goi
module load py-wheel-0.34.2-gcc-9.3.0-rs3zpis
module load py-cached-property-1.5.2-gcc-9.3.0-b7oihyt
module load py-keras-preprocessing-1.1.2-gcc-9.3.0-nw44lcq
module load py-pybind11-2.5.0-gcc-9.3.0-7fjy4zv
module load py-wrapt-1.11.2-gcc-9.3.0-2cekh2a
module load py-cython-0.29.21-gcc-9.3.0-p3orb4j
module load py-mpi4py-3.0.3-gcc-9.3.0-pj765ay
module load py-scipy-1.4.1-gcc-9.3.0-5kslvi5
module load py-docutils-0.15.2-gcc-9.3.0-6bpzrrl
module load py-nose-1.3.7-gcc-9.3.0-2djljpw
module load py-setuptools-50.3.2-gcc-9.3.0-iwxzxh7
module load py-gast-0.3.3-gcc-9.3.0-ugkpf4l
module load py-six-1.14.0-gcc-9.3.0-vdazevs
module load py-google-pasta-0.1.8-gcc-9.3.0-qj4bd2x
module load py-opt-einsum-3.2.1-gcc-9.3.0-khrhoan
module load py-tensorflow-2.3.1-gcc-9.3.0-sq3ajne
module load py3-mpi4py/3.0.3
module load py-keras-preprocessing-1.1.2-gcc-9.3.0-nw44lcq
module load py-keras-applications-1.0.8-gcc-9.3.0-34hecz2
module load py-keras-2.2.4-gcc-9.3.0-hivyubv
module load py-jax-0.2.9-gcc-9.3.0-hcbn53p
module load py-jaxlib-0.1.59-gcc-9.3.0-kqzodnr
module load py-pyarrow-0.17.1-gcc-9.3.0-gp34oiu
module load py-python-snappy-0.6.0-gcc-9.3.0-2xpqbto
python -u sim_pf.py {i}
"""
    f = open("runPF" + i + ".sh", "w")
    f.write(script)
    f.close()

# outputs the shell script
