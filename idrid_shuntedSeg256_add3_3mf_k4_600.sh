#!/bin/bash
#SBATCH --account=osivaz
#SBATCH --time=24:00:00
#SBATCH --job-name=idrid_shuntedSeg256_add3_3mf_k4_600
#SBATCH --output idrid_shuntedSeg256_add3_3mf_k4_600.out     

#SBATCH --partition=akya-cuda
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

export PYTHONNOUSERSITE=True
echo "We have the modules: $(module list 2>&1)" > ${SLURM_JOB_ID}.info

### jobs
apptainer exec --nv --cleanenv /arf/home/osivaz/container-user/miniconda3-user.sif bash -c "source /.singularity.d/env/90-environment.sh && conda activate osivaz61Env && python idrid_shuntedSeg256_add3_3mf_k4_600.py"
exit