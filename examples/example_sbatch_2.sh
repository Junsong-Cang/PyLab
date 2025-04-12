#!/bin/bash                                                                                                                        
#SBATCH --job-name=SKA
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fswork/projects/rech/uzf/umj75ub/Project/SKA_DC/test_p21c/run_py21cmfast.out
#SBATCH --error=/lustre/fswork/projects/rech/uzf/umj75ub/Project/SKA_DC/test_p21c/run_py21cmfast.err
#SBATCH --nodes=1
###SBATCH --mem=48GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -A uzf@cpu
                                                                                                         
#eval "$(conda shell.bash hook)"
#conda activate base

ipython /lustre/fswork/projects/rech/uzf/umj75ub/Project/SKA_DC/test_p21c/run_py21cmfast.py
