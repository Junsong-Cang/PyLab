#!/bin/bash
#SBATCH --job-name=arcade   ##my job name
#SBATCH --time=1:00:00     ##time for the job, two days is maximum
#SBATCH --output=/ceph/hpc/home/eujunsongc/tmp/arcade_.out    #output file
#SBATCH --error=/ceph/hpc/home/eujunsongc/tmp/arcade_.err      #error file
                                                                                                                                                                     
#SBATCH --nodes=2            ##any number of nodes
###SBATCH --mem=48GB # commentted out with ##, not important, anything with ## are commented out
#SBATCH --ntasks-per-node=256   #number of tasks per node, should be the same as mpirun -np XXX
#SBATCH --cpus-per-task=1    #number of cpus per task, should be the same as N_THREADS setting in p21c

eval "$(conda shell.bash hook)"
conda activate 21cmfast       #conda environment, conda activate may fail, replace conda with source

## set up env variables, can also source bashrc but that may cause namespace polution
source /ceph/hpc/home/eujunsongc/intel/oneapi/setvars.sh
source /ceph/hpc/home/eujunsongc/intel/oneapi/compiler/2023.2.0/env/vars.sh
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/ceph/hpc/home/eujunsongc/intel/oneapi/mkl/2023.2.0/include/fftw
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/ceph/hpc/home/eujunsongc/miniconda3/envs/21cmfast/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ceph/hpc/home/eujunsongc/miniconda3/envs/21cmfast/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ceph/hpc/home/eujunsongc/soft/MultiNest/lib

# mpirun --mca plm_rsh_no_tree_spawn true --map-by ppr:7:node -x PATH -x LD_LIBRARY_PATH python $HOME/programs/database/run_single.py --output_dir='/ceph/hpc/home/euivann/project/inikolic/database/output/' --cache_dir='/ceph/hpc/home/euivann/project/inikolic/database/_cache/' --threads=42 --params_dir='/ceph/hpc/home/euivann/'

# might need to replace ipython with python or python3
# don't always need both np and --map-by, use --map-by if using more than 1 node
# numbers after --map-by is the same as that follows -np
## mpirun --map-by ppr:32:node -x PYTHONPATH -x LD_LIBRARY_PATH -x C_INCLUDE_PATH ipython /ceph/hpc/home/eujunsongc/work/Radio_Excess_EDGES/Pop_III/29_MCG_HII_50.py
## -x option might fail

mpirun --map-by ppr:256:node python /ceph/hpc/home/eujunsongc/soft/PyLab/examples/example_multinest.py
## mpirun --map-by ppr:256:node ipython /ceph/hpc/home/eujunsongc/soft/PyLab/examples/example_multinest.py
## try mpiexec if mpirun fails
## mpiexec --map-by ppr:256:node ipython /ceph/hpc/home/eujunsongc/soft/PyLab/examples/example_multinest.py
## mpiexec -n 256 ipython /ceph/hpc/home/eujunsongc/soft/PyLab/examples/example_multinest.py

## When all is ready, submit job using sbatch example_sbatch.sh