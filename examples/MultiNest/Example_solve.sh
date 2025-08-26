#!/bin/bash
#SBATCH --job-name=MN
#SBATCH --time=1:00:00
#SBATCH --output=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/soft/PyLab/examples/MultiNest/Example_solve.out
#SBATCH --error=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/soft/PyLab/examples/MultiNest/Example_solve.err
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=ali

mpirun --map-by ppr:4:node /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/soft/anaconda3/envs/p21c/bin/python3 /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/soft/PyLab/examples/MultiNest/Example_solve.py
