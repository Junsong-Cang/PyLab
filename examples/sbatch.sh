#!/bin/bash
#SBATCH --job-name=FishM
#SBATCH --time=96:00:00
#SBATCH --output=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/DataSet/Run_Fisher_DataSet.out
#SBATCH --error=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/DataSet/Run_Fisher_DataSet.err
#SBATCH --nodes=2
#SBATCH --mem=220GB
#SBATCH --ntasks-per-node=62
#SBATCH --cpus-per-task=1
#SBATCH --partition=ali

mpirun --map-by ppr:62:node /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs//soft/anaconda3/envs/SDM/bin/python3 /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/DataSet/Run_Fisher_DataSet.py
