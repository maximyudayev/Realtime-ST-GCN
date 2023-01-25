#!/bin/bash
# Interactive GPU job on a compute node
qsub -I -X -A lp_stadius_fpga_ai -l nodes=1:ppn=18:gpus=2:skylake -l partition=gpu -l pmem=5gb -l qos=debugging -l walltime=30:00 -N interactive_st_gcn_pkummd_sub_rt9_20 -m bae -M maxim.yudayev@kuleuven.be

