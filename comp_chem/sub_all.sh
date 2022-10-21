#!/bin/bash
#SBATCH -A che220011p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J LA_calcs

module load anaconda3
module load orca
conda activate bayes_opt

python calc.py monomers 32 > monomers.out &
python calc.py dimers 32 > dimers.out &
python calc.py monomer_complexes 32 > monomer_complexes.out &
python calc.py dimer_complexes 32 > dimer_complexes.out &

wait

