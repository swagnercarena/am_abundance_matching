#BSUB -W 1200
#BSUB -n 10
#BSUB -R "span[hosts=1]"     
#BSUB -eo /u/ki/swagnerc/abundance_matching/out_files/temp.er
#BSUB -oo /u/ki/swagnerc/abundance_matching/out_files/temp.out

source ~/.bashrc
cd ~/abundance_matching/am_abundance_matching
python emcee_script.py