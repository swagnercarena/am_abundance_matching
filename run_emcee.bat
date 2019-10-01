#BSUB -W 1200
#BSUB -n 10
#BSUB -R "span[hosts=1]"     
#BSUB -eo /u/ki/swagnerc/abundance_matching/out_files/temp.err
#BSUB -oo /u/ki/swagnerc/abundance_matching/out_files/temp.out
pwd
source /afs/slac.stanford.edu/u/ki/swagnerc/bsub_startup.txt
cd /afs/slac.stanford.edu/u/ki/swagnerc/abundance_matching/am_abundance_matching
python emcee_script.py