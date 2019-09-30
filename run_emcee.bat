#BSUB -W 600                                                                                                                                                                         
#BSUB -n 6                                                                                                                                                                         
#BSUB -R "span[hosts=1]"     
#BSUB -e /u/ki/swagnerc/abundance_matching/out_files/temp.err                                                                                                                                                     
#BSUB -o /u/ki/swagnerc/abundance_matching/out_files/temp.out                                                                                                                     
                                                                                                                                                                                     
source /u/ki/swagnerc/.bashrc
cd /u/ki/swagnerc/abundance_matching/am_abundance_matching
python emcee_script.py