#BSUB -W 1200                                                                                                                                                                         
#BSUB -n 10                                                                                                                                                                         
#BSUB -R "span[hosts=1]"     
#BSUB -eo ~/abundance_matching/out_files/temp.err                                                                                                                                                     
#BSUB -oo ~/abundance_matching/out_files/temp.out                                                                                                                     
                                                                                                                                                                                     
source ~/.bashrc
cd ~/abundance_matching/am_abundance_matching
python emcee_script.py