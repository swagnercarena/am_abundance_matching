#BSUB -W 600                                                                                                                                                                         
#BSUB -n 6                                                                                                                                                                         
#BSUB -R "span[hosts=1]"     
#BSUB -e ~/abundance_matching/out_files/temp.err                                                                                                                                                     
#BSUB -o ~/abundance_matching/out_files/temp.out                                                                                                                     
                                                                                                                                                                                     
source ~/.bashrc
cd ~/abundance_matching/am_abundance_matching
python emcee_script.py