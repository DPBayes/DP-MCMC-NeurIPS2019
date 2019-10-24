This folder contains all the necessacy files to reproduce experiments presented in the paper.

To reproduce the Figures, run script main.py. In the beginning of the file you can adjust, which experiments to run.
NOTE! The approximate X_corr distributions using Seita et al. method we ran manually for different values of C. The original code for Seita et al. implementation should be asked from the authors if it's not directly available e.g. in Github.

For the experiments in supplements, if you want to run experiments from scratch, run first the main.py to produce data for supplement Figures, and then run main_supplement.py.

For the DP-SGLD comparisons run dp_sgld_main.py in dp_sgld/ with seeds defined in seeds.txt.
Then plot the DP-SGLD comparisons run plot_mcmc_vs_sgld.py.
