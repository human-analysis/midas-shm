# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : Mechanics-AE for damage detection and localization
# //   File        : main.py
# //   Description : Main file to run detection/localization for all experiments
# //
# //   Created On: 3/7/2024
# /////////////////////////////////////////////////////////////////

from plots import compare_simu_detection_plot, localization_progress, crack_bc_detection_plot
from config import parse_args
import utils
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


'''    methods:
ae     = Autoencoder
mechae = Mechanics-AE
OCSVM  = One class SVM
LODA   = Lightweight on-line detector of anomalies. https://anlearn.readthedocs.io/en/latest/loda.html
IF     = Isolation forest
SPIRIT = Streaming Pattern Discovery in Multiple Time-Series
'''

def main():
    # parse the arguments
    args = parse_args()
    
    detection_methods = ['ae', 'mechae', 'OCSVM', 'LODA', 'IF'] 
    localization_methods = ['ae', 'mechae', 'SPIRIT']           
    
    # ================= For simulation
    if args.experiment == 'simu':
        print('------conduct simulation\n')
        data = utils.load_simu_data(args)
    
        utils.compare_mechae_ae_simu_detection(data, args)  #ae vs mechae for crack propagation detection 
        
        for method in detection_methods: # detection_methods for 37 cases
            utils.save_detection_metric_all_case(method, data, args)
        compare_simu_detection_plot()
         
        for method in localization_methods: ## localization for case 22
            utils.localization_simu(method, data, args, case=22)
            
        ## 'mechae' localization for all other cases
        for case in range(1,37+1): # generate localization figures for each case
            utils.localization_simu('mechae', data, args, case=case)
        ## Note: localization are manually inspected & summarized in a excel file
        localization_progress()
    
    # ================= For crack and boundary condition variation (bc)
    if args.experiment !='simu':
        print(f'------conduct {args.experiment} experiment\n')
        data = utils.load_undamage_data(args)#[50:,:,:]

        for method in detection_methods: # detection
            utils.detection_crack_bc(method, data, args)
        # The plot generates results for both crack and bc
        crack_bc_detection_plot()
        
        for method in localization_methods: # localization
            utils.localization_crack_bc(method, data, args)
        
    print('------evaluation done\n')
    




if __name__ == "__main__":
    main()

