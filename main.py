# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : MIDAS-shm
# //   File        : main.py
# //   Description : Main file
# //
# //   Created On: 8/19/2024
# /////////////////////////////////////////////////////////////////

from config import parse_args
import utils, plots
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

'''    methods:
MIAE   = Mechanics-informed autoencoder (ours)
AE     = Autoencoder
OCSVM  = One class SVM
LODA   = Lightweight on-line detector of anomalies. https://anlearn.readthedocs.io/en/latest/loda.html
IF     = Isolation forest
SPIRIT = Streaming Pattern Discovery in Multiple Time-Series
'''

def main():
    try:
        plt.style.use(['science', 'nature'])
    except:
        print("Warning: science plot style not found. Default plot style will be used.")

    detection_methods = ['AE', 'MIAE', 'OCSVM', 'LODA', 'IF']
    localization_methods = ['SPIRIT', 'AE', 'MIAE']
    
    args = parse_args()
    ##############################################################
    ##############################################################
    if args.experiment == 'simu':
        args.sensor_index = np.arange(args.all_sensors)
        data = utils.load_update_args(args)
        
        # ======= fig2e detection
        utils.detection_simu_progress(data, args)  #ae vs mechae for crack propagation detection 
        
        # ======= fig2f detection
        for method in detection_methods: # detection_methods for 37 cases
            utils.save_detection_metric_all_case(method, data, args)
        plots.detection_simu(args)
        
        # ======= fig3a localization
        args.is_bar = False # for adding the color bar, default off
        for method in localization_methods:
            utils.localization_simu(method, data, args, case=22, group_fig=False, crack_text=False)
        
        # ======= fig3b localization
        plots.localization_simu_progress()
        
        ##############################################################
        # ======= fig4a detection
        n_sensor_list = [4,6,8,10,16] # variable number of sensors
        metrics = utils.detection_simu_fewer_sensors_metric(args, n_sensor_list, evaluate=False)
        plots.detection_simu_fewer_sensors(metrics, n_sensor_list)
        
        # ======= fig4b-c localization
        args.sensor_index = np.array([9, 30, 13, 34]) - 1 # choose 4 sensors
        data = utils.load_update_args(args)
        
        # ======= fig4b localization map
        args.visual = True
        utils.localization_fewer_sensors('AE', args, data, cases=[11,22]) 
        utils.localization_fewer_sensors('MIAE', args, data, cases=[11,22])
        
        # ======= fig4c localization progress
        args.cracks_sizes = [0.008, 0.015, 0.02, 0.03, 0.04]
        args.visual = False # used in calculating distance
        dist_metric = np.stack([utils.localization_fewer_sensors(method, args, data) for method in localization_methods])
        plots.localization_simu_fewer_sensors_progress(args, dist_metric)
        
        
    ##############################################################
    ##############################################################
    if args.experiment == 'simu_noise': # fewer sensors
        args.sensor_index = np.array([9,13,34,30])-1
        data = utils.load_update_args(args)
        
        # ======= fig4d detection
        for method in detection_methods: # detection_methods for 37 cases
            utils.save_detection_metric_all_case(method, data, args)
        plots.detection_simu_noise(args, damage_level=3) # uses the largest crack
    
    
    ##############################################################
    ##############################################################
    if args.experiment == 'simu_temp': # fewer sensors
        args.sensor_index = np.array([13,34,9,30])-1
        data = utils.load_update_args(args)
        
        # ====== fig4e detection
        for args.temp in [10,13]:
            for method_name in detection_methods: # test both undamaged/damaged cases
                args.test_damage = False
                utils.save_detection_metric_all_case(method_name, data, args)
                args.test_damage = True
                utils.save_detection_metric_all_case(method_name, data, args)
        plots.detection_simu_temp([10,13,10,13], args)
    
    
    ##############################################################
    ##############################################################
    if args.experiment in ['bc', 'crack']: # crack and boundary condition variation (bc)
        args.sensor_index = np.arange(args.all_sensors)
        data = utils.load_update_args(args)
        
        # ====== localization
        args.is_bar = False
        for method in localization_methods:
            utils.localization_crack_bc(method, data, args)
        
        ##############################################################
        args.sensor_index = np.array([1,5,16,20])-1  # fewer sensors
        data = utils.load_update_args(args)
        
        # ====== fig5b-c localization
        args.visual = True
        for method in localization_methods: # localization
            utils.localization_fewer_sensors(method, args, data)
    
    
    ##############################################################
    ##############################################################
    if args.experiment =='beam_column':
        args.sensor_index = np.arange(args.all_sensors)
        data = utils.load_update_args(args)
        
        # ====== fig6c detection
        for method in detection_methods:
            utils.save_detection_metric_beam_column(method, data, args)
        plots.detection_beam_column(args)
    
        # ====== fig6d localization
        metrics = [utils.localization_damage_score_beam_column(method, args) for method in localization_methods[1:3] ]
        utils.localization_beam_column(args, metrics)
        
        ##############################################################
        args.sensor_index = np.array([4,3,2,1])-1 # fewer sensors
        data = utils.load_update_args(args)
        
        # ====== fig6e localization
        metrics = [utils.localization_damage_score_beam_column(method, args) for method in localization_methods[1:3] ]
        utils.localization_beam_column(args, metrics)
    
if __name__ == "__main__":
    main()

