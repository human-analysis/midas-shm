# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : Mechanics-AE for damage detection and localization
# //   File        : config.py
# //   Description : Configuration file
# //
# //   Created On: 3/7/2024
# /////////////////////////////////////////////////////////////////


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', default='simu', required=False, type=str, help='choose which experiment')
    parser.add_argument('--is_bar', action='store_true', help='if localization contour outputs colorbar')
    parser.add_argument('--is_save', action='store_false', dest='is_save', help='if localization contour saves figure')
    args = parser.parse_args()
    
    # configure different paramters for different experiment
    if args.experiment =='crack':
        args.alpha = 0.5 # parameter balance the mu/sigma data contribution, denoted as lambda in the article
        args.sensor_percentage = 0.08 # threshold, min # of positve pred sensors indicating damage
        args.fpr_target = 0.01 # false positive rate threshold, for ae & mechae detection metric
        args.n_sensors = 26 # number of available sensors
        args.h_dim = 8 # size of the network
    elif args.experiment =='bc':
        args.alpha = 0.8
        args.sensor_percentage = 0.08
        args.fpr_target = 0.01
        args.n_sensors = 26
        args.h_dim = 8
    elif args.experiment =='simu':
        args.alpha = 0.5
        args.n_sensors = 45
        args.h_dim = 20
        args.sensor_percentage = 0.08
        args.fpr_target = 0.055
    return args

