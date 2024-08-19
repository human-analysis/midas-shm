# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : MIDAS-shm
# //   File        : config.py
# //   Description : Configuration file
# //
# //   Created On: 8/19/2024
# /////////////////////////////////////////////////////////////////

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', default='bc', required=False, type=str, help='choose which experiment')
    parser.add_argument('--is_bar', action='store_true', help='if localization contour outputs colorbar')
    parser.add_argument('--is_save', action='store_false', dest='is_save', help='if localization contour saves figure')
    parser.add_argument('--lr', default=0.5e-3, type=float, help='learning rate')
    parser.add_argument('--net_size', default=[8,16,32,16,8], help='network size')
    parser.add_argument('--seed', default=10, help='torch random seed')
    parser.add_argument('--epochs', default=200, help='number of training epoch')
    args = parser.parse_args()
    
    args.interpolate_function = 'gaussian' # multiquadric, linear, gaussian, inverse
    args.interpolate_alg = lambda x: x
    args.alpha = 0.5 # parameter balance the mu/sigma data contribution, denoted as lambda in the article
    args.sensor_percentage = 0.08 # threshold, min # of positve pred sensors indicating damage
    args.nu = 0.05 #coefficient for 'OCSVM'
    args.contamination = 0.06 #coefficient for 'IF'
    
    # configure different paramters for different experiment
    if args.experiment =='crack':
        args.all_sensors = 26  # number of all sensors
        args.fpr_target = 0.01 # false positive rate threshold
        args.h_dim = 8
    elif args.experiment =='bc':
        args.all_sensors = 26
        args.fpr_target = 0.01 #0.01
        args.h_dim = 8
    elif args.experiment =='simu':
        args.all_sensors = 45
        args.fpr_target = 0.015
        args.cracks_sizes = [0.008, 0.02, 0.04]
        args.h_dim = 20
    elif args.experiment == 'simu_noise':
        args.all_sensors = 45
        args.fpr_target = 0.005
        args.cracks_sizes = [0.008, 0.02, 0.04]
        args.nu = 0.1
    if args.experiment == 'simu_temp':
        args.all_sensors = 45
        args.net_size = [4, 8, 16, 8, 4]
        args.fpr_target = 0.03
        args.cracks_sizes = [0.008, 0.02, 0.04]
        args.lr = 0.2e-3
    elif args.experiment == 'beam_column':
        args.all_sensors = 8
        args.fpr_target = 0.026
        args.nu = 0.03
        args.contamination = 0.04
    return args

