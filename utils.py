# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : MIDAS for damage detection and localization
# //   File        : utils.py
# //   Description : Utility functions
# //
# //   Created On: 8/17/2024
# /////////////////////////////////////////////////////////////////

import torch
import numpy as np
import scipy.io as spio
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from models import MIAE, AE, MIAE_AS, AE_AS
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.interpolate import Rbf
from matplotlib.path import Path
from matplotlib import patches
from shapely.geometry import Point, Polygon
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from spirit.pipeline import Pipeline, SpiritStage
from sklearn.ensemble import IsolationForest
from sklearn import svm
from anlearn.loda import LODA
from plots import get_detection_metric_data_all
from load_data import load, load_undamage_data, load_damage_data
from geometry_functions import get_geo_info, get_experimental_crack_info, get_simu_crack_info, find_length

def load_update_args(args):
    print(f'------conduct {args.experiment} experiment\n')
    args.data_index = np.concatenate( (args.sensor_index, args.sensor_index+args.all_sensors) )
    args.active_sensors = len(args.sensor_index)
    if args.all_sensors == args.active_sensors and args.experiment != 'beam_column':
        print('------use all sensors...')
        args.dirs = f'saved_models/saved_models-allsensors/{args.experiment}'
    else:
        print('------use fewer sensors...')
        args.dirs = f'saved_models/saved_models-{args.active_sensors}sensors/{args.experiment}-{args.net_size}-{args.sensor_index}-{args.epochs}'
    data = load_undamage_data(args)[:, :, args.data_index]
    return data

def choose_model(method_name, train_data, test_data, args, is_train=False):
    torch.manual_seed(10)
    if len(args.sensor_index) == args.all_sensors and args.experiment in ['crack', 'bc', 'simu']: # all sensors, different setting
        if method_name == 'AE':
            model = AE_AS(args.h_dim, args.experiment)
        elif method_name == 'MIAE':
            model = MIAE_AS(args.h_dim)
            model.get_weight(args) # extra step to load weight function
    elif args.experiment in ['simu', 'simu_temp', 'simu_noise', 'crack', 'bc', 'beam_column']: # simulations
        if method_name == 'AE':
            model = AE(args.sensor_index, args.lr, args.net_size)
        elif method_name == 'MIAE':
            model = MIAE(args.sensor_index, args.lr, args.net_size)
            model.get_weight(args)
    else:
        print('******no matched models******')
    
    # Model setup
    os.makedirs(args.dirs, exist_ok=True)
    model.load_data(train_data, test_data)
    if not is_train:
        logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", log_graph=False)
        trainer = pl.Trainer(accelerator="gpu", devices=1, enable_progress_bar=False, logger=logger)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # get all ckpt filenames and filter with method name 'AE' or 'MIAE'
        checkpoint_files = [file for file in os.listdir(args.dirs) if file.endswith('.ckpt') and file.startswith(method_name)]
        checkpoint_path = os.path.join(args.dirs, checkpoint_files[0])
        if len(args.sensor_index) == args.all_sensors and args.experiment in ['crack', 'bc', 'simu']: # load all sensors with different settings
            model = model.load_from_checkpoint(checkpoint_path, h_dim=args.h_dim, experiment=args.experiment, strict=False, 
                                               map_location=torch.device(device))
        else:
            model = model.load_from_checkpoint(checkpoint_path, sensor_index=args.sensor_index, 
                                               lr=args.lr, net_size=args.net_size, map_location=torch.device(device))
        print(f'\nloading model {checkpoint_files[0]}...')
        model.freeze()
        return model, trainer
    else: #Train the model
        pass
    return model, trainer

def model_test_data(model, trainer, data):
    test_dataloader = DataLoader(data.astype(np.float32), batch_size=1, shuffle=False)
    losses = torch.stack(trainer.predict(model, dataloaders=test_dataloader), dim=0).cpu().detach().numpy()
    test_mseloss = losses[:,0,:]
    test_normloss = losses[:,1,:]
    return test_mseloss, test_normloss

def find_threshold_for_fpr(data, fpr):
    thresholds = np.percentile(data, (1-fpr)*100, axis=0)
    return thresholds
def get_single_decision(score, threshold):
    decision1 = -np.ones( (len(score)) ) # defualt -1
    index = np.array(np.where(score > threshold))
    decision1[index] = 1
    return decision1
def get_decision_labels(label_pred_sample_by_sensor, sensor_percentage):
    num_sensors = label_pred_sample_by_sensor.shape[1] # number of sensor features
    label_pred = -np.ones( (len(label_pred_sample_by_sensor)) ) # assume negatives
    for i in range( len(label_pred_sample_by_sensor) ): # take sample dimension
        label_pred[i] = 1 if np.count_nonzero((label_pred_sample_by_sensor[i,:] == 1)) >= sensor_percentage*num_sensors else -1
    return label_pred

def calculate_detection_metric(args, method_name, score, train_error, label_true, test_damage):
    if method_name in ['OCSVM', 'LODA', 'IF']: # binary classification method, return score and get decision in the next step
        label_pred_sample_by_sensor = score
    elif method_name in ['AE', 'MIAE']:
        thresholds = find_threshold_for_fpr(train_error, args.fpr_target)
        label_pred_sample_by_sensor = np.zeros(score.shape)
        for dm in range(train_error.shape[1]): # number of sensor features or number of sensors*2
            label_pred_sample_by_sensor[:,dm] = get_single_decision(score[:,dm], thresholds[dm])
    # Now we have 2d decisions per sample and per sensor
    # each sample should give one decision, so reduce the sensor dimension by the "sensor_percentage"
    label_pred = get_decision_labels(label_pred_sample_by_sensor, args.sensor_percentage)
    if test_damage:
        metric1 = np.array([accuracy_score(label_true, label_pred), 
                            precision_score(label_true, label_pred), 
                            recall_score(label_true, label_pred), 
                            f1_score(label_true, label_pred), 
                            roc_auc_score(label_true, label_pred)])
    else: # other metrics cannot be calculated, if no positive data
        metric1 = np.array([accuracy_score(label_true, label_pred)])
    return metric1

def get_detection_metrics(args, train_error, test_error, method_name, test_damage=True):
    # Evaluate per sample decision at each sensor, based on the error distribution at that sensor
    score = np.concatenate((train_error, test_error), axis=0) # combine all the samples
    if test_damage:
        label_true = np.concatenate( (-np.ones(len(train_error)), np.ones(len(test_error))), ) # per sample only
        smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=6)
        enn = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=15)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)
        score, label_true = smote_enn.fit_resample(score, label_true)
        
    else: # if test undamaged data, no need for SMOTE due to enough number of samples
        label_true = np.concatenate( (-np.ones(len(train_error)), -np.ones(len(test_error))), )
    
    return calculate_detection_metric(args, method_name, score, train_error, label_true, test_damage)

def get_damage_score(error, error_train, num_sensors, alpha = 0.5):
    max_train1 = np.max(error_train[:num_sensors])
    max_train2 = np.max(error_train[num_sensors:])
    damage_scores = ( alpha*error[:num_sensors]/max_train1 + 
                     (1-alpha)*error[num_sensors:]/max_train2 )/2
    return damage_scores

def detection_simu_progress(data, args):
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    def get_detection_mean_std(method_name, train_data, test_data, args):
        model, trainer = choose_model(method_name, train_data, test_data, args, False)
        # =============== Baseline evaluation
        train_mseloss, _ = model_test_data(model, trainer, train_data)
        test_mseloss, _ = model_test_data(model, trainer, test_data)
        metric0 = get_detection_metrics(args, train_mseloss, test_mseloss, method_name, test_damage=False)

        path = f'data/data-damaged-{args.experiment}/case22-more'
        filenames = os.listdir(path)
        length_list = find_length(filenames)
        lengths = sorted(set(length_list)) # find unique length
        metrics = np.zeros((len(lengths), 5, int(len(length_list)/len(lengths))))
        for i,length in enumerate(lengths):
            indexs = np.where(length_list==length)[0]#[0]
            metric = np.zeros( (5, int(len(length_list)/len(lengths))) )
            for j,index in enumerate(indexs): # check each file
                damage_data = np.array( load( filenames[index], folder=path ).reshape(-1,12,args.all_sensors*2) )
                damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
                metric[:,j] = get_detection_metrics(args, train_mseloss, damage_mseloss, method_name, test_damage=True)
            metrics[i,:,:] = metric
        m_mean = np.mean(metrics,axis=2)
        m_std = np.std(metrics,axis=2)
        return metric0, m_mean, m_std
    
    method_name1, method_name2 = 'AE', 'MIAE' 
    metric01, m1_mean, m1_std = get_detection_mean_std(method_name1, train_data, test_data, args)
    metric02, m2_mean, m2_std = get_detection_mean_std(method_name2, train_data, test_data, args) 
    
    scale = 2
    y2_lower = np.clip(m2_mean - m2_std/scale, 0,1)
    y2_upper = np.clip(m2_mean + m2_std/scale, 0,1)
    y1_lower = np.clip(m1_mean - m1_std/scale, 0,1)
    y1_upper = np.clip(m1_mean + m1_std/scale, 0,1)
    
    c1, c2 = '#1FBF8F', '#DD6CAF'
    alpha = 0.4
    lengths = np.array([0.004, 0.008, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06] )
    xx = lengths*1e2
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUROC']
    lists = [0,1,2,3,4]

    fig, axs = plt.subplots(1, len(lists), figsize=(3.8*len(lists), 3.8))
    for i,index in enumerate(lists):
        plt.subplot(1,len(lists),i+1)
    
        plt.plot(xx, m1_mean[:,index], label='Autoencoder', linewidth=2, marker='^', 
                 markersize=10, color=c1)
        plt.fill_between(xx, y1_lower[:,index], y1_upper[:,index], alpha=alpha, 
                         color=c1)
        
        plt.plot(xx, m2_mean[:,index], label='MIAE', linewidth=2, marker='o', 
                 markersize=10, color=c2)
        plt.fill_between(xx, y2_lower[:,index], y2_upper[:,index], alpha=alpha, 
                         color=c2)
        plt.grid('on')
        plt.title(titles[index], fontsize=18)
        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.12),  
                     fontsize=16, fancybox=True, frameon=True,
                     ncol=len(labels), labelspacing = 1)
    fig.supxlabel('Crack length (cm)', fontsize=18)
    plt.tight_layout(w_pad=2)
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/simu-progress.svg', dpi=1000)


def localization_simu(method_name, data, args, case=22, group_fig=True, crack_text=True):
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    save_name = f'{method_name}-{args.experiment}'
    if method_name == 'SPIRIT':
        loss_scores_baseline, pipeline, w_undmg, w_undmg2, input = calculate_localization_baseline(method_name, data, args)
    else: # MIAE or AE
        loss_scores_baseline, model, trainer, train_normloss, test_normloss, norm = calculate_localization_baseline(method_name, data, args)
        
    # =============== Test
    os.makedirs('figs/localization', exist_ok=True)
    path = f'data/data-damaged-{args.experiment}/case{case}'
    if args.is_save:
        p_score = get_damage_score(loss_scores_baseline, loss_scores_baseline, args.all_sensors, alpha=args.alpha)

    # Plots for localization
    plt.subplots(figsize=(5, 5))
    plot_localization_simu(p_score, None, is_crack=False, cmap='viridis', is_bar=args.is_bar, crack_text=crack_text)
    plt.savefig(f'figs/localization/{save_name}-case-{case}-0mm.svg', dpi=1000, transparent=True)
    
    # =============== Test damaged cases
    filenames = os.listdir(path)
    length_list = find_length(filenames)
    lengths = np.sort(length_list)
    
    if group_fig:
        fig, ax = plt.subplots(1,len(lengths)+1,figsize=(len(lengths)*5,4.5))
        plt.subplot(1,len(lengths)+1,1) # plot undamaged first
        p_score = get_damage_score(loss_scores_baseline, loss_scores_baseline, args.all_sensors, alpha=args.alpha)
        plot_localization_simu(p_score, None, is_crack=False, is_bar=args.is_bar, crack_text=crack_text)
        
    for i,length in enumerate(lengths):
        indexs = np.where(length_list==length)[0]
        for index in indexs:
            damage_data = np.array( load( filenames[index], folder=path ).reshape(-1,12,args.all_sensors*2) )
            
            # Localization metric
            if method_name == 'SPIRIT':
                mat_dmg = damage_data.reshape(-1,2*args.all_sensors)
                input['data'] = mat_dmg.T
                output_dmg = pipeline.run(input)
                w_dmg = output_dmg['projection']  
                loss_scores2 = np.array([np.sqrt((w_dmg[i,0]-w_undmg[i,0])**2 + (w_dmg[i,1]-w_undmg[i,1])**2) for i in range(2*args.all_sensors)])
                p_score = get_damage_score(loss_scores2, loss_scores_baseline, args.all_sensors, alpha=args.alpha)
                
            else:
                damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
                loss_scores_damage_test = np.abs( np.mean(damage_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
                p_score = get_damage_score(loss_scores_damage_test, loss_scores_baseline, args.all_sensors, alpha=args.alpha)

            plt.subplot(1,len(lengths)+1,i+1+1) if group_fig else plt.subplots(figsize=(5, 5))
            plot_localization_simu(p_score, filenames[index], is_crack=True, cmap='viridis', 
                                    is_bar=args.is_bar, crack_text=crack_text)
            if args.is_save and not group_fig:
                plt.savefig(f'figs/localization/{save_name}-case-{case}-{int(length*1e3)}mm.svg', dpi=1000, transparent=True)
                
    if args.is_save and group_fig:
        plt.savefig(f'figs/localization/{save_name}-case-{case}-group.svg', dpi=1000, transparent=True)
    
def localization_crack_bc(method_name, data, args, save_name=None, index=np.arange(60,80), override=None):
    def plot_bc_crack(experiment, ax, p_score, is_damage=True, is_bar=True, cmap='viridis'):
        px, py, index_s, xv, yv, xs, ys = get_geo_info(experiment)
        
        if experiment == 'crack': # add transform due to different data collection order
            mapping = np.array([1,2,3,4,21,5,6,7,8,22,9,10,11,12,23,13,14,15,16,24,17,18,19,25,20,26])-1
            p_score = p_score[mapping]
        
        p_score = np.insert(p_score, p_score.size, p_score[-1])
        xi, yi, zi = interpolate_polygon(px, py, xs, ys, p_score, 'gaussian', resolution=200)
        zi[zi<0] = 0 # force non negative values
        vmax = max(10,np.nanmax(zi)) # if is_crack else 1
        cs = plt.contourf(xi, yi, zi/vmax, vmax=1.0, levels=np.linspace(0, 1.0, 21), cmap=cmap)
        plt.plot(px, py, lw=2, color='k')
        
        if is_damage:
            if experiment == 'crack':
                cpx, cpy, _, _, _, _, _ = get_experimental_crack_info()
                plt.fill(cpx, cpy, label='crack', color='r',)
            elif experiment == 'bc':
                import matplotlib.patches as patches
                ellipse = patches.Ellipse((29.5, 3), width=5, height=5, edgecolor='r', linewidth=2,
                                          facecolor='none', linestyle='--', label='Bolt loose')
                ax.add_patch(ellipse)
    
        plt.scatter(xs, ys, color='black', marker='.', label='sensors', s=12)
        if is_bar:
            cbar = plt.colorbar(cs, ticks=np.arange(0,1.05,0.25))# if not is_damage else plt.colorbar(cs,ticks=np.linspace(0,np.ceil(max(p_score)),5))
            cbar.ax.tick_params(labelsize=24) 
        plt.xlim([-1,37])
        plt.ylim([-1,46])
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.axis('off')
    
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    damage_data = load_damage_data(args, index=index)
    save_name = f'{method_name}-{args.experiment}'
    
    if method_name == 'SPIRIT':
        loss_scores_baseline, pipeline, w_undmg, w_undmg2, input = calculate_localization_baseline(method_name, data, args)

        mat_dmg = damage_data.reshape(-1,2*args.all_sensors)
        input['data'] = mat_dmg.T
        output_dmg = pipeline.run(input)
        w_dmg = output_dmg['projection']  
        loss_scores2 = np.array([np.sqrt((w_dmg[i,0]-w_undmg[i,0])**2 + (w_dmg[i,1]-w_undmg[i,1])**2) for i in range(2*args.all_sensors)])
        p_score = get_damage_score(loss_scores2, loss_scores_baseline, args.all_sensors, alpha=args.alpha)
        
    else:
        loss_scores_baseline, model, trainer, train_normloss, test_normloss, norm = calculate_localization_baseline(method_name, data, args)
        
        damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
        loss_scores_damage_test = np.abs( np.mean(damage_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
        p_score = get_damage_score(loss_scores_damage_test, loss_scores_baseline, args.all_sensors, alpha=args.alpha)
    
    fig, ax = plt.subplots(figsize=(3, 4)) # plot localization
    if args.experiment == 'crack':
        plot_bc_crack(args.experiment, ax, p_score, is_damage=True, is_bar=args.is_bar)
    elif args.experiment == 'bc':
        plot_bc_crack(args.experiment, ax, p_score, is_damage=True, is_bar=args.is_bar)
    
    if args.is_save:
        os.makedirs('figs/localization', exist_ok=True)
        plt.savefig(f'figs/localization/{save_name}-{args.alpha}.svg', dpi=1000, transparent=True)

def classification_model_predict(models, testdata):
    damage_error =  np.zeros( (testdata.shape[0],testdata.shape[-1]) )
    for sensor_id in range(testdata.shape[-1]):
        fdata = testdata[:,:,sensor_id]
        damage_error[:,sensor_id] = models[sensor_id].predict( fdata )
    return -damage_error 

def save_detection_metric_all_case(method_name, data, args):
    def test_damage_data(method_name, args, damage_data, model, test_mseloss, trainer):
        old_settings = np.seterr(all = 'ignore')
        if method_name in  ['OCSVM', 'LODA', 'IF']:
            damage_mseloss = classification_model_test_data(model, damage_data)
            metrics = get_detection_metrics(args, test_mseloss, damage_mseloss, method_name)
        elif method_name in ['AE', 'MIAE']:
            damage_mseloss, _ = model_test_data(model, trainer, damage_data)
            metrics = get_detection_metrics(args, test_mseloss, damage_mseloss, method_name)
        np.seterr(**old_settings)
        return metrics
    
    # training/loading model
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    if method_name in ['OCSVM', 'LODA', 'IF']: # for OCSVM, LODA, IF
        model, train_mseloss, test_mseloss = detection_model_train_test(args, method_name, data)
        trainer = None
    elif method_name in ['AE', 'MIAE']:
        model, trainer = choose_model(method_name, train_data, test_data, args, False)
        train_mseloss, _ = model_test_data(model, trainer, train_data)
        test_mseloss, _ = model_test_data(model, trainer, test_data)
        
    # save undamaged metric
    metrics = get_detection_metrics(args, train_mseloss, test_mseloss, method_name, test_damage=False)
    np.savetxt(f'{args.dirs}/{method_name}-{0}-all.txt', metrics)
    
    # ========= for bc and crack (single test)
    if args.experiment in ['crack', 'bc']:
        damage_data = load_damage_data(args)[:, :, args.data_index]
        metrics = test_damage_data(method_name, args, damage_data, model, test_mseloss, trainer)
        
        info = args.experiment
        np.savetxt(f'{args.dirs}/{method_name}-{info}-all.txt', metrics)
        
    # ========= for simu (multiple tests)
    elif args.experiment == 'simu':
        for length in args.cracks_sizes: #[0.008, 0.02, 0.04]: # small, medium, large cracks
            metrics = np.zeros((37,5))     # 37 cases and 5 metrics
            for i,case in enumerate(range(1,37+1)):
                path = f'data/data-damaged-{args.experiment}/case{case}'
                filenames = os.listdir(path)
                length_list = find_length(filenames)
                index = np.where(length_list==length)[0][0]
                damage_data = load( filenames[index], folder=path )[:, :, args.data_index]
                metrics[i,:] = test_damage_data(method_name, args, damage_data, model, test_mseloss, trainer)
                
            np.savetxt(f'{args.dirs}/{method_name}-{length}-all.txt', metrics)
    
    elif args.experiment == 'simu_noise':
        cases = np.array([1, 2, 3, 7, 11, 12, 13, 14, 15, 16, 19, 21, 22, 25, 26, 28, 37])
        for length in [0.015, 0.03, 0.04]: #0.015, 0.03, 0.04# small, medium, large cracks
            metrics = np.zeros((len(cases),5))     # 37 cases and 5 metrics
            for i,case in enumerate(cases):#enumerate(range(22,37+1)):
                path = f'data/data-damaged-{args.experiment}/ms-case{case}noise'
                filenames = os.listdir(path)
                length_list = find_length(filenames)
                index = np.where(length_list==length)[0][0]
                damage_data = load( filenames[index], folder=path )[:, :, args.data_index]
                damage_mseloss = model_test_damage_data(method_name, args, damage_data, model, trainer)
                metrics[i,:] = get_detection_metrics(args, test_mseloss, damage_mseloss, method_name)
            
            np.savetxt(f'{args.dirs}/{method_name}-{length}-all.txt', metrics)
            
    elif args.experiment == 'simu_temp' and not args.test_damage:
        path = f'data/data-undamaged-{args.experiment}-{args.temp}'
        filenames = os.listdir(path)
        undamage_data_test = np.array([load( files, folder=path) for files in filenames]).astype(np.float32).reshape(-1,12,args.all_sensors*2)
        data1 = args.scaler1.transform(undamage_data_test[:,:,:args.all_sensors].reshape(-1, args.all_sensors)).reshape(-1,12,args.all_sensors)
        data2 = args.scaler2.transform(undamage_data_test[:,:,args.all_sensors:].reshape(-1, args.all_sensors)).reshape(-1,12,args.all_sensors)
        undamage_data_test = np.concatenate( (data1,data2), axis=2)
        
        undamage_data_test = undamage_data_test[:, :, args.data_index]
        test_mseloss2 = model_test_damage_data(method_name, args, undamage_data_test, model, trainer)
        metric = get_detection_metrics(args, train_mseloss, test_mseloss2, method_name, test_damage=False)
        np.savetxt(f'{args.dirs}/{method_name}-temp-{args.temp}.txt', metric)
        
    elif args.experiment == 'simu_temp' and args.test_damage:
        path = f'data/data-damaged-{args.experiment}-{args.temp}'
        filenames = os.listdir(path)
        undamage_data_test = np.array([load( files, folder=path) for files in filenames]).astype(np.float32).reshape(-1,12,args.all_sensors*2)
        # if args.transform:
        data1 = args.scaler1.transform(undamage_data_test[:,:,:args.all_sensors].reshape(-1, args.all_sensors)).reshape(-1,12,args.all_sensors)
        data2 = args.scaler2.transform(undamage_data_test[:,:,args.all_sensors:].reshape(-1, args.all_sensors)).reshape(-1,12,args.all_sensors)
        undamage_data_test = np.concatenate( (data1,data2), axis=2)
        
        undamage_data_test = undamage_data_test[:, :, args.data_index]
        test_mseloss2 = model_test_damage_data(method_name, args, undamage_data_test, model, trainer)
        metric = get_detection_metrics(args, train_mseloss, test_mseloss2, method_name, test_damage=True)
        np.savetxt(f'{args.dirs}/{method_name}-damaged-temp-{args.temp}.txt', metric)
        
def model_test_damage_data(method_name, args, damage_data, model, trainer):
    old_settings = np.seterr(all = 'ignore')
    if method_name in ['OCSVM', 'LODA', 'IF']:
        damage_mseloss = classification_model_test_data(model, damage_data)
    elif method_name in ['AE', 'MIAE']:
        damage_mseloss, _ = model_test_data(model, trainer, damage_data)
    np.seterr(**old_settings)
    return damage_mseloss

def detection_model_train_test(args, method_name, data_all):
    train_error = np.zeros( (int(len(data_all)*0.7),data_all.shape[-1]) )
    test_error =  np.zeros( (len(data_all)-len(train_error),data_all.shape[-1]) )
    models = [() for _ in range(data_all.shape[-1])]
    for sensor_id in range(data_all.shape[-1]):
        data = data_all[:,:,sensor_id]
        train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
        if method_name == 'IF':
            model = IsolationForest(n_estimators=100, contamination=args.contamination, random_state=0)
        elif method_name == 'OCSVM':
            model = svm.OneClassSVM(kernel='rbf', nu=args.nu)
        elif method_name =='LODA':
            model = LODA(n_estimators=10, bins=10, random_state=10)
        model.fit(train_data)
        train_error[:,sensor_id] = model.predict( train_data ) #246
        test_error[:,sensor_id]  = model.predict( test_data ) #106
        models[sensor_id] = model
    return models, -train_error, -test_error # negative because model predicts 1 as normal/undamaged data

def classification_model_test_data(models, testdata):
    damage_error = np.zeros( (testdata.shape[0], testdata.shape[-1]) )
    for sensor_id in range(testdata.shape[-1]):
        damage_error[:,sensor_id] = models[sensor_id].predict( testdata[:,:,sensor_id] )
    return -damage_error # negative because model predicts 1 as normal/undamaged data

def calculate_localization_baseline(method_name, data, args):
    def get_default_pipeline(args):
        spirit = SpiritStage(ispca=False,thresh=0.01,ebounds=(0.96,1.1), startm=2*args.active_sensors)
        pipeline = Pipeline()
        pipeline.append_stage(spirit)
        return pipeline
    
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    if method_name == 'SPIRIT':
        pipeline = get_default_pipeline(args)
        input = {}
        mat_undmg = train_data.reshape(-1,2*args.active_sensors)
        mat_undmg2 = test_data.reshape(-1,2*args.active_sensors)
    
        input['data'] = mat_undmg.T
        output_undmg = pipeline.run(input)
        w_undmg = output_undmg['projection']
    
        input['data'] = mat_undmg2.T
        output_undmg2 = pipeline.run(input)
        w_undmg2 = output_undmg2['projection']
        
        loss_scores_baseline = np.array([np.sqrt((w_undmg2[i,0]-w_undmg[i,0])**2 + (w_undmg2[i,1]-w_undmg[i,1])**2) for i in range(2*args.active_sensors)])
        return loss_scores_baseline, pipeline, w_undmg, w_undmg2, input
    
    else: #for ae and MIAE
        model, trainer = choose_model(method_name, train_data, test_data, args, False)
        _, train_normloss = model_test_data(model, trainer, train_data)
        _, test_normloss = model_test_data(model, trainer, test_data)
        loader = DataLoader(train_data, batch_size=1, shuffle=False)
        norm = np.mean( [torch.norm(x[0,:,:], dim=0).detach().numpy() for x in loader], axis=0)
        loss_scores_baseline = np.abs( np.mean(train_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
        return loss_scores_baseline, model, trainer, train_normloss, test_normloss, norm

def get_p_score(method_name, damage_data, args, bulk):
    if method_name == 'SPIRIT':
        loss_scores_baseline, pipeline, w_undmg, w_undmg2, input = bulk
        mat_dmg = damage_data.reshape(-1,2*args.active_sensors) # reshape 3d to 2d
        input['data'] = mat_dmg.T
        output_dmg = pipeline.run(input)
        w_dmg = output_dmg['projection']  
        loss_scores_damage_test = np.array([np.sqrt((w_dmg[i,0]-w_undmg[i,0])**2 + (w_dmg[i,1]-w_undmg[i,1])**2) for i in range(2*args.active_sensors)])
    else:
        loss_scores_baseline, model, trainer, train_normloss, test_normloss, norm = bulk
        damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
        loss_scores_damage_test = np.abs( np.mean(damage_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
    
    p_score = get_damage_score(loss_scores_damage_test, loss_scores_baseline, args.active_sensors, alpha=args.alpha)
    return p_score
    
def weighted_centroid(x_polygon, y_polygon, measurements):
    total_weight = sum(measurements)
    
    weighted_x = sum(x * m for x, m in zip(x_polygon, measurements))
    weighted_y = sum(y * m for y, m in zip(y_polygon, measurements))
    
    centroid_x = weighted_x / total_weight
    centroid_y = weighted_y / total_weight
    return centroid_x, centroid_y

def interpolate_polygon_with_centroid(x_polygon, y_polygon, x_sensor, y_sensor, z_sensor, function, resolution):
    poly_path = Path(list(zip(x_polygon, y_polygon)))   # Create polygon path
    x_min, x_max = min(x_polygon), max(x_polygon)       # Create a grid
    y_min, y_max = min(y_polygon), max(y_polygon)
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi, yi = np.meshgrid(xi, yi)
    # Mask for points inside the polygon
    points = np.vstack((xi.flatten(), yi.flatten())).T
    mask = poly_path.contains_points(points).reshape(xi.shape)
    
    # Interpolate the data, add the centroid control point in the middle
    centroid_x, centroid_y = weighted_centroid(x_sensor, y_sensor, z_sensor)
    x = np.append(x_sensor, centroid_x)
    y = np.append(y_sensor, centroid_y)
    z = np.append(z_sensor, sum(z_sensor))

    rbf = Rbf(x, y, z, function=function)
    zi = rbf(xi, yi)
    zi[~mask] = np.nan
    return xi, yi, zi, centroid_x, centroid_y

def interpolate_polygon(x_polygon, y_polygon, x_sensor, y_sensor, z_sensor, function, resolution):
    poly_path = Path(list(zip(x_polygon, y_polygon)))   # Create polygon path
    x_min, x_max = min(x_polygon), max(x_polygon)       # Create a grid
    y_min, y_max = min(y_polygon), max(y_polygon)
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi, yi = np.meshgrid(xi, yi)
    # Mask for points inside the polygon
    points = np.vstack((xi.flatten(), yi.flatten())).T
    mask = poly_path.contains_points(points).reshape(xi.shape)

    rbf = Rbf(x_sensor, y_sensor, z_sensor, function=function)
    zi = rbf(xi, yi)
    zi[~mask] = np.nan
    return xi, yi, zi

def localization_contour(method_name, args, xi, yi, zi, x_polygon, y_polygon, x_sensor, y_sensor, 
                         cpx, cpy, centroid_x, centroid_y):
    x_min, x_max = min(x_polygon), max(x_polygon)
    y_min, y_max = min(y_polygon), max(y_polygon)
    if args.experiment == 'simu':
        figsize=(3, 3)
    elif args.experiment in ['crack', 'bc']:
        figsize=(3, 4) 
        
    zi[zi<0] = 0 # force non negative values
    vmax = max(10,np.nanmax(zi))
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.contourf(xi, yi, zi/vmax, vmax=1.0, levels=np.linspace(0, 1.0, 21), cmap='viridis')
    plt.scatter(x_sensor, y_sensor, color='k', marker='o', s=20)
    plt.plot(x_polygon, y_polygon, lw=2, color='k')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    if args.experiment == 'bc':
        ellipse = patches.Ellipse((29.5, 3), width=5, height=5, edgecolor='r', linewidth=2,
                                  facecolor='none', linestyle='--', label='Bolt loose')
        ax.add_patch(ellipse)
    elif args.experiment in ['crack', 'simu']:
        plt.fill(cpx, cpy, label='Crack', color='r',)
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')
    
    os.makedirs('figs/localization', exist_ok=True)
    if args.experiment == 'simu' and args.temp_crack_length == 0.04:
        save_name = f'figs/localization/{args.experiment}-{args.active_sensors}-{method_name}-{args.temp_case}-{args.temp_crack_length}.svg'
    elif args.experiment in ['crack', 'bc']:
        save_name = f'figs/localization/{args.experiment}-{args.active_sensors}-{method_name}.svg'
    plt.savefig(save_name, dpi=1000, transparent=True)
    plt.show()


def get_localization_dist(method_name, args, damage_data, bulk, filename=None):
    def distance_to_rectangle(x, y, cpx, cpy):
        point = Point(x, y)
        polygon = Polygon(zip(cpx, cpy))
        distance = point.distance(polygon)
        return distance
    
    x_polygon, y_polygon, _, _, _, xs, ys = get_geo_info(args.experiment) #plate x,y
    x_sensor, y_sensor = xs[args.sensor_index], ys[args.sensor_index]
    p_score = get_p_score(method_name, damage_data, args, bulk)
    # get centroid for localization
    resolution = 200
    xi, yi, zi, centroid_x, centroid_y = interpolate_polygon_with_centroid(
        x_polygon, y_polygon, x_sensor, y_sensor, args.interpolate_alg(p_score), 
        args.interpolate_function, resolution=resolution)
    
    # get crack points 
    if args.experiment == 'simu':
        cpx, cpy, _, crack_length, _, _, _ = get_simu_crack_info(filename)
        args.temp_crack_length = crack_length
    elif args.experiment in ['crack', 'bc']:
        cpx, cpy, _, _, _, _, _ = get_experimental_crack_info()
    # get distance
    dist = distance_to_rectangle(centroid_x, centroid_y, cpx, cpy)
    # visualize and save plots
    if args.visual:
        if args.experiment == 'simu' and args.temp_crack_length < 0.04:
            pass
        else:
            localization_contour(method_name, args, xi, yi, zi, x_polygon, y_polygon, 
                                 x_sensor, y_sensor, cpx, cpy, centroid_x, centroid_y,)
    return dist

def localization_fewer_sensors(method_name, args, data, cases=np.array(
        [1, 2, 3, 7, 11, 12, 13, 14, 15, 16, 19, 21, 22, 25, 26, 28, 37])):
    def get_simu_localization_dist_by_case(method_name, args, case, bulk):
        path = f'data/data-damaged-{args.experiment}/case{case}'
        filenames = os.listdir(path)
        length_list = find_length(filenames)
        args.temp_case = case # for save simu plots only
        dists = np.zeros(len(args.cracks_sizes),)
        for i, length in enumerate(args.cracks_sizes): # use specified crack length from cracks_sizes
            index = np.where(length_list==length)[0][0]# only one index
            damage_data = load( filenames[index], folder=path )[:, :, args.data_index]
            dists[i] = get_localization_dist(method_name, args, damage_data, bulk, filename=filenames[index])
        return dists
    
    bulk = calculate_localization_baseline(method_name, data, args)
    if args.experiment == 'simu':
        dist_data = np.array([get_simu_localization_dist_by_case(method_name, args, case, bulk) for case in cases])
    elif args.experiment in ['crack', 'bc']:
        damage_data = load_damage_data(args)[:, :, args.data_index]
        dist_data = get_localization_dist(method_name, args, damage_data, bulk)
    return dist_data

def save_detection_metric_beam_column(method_name, data, args, levels_list=[660, 664, 685]):
    # training/loading model
    undamage_data = data[424:685]
    train_data, test_data = train_test_split(undamage_data, train_size=0.7, random_state=24)
    if method_name in ['OCSVM', 'LODA', 'IF']: # for OCSVM, LODA, IF
        model, train_mseloss, test_mseloss = detection_model_train_test(args, method_name, undamage_data)
        trainer = None
    elif method_name in ['AE', 'MIAE']:
        model, trainer = choose_model(method_name, train_data, test_data, args, False)
        train_mseloss, _ = model_test_data(model, trainer, train_data)
        test_mseloss, _ = model_test_data(model, trainer, test_data)
        
    # save undamaged metric
    metrics0 = get_detection_metrics(args, train_mseloss, test_mseloss, method_name, test_damage=False)
    np.savetxt(f'{args.dirs}/{method_name}-D{0}-all.txt', metrics0)
    
    threshold = 676
    for levels, a in enumerate(levels_list):
        b = a + 20 # gap of 20 each time
        if b>threshold and a>=threshold:
            damage_data = data[a:b]
            undamage_data = None
            if method_name in  ['OCSVM', 'LODA', 'IF']:
                damage_mseloss = classification_model_test_data(model, damage_data)
            elif method_name in ['AE', 'MIAE']:
                damage_mseloss, _ = model_test_data(model, trainer, damage_data)
            metrics = get_detection_metrics_mix(args, test_mseloss, None, damage_mseloss, method_name, test_damage=True)
            
        elif b>threshold and a<threshold:
            undamage_data = data[a:threshold]
            damage_data = data[threshold:b]
            if method_name in  ['OCSVM', 'LODA', 'IF']:
                damage_mseloss = classification_model_test_data(model, damage_data)
                undamage_mseloss = classification_model_test_data(model, undamage_data)
            elif method_name in ['AE', 'MIAE']:
                damage_mseloss, _ = model_test_data(model, trainer, damage_data)
                undamage_mseloss, _ = model_test_data(model, trainer, undamage_data)
            metrics = get_detection_metrics_mix(args, test_mseloss, undamage_mseloss, damage_mseloss, method_name, test_damage=True)
            
        else: #undamaged
            undamage_data = data[a:b]
            if method_name in  ['OCSVM', 'LODA', 'IF']:
                undamage_mseloss = classification_model_test_data(model, undamage_data)
            elif method_name in ['AE', 'MIAE']:
                undamage_mseloss, _ = model_test_data(model, trainer, undamage_data)
            metrics = get_detection_metrics_mix(args, test_mseloss, undamage_mseloss, None, method_name, test_damage=False)
        np.savetxt(f'{args.dirs}/{method_name}-D{levels+1}-all.txt', metrics)

def get_detection_metrics_mix(args, train_error, test_error, damage_error, method_name, test_damage=True):
    # mixed condition, damage and undamage sample overlap
    if test_damage:
        if test_error is None:
            label_true = np.concatenate( (-np.ones(len(train_error)), np.ones(len(damage_error))), ) # per sample only
            score = np.concatenate((train_error, damage_error), axis=0)
        else:
            label_true = np.concatenate( (-np.ones(len(train_error)), -np.ones(len(test_error)), np.ones(len(damage_error))), ) # per sample only
            score = np.concatenate((train_error, test_error, damage_error), axis=0)

        smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=3)
        enn = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=15)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)
        score, label_true = smote_enn.fit_resample(score, label_true)
        
    else: # if test undamaged data, no need for SMOTE due to enough number of samples
        label_true = np.concatenate( (-np.ones(len(train_error)), -np.ones(len(test_error))), )
        score = np.concatenate((train_error, test_error), axis=0)
    
    return calculate_detection_metric(args, method_name, score, train_error, label_true, test_damage)

def localization_damage_score_beam_column(method_name, args, level_list=np.arange(396,760,3), gap=15):
    args.data_index = np.concatenate( (args.sensor_index, args.sensor_index+args.all_sensors) )
    data = load_undamage_data(args)[:, :, args.data_index]
    undamage_data = data[424:685]
    bulk = calculate_localization_baseline(method_name, undamage_data, args)
    p_scores = []
    for levels in level_list:
        a = levels
        b=a+gap
        damage_data = data[b:b+gap]
        p_score = get_p_score(method_name, damage_data, args, bulk)
        p_scores.append(p_score)
    return np.stack(p_scores)

def localization_beam_column(args, metrics, damage_level_list=[40,91,94,120]):
    def plot_beam_column(args, p_score, level=''):
        if len(p_score) == 4:
            x_sensor = np.array([5.7559, 5.7559, 2.7859, 2.7859])
            y_sensor = np.array([3.4419, 0.6181, 3.4419, 0.6181])
        elif len(p_score) == 6:
            x_sensor = np.array([5.7559, 5.7559, 2.7859, 2.7859, 0, 0])
            y_sensor = np.array([3.4419, 0.6181, 3.4419, 0.6181, 3.4419, 0.6181])
        
        xl, yl = 7.8-1, 4.06
        x_polygon = [0,xl,xl,0,0]
        y_polygon = [0,0,yl,yl,0]
        x_min, x_max = min(x_polygon), max(x_polygon)
        y_min, y_max = min(y_polygon), max(y_polygon)
        
        xi, yi, zi, centroid_x, centroid_y = interpolate_polygon_with_centroid(
            x_polygon, y_polygon, x_sensor, y_sensor, args.interpolate_alg(p_score), 
            args.interpolate_function, resolution=200)
        zi[zi<0] = 0 # force non negative values
    
        fig, ax = plt.subplots(figsize=(4.5,3))
        vmax = max(2,np.nanmax(zi))
        plt.contourf(xi, yi, zi/vmax, vmax=1.0, levels=np.linspace(0, 1.0, 21), cmap='viridis')
        plt.scatter(x_sensor, y_sensor, color='k', marker='o', s=20)
        plt.plot(x_polygon, y_polygon, lw=2, color='k')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        ellipse = patches.Ellipse((1.5, 1), width=1, height=1, edgecolor='r', linewidth=2,
                                  facecolor='none', linestyle='--', label='Bolt loose')
        ax.add_patch(ellipse)
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'D{level}', y=0.95, fontsize=24)
        plt.tight_layout()
    
    for index, method in enumerate(['MIAE', 'AE']):
        data = metrics[index]
        for i,k in enumerate(damage_level_list):
            if args.active_sensors == 8:
                score = np.maximum(data[k,:4]-np.max(data[:10,:4],0), 1e-5) # some filtering
            elif args.active_sensors == 4:
                score = np.maximum(data[k,args.sensor_index]-np.max(data[:60,args.sensor_index],0), 1e-5) # some filtering
            plot_beam_column(args, score, level=i)
            plt.savefig(f'figs/localization/{method}-{args.active_sensors}sensors-D{i}.svg', dpi=1000, transparent=True)

def plot_localization_simu(p_score, name, is_crack=True, cmap='viridis', is_bar=True, crack_text=True):
    px, py, index_s, xv, yv, xs, ys = get_geo_info('simu')

    xi, yi, zi = interpolate_polygon(px, py, xs, ys, p_score, 'gaussian', resolution=200)
    zi[zi<0] = 0 # force non negative values
    vmax = max(2,np.nanmax(zi)) if is_crack else 1
    cs = plt.contourf(xi, yi, zi/vmax, vmax=1.0, levels=np.linspace(0, 1.0, 21), cmap=cmap)
        
    plt.plot(px, py, lw=2, color='k')
    if is_crack:
        cpx, cpy, _, L, _, _, _ = get_simu_crack_info(name)
        plt.fill(cpx, cpy, label='crack', color='r',)
        if crack_text:
            plt.title(f'$l$={L*1000:.0f}mm', fontsize=20, color='blue')
    if not is_crack and crack_text:
        plt.title('intact', fontsize=20, color='blue')
    plt.scatter(xs, ys, color='black', marker='.', label='sensors', s=12)
    if is_bar:
        cbar = plt.colorbar(cs,ticks=np.arange(0,1.05,0.25))
        cbar.ax.tick_params(labelsize=24)
    plt.xlim([-33,33])
    plt.ylim([-35,36])
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')

def detection_simu_fewer_sensors_metric(args, pick_n_sensor_list, evaluate=False): # for fewer sensors simu detection
    y1 = []
    for pick_n_sensors in pick_n_sensor_list:
        folder = f'saved_models/saved_models-{pick_n_sensors}sensors'
        os.makedirs(folder, exist_ok=True)
        crack1 = []
        if pick_n_sensors in [4,6,8]:
            if pick_n_sensors == 4:
                lists = (np.array([9, 13, 36, 40])-1, np.array([11, 23, 27, 38])-1, 
                         np.array([10, 30, 39, 20])-1, np.array([12, 16, 37, 34])-1,
                         np.array([9, 12, 30, 33])-1, np.array([17, 19, 37, 39])-1)
            elif pick_n_sensors == 6:
                lists = (np.array([10, 12, 23, 27, 36, 40])-1, np.array([9, 13, 23, 36, 27, 40])-1, 
                         np.array([11, 38, 16, 30, 20, 34])-1, np.array([9, 30, 11, 32, 13, 34])-1,
                         np.array([10, 12, 24, 26, 37, 39])-1, np.array([9, 38, 13, 34, 30, 18])-1)
            elif pick_n_sensors == 8:
                lists = (np.array([9, 10, 16, 17, 33, 34, 39, 40])-1, np.array([30, 31, 36, 37, 12, 13, 19, 20])-1, 
                         np.array([8, 10, 29, 31, 12, 14, 33, 35])-1, np.array([2, 15, 30, 17, 19, 6, 21, 34])-1,
                         np.array([9, 22, 36, 24, 26, 13, 28, 40])-1, np.array([22, 31, 9, 18, 43, 13, 33, 28])-1)
                
            for args.sensor_index in lists:
                args.active_sensors = len(args.sensor_index)
                folder = f'saved_models/saved_models-{args.active_sensors}sensors'
                args.data_index = np.concatenate( (args.sensor_index, args.sensor_index+args.all_sensors) )
                args.dirs = f'{folder}/{args.experiment}-{args.net_size}-{args.sensor_index}-{args.epochs}'
                
                # ====== detection
                if evaluate:
                    data = load_undamage_data(args)[:, :, args.data_index]
                    save_detection_metric_all_case('MIAE', data, args)
                    save_detection_metric_all_case('AE', data, args)
                    save_detection_metric_all_case('IF', data, args)
                    save_detection_metric_all_case('OCSVM', data, args)
                    save_detection_metric_all_case('LODA', data, args)
                    
                crack1.append( np.stack(get_detection_metric_data_all(
                    args.cracks_sizes[0], path=args.dirs) ) )
                crack11 = np.concatenate(crack1, axis=1)
            y1.append(crack11)
          
        else:
            for seed in [231,232,233,244,248,246]:
                np.random.seed(seed)
                args.sensor_index = np.random.choice(args.all_sensors-7-4, pick_n_sensors, replace=False)+7
                args.data_index = np.concatenate( (args.sensor_index, args.sensor_index+args.all_sensors) )
                args.dirs = f'{folder}/{args.experiment}-{args.net_size}-{args.sensor_index}-{args.seed}-{args.epochs}'
                
                # ====== detection
                if evaluate:
                    data = load_undamage_data(args)[:, :, args.data_index]
                    save_detection_metric_all_case('MIAE', data, args)
                    save_detection_metric_all_case('AE', data, args)
                    save_detection_metric_all_case('IF', data, args)
                    save_detection_metric_all_case('OCSVM', data, args)
                    save_detection_metric_all_case('LODA', data, args)
                
                crack1.append( np.stack(get_detection_metric_data_all(
                    args.cracks_sizes[0], path=args.dirs) ) )
                
                crack11 = np.concatenate(crack1, axis=1) # (method, cases, metric)
            y1.append(crack11)
    metrics = np.stack(y1)
    return metrics
