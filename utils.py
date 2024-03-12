# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : Mechanics-AE for damage detection and localization
# //   File        : utils.py
# //   Description : Utility functions
# //
# //   Created On: 3/7/2024
# /////////////////////////////////////////////////////////////////


import torch
import numpy as np
import scipy.io as spio
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from models import MechanicsAutoEncoderSimu, MechanicsAutoEncoder, AutoEncoder, AutoEncoderSimu

def load(file_name, folder, length = 24, gap = 2):
    load_name = os.path.join(folder,file_name)
    mat = spio.loadmat(load_name, squeeze_me=True)['ms']  # nx8
    samples = int( (mat.shape[0] - length + 1)/gap )
    data = np.zeros( (samples, length, mat.shape[1]) )
    for i in range(samples):
        data[i,:,:] = mat[i*gap:i*gap+length,:]
    return data
    
def load_undamage_data(args):
    filepath = f'data/data-undamaged-{args.experiment}'
    filenames = os.listdir(filepath)
    testdata = np.array([load( files, folder=filepath ) for files in filenames])
    if args.experiment == 'crack':
        testdata = testdata.reshape(-1,12,args.n_sensors*2)
    elif args.experiment == 'bc':
        testdata = testdata.transpose(1,2,0,3).reshape(-1,12,args.n_sensors*2)[20:,:,:] # remove the initial data varitions
    return testdata.astype(np.float32)

def load_damage_data(args):
    filepath = f'data/data-damaged-{args.experiment}'
    filenames = os.listdir(filepath)
    damage_data = np.array([load( files, folder=filepath ) for files in filenames]).astype(np.float32)
    if args.experiment == 'crack':
        return damage_data.reshape(-1,12,args.n_sensors*2)[20:40]
    elif args.experiment == 'bc':
        return damage_data.transpose(1,2,0,3).reshape(-1,12,args.n_sensors*2)[60:80]

def load_simu_data(args):
    filepath = f'data/data-undamaged-{args.experiment}'
    filenames = os.listdir(filepath)
    testdata = np.array([load( files, folder=filepath ) for files in filenames])
    testdata = testdata.reshape(-1,12,args.n_sensors*2)
    return testdata.astype(np.float32)

def choose_model(model_name, train_data, test_data, args, is_train=False, epochs=0):
    if args.experiment == 'simu': # simulations
        if model_name == 'ae' or model_name == 'AE':
            model = AutoEncoderSimu(args.h_dim)
        elif model_name == 'mechae': #'Mechanics-AE'
            model = MechanicsAutoEncoderSimu(args.h_dim)
    else:
        if model_name == 'ae' or model_name == 'AE':
            model = AutoEncoder(args.h_dim)
        elif model_name == 'mechae':
            model = MechanicsAutoEncoder(args.h_dim)
            model.get_weight(args)
            
    # Model setup
    model.load_data(train_data, test_data)
    if not is_train:
        logger = pl.loggers.TensorBoardLogger(save_dir="logs", log_graph=False)
        trainer = pl.Trainer(enable_progress_bar=False, logger=logger)
        PATH = f'saved_models/model-{model_name}-{args.experiment}.ckpt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.load_from_checkpoint(PATH, h_dim=args.h_dim, map_location=torch.device(device))
        model.freeze()
        return model, trainer
    else: #Train the model
        pass


def model_test_data(model, trainer, data):
    test_dataloader = DataLoader(data.astype(np.float32), batch_size=1, shuffle=False)
    losses = torch.stack(trainer.predict(model, dataloaders=test_dataloader), dim=0).cpu().detach().numpy()
    test_mseloss = losses[:,0,:]
    test_normloss = losses[:,1,:]
    return test_mseloss, test_normloss

def get_geo_info():
    px = [0, 36.195, 36.195, 23.495, 0, 0]
    py = [0, 0, 31.877, 44.577, 44.577, 0]

    s0 = [20,25,26]
    s1 = np.arange(3)
    index_s=s0-s1
    
    xv, yv = np.meshgrid(np.arange(6,36,6.5),np.arange(7,44,6.5))
    xv = np.flip(xv)
    
    xs = xv.reshape(-1,1)
    ys = yv.reshape(-1,1)
    xs = np.delete(xs,s0)
    ys = np.delete(ys,s0)
    return px, py, index_s, xv, yv, xs, ys


def get_detection_metrics(args, train_error, test_error, is_other_methods=False):
    def find_threshold_for_fpr(label, score, fpr_target1):
        fpr, tpr, thresholds = roc_curve(label, score)
        closest_index = np.argmin(np.abs(fpr - fpr_target1))
        threshold = thresholds[closest_index if closest_index!=0 else 1]
        return threshold
    def get_single_decision(score, threshold):
        decision1 = -np.ones( (len(score)) ) # defualt -1
        index = np.array(np.where(score > threshold))
        decision1[index] = 1
        return decision1
    def get_decision_labels(label_pred_sample_by_sensor, sensor_percentage):
        num_sensors = label_pred_sample_by_sensor.shape[1] # number of sensor features
        label_pred = -np.ones( (len(label_pred_sample_by_sensor)) )
        for i in range( len(label_pred_sample_by_sensor) ): # take sample dimension
            label_pred[i] = 1 if np.count_nonzero((label_pred_sample_by_sensor[i,:] == 1)) >= sensor_percentage*num_sensors else -1
        return label_pred
    
    num_sensors = train_error.shape[1] # number of sensor features or number of sensors*2
    
    # Evaluate per sample decision at each sensor, based on the error distribution at that sensor
    score = np.concatenate((train_error, test_error), axis=0) # combine all the samples
    label_true = np.concatenate( (-np.ones(len(train_error)), np.ones(len(test_error))), ) # per sample only
    
    from imblearn.over_sampling import SMOTE # resolve the scarce data issue with SMOTE
    score, label_true = SMOTE().fit_resample(score, label_true)
    
    if is_other_methods: # binary classification method, return score and get decision in the next step
        label_pred_sample_by_sensor = score
    else:
        label_pred_sample_by_sensor = np.zeros(score.shape)
        for dm in range(num_sensors): 
            threshold = find_threshold_for_fpr(label_true, score[:,dm], args.fpr_target)
            label_pred_sample_by_sensor[:,dm] = get_single_decision(score[:,dm], threshold)
    
    # Now we have 2d decisions per sample and per sensor
    # each sample should give one decision, so reduce the sensor dimension by the "sensor_percentage"
    label_pred = get_decision_labels(label_pred_sample_by_sensor, args.sensor_percentage)
    metric1 = np.array([accuracy_score(label_true, label_pred), 
                        precision_score(label_true, label_pred), 
                        recall_score(label_true, label_pred), 
                        f1_score(label_true, label_pred), 
                        roc_auc_score(label_true, label_pred)])
    return metric1

def get_damage_score(error, error_train, n_sensors, alpha = 0.3):
    max_train1 = np.max(error_train[:n_sensors])
    max_train2 = np.max(error_train[n_sensors:])
    damage_scores = ( alpha*error[:n_sensors]/max_train1 + 
                     (1-alpha)*error[n_sensors:]/max_train2 )/2
    return damage_scores


def find_length(filenames):
    length_list = []
    for name in filenames:
        idx3 = name.find('-l')
        idx4 = name.find('-w')
        length = float(name[idx3+2:idx4])
        length_list.append(length)
    return np.array(length_list)


def compare_mechae_ae_simu_detection(data, args):
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    def get_detections_range_case22(model_name, train_data, test_data, args):
        model, trainer = choose_model(model_name, train_data, test_data, args, False, 0)
        # =============== Baseline evaluation
        train_mseloss, _ = model_test_data(model, trainer, train_data)
        test_mseloss, _ = model_test_data(model, trainer, test_data)
        metric0 = get_detection_metrics(args, train_mseloss, test_mseloss,  is_other_methods=False)

        path = f'data/data-damaged-{args.experiment}/case22-more'
        filenames = os.listdir(path)
        length_list = find_length(filenames)
        lengths = sorted(set(length_list)) # find unique length
        metrics = np.zeros((len(lengths), 5, int(len(length_list)/len(lengths))))
        for i,length in enumerate(lengths):
            indexs = np.where(length_list==length)[0]#[0]
            metric = np.zeros( (5, int(len(length_list)/len(lengths))) )
            for j,index in enumerate(indexs): # check each file
                damage_data = np.array( load( filenames[index], folder=path ).reshape(-1,12,args.n_sensors*2) )
                damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
                metric[:,j] = get_detection_metrics(args, train_mseloss, damage_mseloss, is_other_methods=False)
            metrics[i,:,:] = metric
        m_mean = np.mean(metrics,axis=2)
        m_std = np.std(metrics,axis=2)
        m_mean = np.concatenate( (metric0.reshape(1,-1), m_mean), axis=0)
        m_std = np.concatenate( (np.zeros( (1, m_std.shape[1]) ), m_std), axis=0)
        return m_mean, m_std
    
    model_name1 = 'ae' 
    model_name2 = 'mechae' 
    m1_mean, m1_std = get_detections_range_case22(model_name1, train_data, test_data, args)
    m2_mean, m2_std = get_detections_range_case22(model_name2, train_data, test_data, args) 
    
    scale = 1.5
    y2_lower = np.clip(m2_mean - m2_std/scale, 0,1)
    y2_upper = np.clip(m2_mean + m2_std/scale, 0,1)
    y1_lower = np.clip(m1_mean - m1_std/scale, 0,1)
    y1_upper = np.clip(m1_mean + m1_std/scale, 0,1)
    
    c1 = '#1FBF8F'
    c2 = '#DD6CAF'
    alpha = 0.4
    lengths = np.array([0.004, 0.008, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06] )
    xx = np.concatenate(([0],lengths))*1e2
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUROC']
    lists = [0,1,2,3,4]

    fig, axs = plt.subplots(1, len(lists), figsize=(3.8*len(lists), 3.8))
    for i,index in enumerate(lists):
        plt.subplot(1,len(lists),i+1)
    
        plt.plot(xx, m1_mean[:,index], label='Autoencoder', linewidth=2, marker='^', 
                 markersize=10, color=c1)
        plt.fill_between(xx, y1_lower[:,index], y1_upper[:,index], alpha=alpha, 
                         color=c1)
        
        plt.plot(xx, m2_mean[:,index], label='Mechanics-AE', linewidth=2, marker='o', 
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
    fig.supxlabel('Crack propagation (cm)', fontsize=18)
    plt.tight_layout(w_pad=2)
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/detection-distribution.svg', dpi=1000)
    plt.close()
    print('------detection performance demo saved at figs/detection/detection-distribution.svg\n')


def save_detection_metric_all_case(model_name, data, args):
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    if model_name != 'ae' and model_name != 'mechae': # for OCSVM, LODA, IF
        model, train_mseloss, test_mseloss = detection_model_train_test(model_name, data)
    else:
        model, trainer = choose_model(model_name, train_data, test_data, args, False, 0)
        train_mseloss, _ = model_test_data(model, trainer, train_data)
        
    for length in [0.015, 0.03, 0.04]: # small, medium, large cracks
        metrics = np.zeros((37,5))     # 37 cases and 5 metrics
        for i,case in enumerate(range(1,37+1)):
            path = f'data/data-damaged-{args.experiment}/case{case}'
            filenames = os.listdir(path)
            length_list = find_length(filenames)
            index = np.where(length_list==length)[0][0]
            damage_data = np.array( load( filenames[index], folder=path ).reshape(-1,12,args.n_sensors*2) )
            if model_name != 'ae' and model_name != 'mechae':
                damage_mseloss = classification_model_predict(model, damage_data)
                metrics[i] = get_detection_metrics(args, train_mseloss, damage_mseloss, is_other_methods=True)
            else:
                damage_mseloss, _ = model_test_data(model, trainer, damage_data)
                metrics[i,:] = get_detection_metrics(args, train_mseloss, damage_mseloss)
            
        os.makedirs('data/metric-detection-simu-all', exist_ok=True)
        np.savetxt(f'data/metric-detection-simu-all/{model_name}-{length}-all.txt', metrics)
        print(f'------{model_name}-crack {length}mm-all metric saved')


def detection_model_train_test(method_name, data_all):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import IsolationForest
    from sklearn import svm
    from anlearn.loda import LODA
    train_error = np.zeros( (int(len(data_all)*0.7),data_all.shape[-1]) )
    test_error =  np.zeros( (len(data_all)-len(train_error),data_all.shape[-1]) )
    models = [() for _ in range(data_all.shape[-1])]
    for sensor_id in range(data_all.shape[-1]):
        data = data_all[:,:,sensor_id]
        train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
        if method_name == 'IF':
            model = IsolationForest(random_state=0)
        elif method_name == 'OCSVM':
            model = svm.OneClassSVM(kernel='rbf', nu=0.05)
        elif method_name =='LODA':
            model = LODA(n_estimators=200, bins=50, random_state=10)
        model.fit(train_data)
        train_error[:,sensor_id] = model.predict( train_data ) #246
        test_error[:,sensor_id]  = model.predict( test_data ) #106
        models[sensor_id] = model
    return models, -train_error, -test_error # negative because model predicts 1 as normal/undamaged data


def classification_model_predict(models, testdata):
    damage_error =  np.zeros( (testdata.shape[0],testdata.shape[-1]) )
    for sensor_id in range(testdata.shape[-1]):
        fdata = testdata[:,:,sensor_id]
        damage_error[:,sensor_id] = models[sensor_id].predict( fdata )
    return -damage_error # negative because model predicts 1 as normal/undamaged data

def detection_crack_bc(method_name, data, args):
    damage_data = load_damage_data(args)
    
    old_settings = np.seterr(all = 'ignore')
    if method_name == 'ae' or method_name== 'mechae':
        train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
        model, trainer = choose_model(method_name, train_data, test_data, args, False, 0)
        train_mseloss, _ = model_test_data(model, trainer, train_data)
        test_mseloss, _ = model_test_data(model, trainer, test_data)
    else:
        models, train_mseloss, test_mseloss = detection_model_train_test(method_name, data)
        
    if method_name == 'ae' or method_name== 'mechae': # test damage case
        damage_mseloss, _ = model_test_data(model, trainer, damage_data)
        metrics = get_detection_metrics(args, test_mseloss, damage_mseloss, is_other_methods=False)
    else:
        damage_mseloss = classification_model_predict(models, damage_data)
        metrics = get_detection_metrics(args, test_mseloss, damage_mseloss, is_other_methods=True)
    os.makedirs(f'data/metric-detection-{args.experiment}', exist_ok=True)
    np.savetxt(f'data/metric-detection-{args.experiment}/{method_name}.txt', metrics)
    print(f'------{method_name} detection metric saved')
    np.seterr(**old_settings)
    return metrics


def get_normx(test_d):
    test_loader = DataLoader(test_d, batch_size=1, shuffle=False)
    return np.mean( [torch.norm(x[0,:,:],dim=0).detach().numpy() for i,x in enumerate(test_loader)], axis=0)

def localization_crack_bc(model_name, data, args, save_name=None):
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    damage_data = load_damage_data(args)
    save_name = f'{model_name}-{args.experiment}'

    if model_name == 'SPIRIT':
        from spirit.pipeline import Pipeline, SpiritStage
        def get_default_pipeline():
            spirit = SpiritStage(ispca=False,thresh=0.01,ebounds=(0.96,1.1), startm=52)
            pipeline = Pipeline()
            pipeline.append_stage(spirit)
            return pipeline
        pipeline = get_default_pipeline()
        input = {}
        mat_undmg = train_data.reshape(-1,2*args.n_sensors)
        mat_undmg2 = test_data.reshape(-1,2*args.n_sensors)

        input['data'] = mat_undmg.T
        output_undmg = pipeline.run(input)
        w_undmg = output_undmg['projection']

        input['data'] = mat_undmg2.T
        output_undmg2 = pipeline.run(input)
        w_undmg2 = output_undmg2['projection']
        
        loss_scores_baseline = np.array([np.sqrt((w_undmg2[i,0]-w_undmg[i,0])**2 + (w_undmg2[i,1]-w_undmg[i,1])**2) for i in range(2*args.n_sensors)])
        mat_dmg = damage_data.reshape(-1,2*args.n_sensors)
        input['data'] = mat_dmg.T
        output_dmg = pipeline.run(input)
        w_dmg = output_dmg['projection']  
        loss_scores2 = np.array([np.sqrt((w_dmg[i,0]-w_undmg[i,0])**2 + (w_dmg[i,1]-w_undmg[i,1])**2) for i in range(2*args.n_sensors)])
        p_score = get_damage_score(loss_scores2, loss_scores_baseline, args.n_sensors, alpha=args.alpha)
        
    else:
        model, trainer = choose_model(model_name, train_data, test_data, args, False, 0)
        _, train_normloss = model_test_data(model, trainer, train_data)
        _, test_normloss = model_test_data(model, trainer, test_data)

        norm = get_normx(train_data)
        loss_scores_baseline = np.abs( np.mean(train_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
        damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
        loss_scores_damage_test = np.abs( np.mean(damage_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
        p_score = get_damage_score(loss_scores_damage_test, loss_scores_baseline, args.n_sensors, alpha=args.alpha)
    
    fig, ax = plt.subplots(figsize=(4, 5)) # plot localization
    if args.experiment == 'crack':
        plot_crack_damage_score(p_score, is_crack=True, is_bar=args.is_bar)
    elif args.experiment == 'bc':
        plot_bc_damage_score(ax, p_score, is_damage=True, is_bar=args.is_bar)
    if model_name == 'ae':
        title_name = 'Autoencoder' 
    elif model_name == 'mechae':
        title_name = 'Mechanics-AE'
    elif model_name == 'SPIRIT':
        title_name = 'SPIRIT'
    plt.title(f'{title_name}', fontsize=18)
    
    if args.is_save:
        os.makedirs('figs/localization', exist_ok=True)
        plt.savefig(f'figs/localization/{save_name}.svg', dpi=1000, transparent=True)
        plt.close()
        print(f'------{model_name} localization figure saved at figs/localization/{save_name}.svg\n')
    
    
def plot_crack_damage_score(p_score, is_crack=True, is_bar=True, cmap='viridis'):
    def get_crack_info():
        l = 4
        cpx = [21.195, 21.195+l, 21.195+l, 21.195, 21.195]
        cpy = [22.910, 22.910, 23.310, 23.310, 22.910]
        angle = 0
        w = 23.310-22.910
        x = 21.195 + l/2
        y = (22.910+23.310)/2
        return cpx, cpy, angle, l, w, x, y
    cpx, cpy, _, _, _, _, _ = get_crack_info()
    px, py, index_s, xv, yv, xs, ys = get_geo_info()
    
    mapping = np.array([1,2,3,4,21,5,6,7,8,22,9,10,11,12,23,13,14,15,16,24,17,18,19,25,20,26])-1
    p_score = p_score[mapping]

    ss = np.insert(p_score, index_s, np.nan)
    ss = np.insert(ss,28,ss[28])
    score2d = ss.reshape(6,5)
    if not is_bar:
        vmax = max(10, np.max(p_score)) if is_crack else 1
        cs = plt.contourf(xv, yv, score2d, vmax=vmax, levels=np.linspace(0, vmax, 21) if not is_crack else 21, cmap=cmap)
    else:
        cs = plt.contourf(xv, yv, score2d, levels=np.linspace(0, 1.0, 21) if not is_crack else 21, cmap=cmap)
    plt.fill(px, py, label='geometry', color='grey', alpha=0.15)
    if is_crack:
        plt.fill(cpx, cpy, label='crack', color='r',)
    plt.scatter(xs, ys, color='black', marker='.', label='sensors', s=12)
    if is_bar:
        cbar = plt.colorbar(cs, shrink=0.2, pad=0.04, ticks=np.arange(0,1.05,0.5)) if not is_crack else plt.colorbar(cs,ticks=np.linspace(0,np.ceil(max(p_score)),5))
        cbar.ax.tick_params(labelsize=16) 
    plt.xlim([-1,37])
    plt.ylim([-1,46])
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')
    
    
def plot_bc_damage_score(ax, p_score, is_damage=True, is_bar=True, cmap='viridis'):
    px, py, index_s, xv, yv, xs, ys = get_geo_info()
    ss = np.insert(p_score, index_s, np.nan)
    ss = np.insert(ss,28,ss[28])
    score2d = ss.reshape(6,5)

    if not is_bar:
        vmax = np.max(p_score)
        cs = plt.contourf(xv, yv, score2d, vmax=vmax, levels=np.linspace(0, vmax, 21) if not is_damage else 21, cmap=cmap)
    else:
        cs = plt.contourf(xv, yv, score2d, levels=np.linspace(0, 1.0, 21) if not is_damage else 21, cmap=cmap)
    plt.fill(px,py,label='geometry',color='grey',alpha=0.15)
    
    if is_damage:
        import matplotlib.patches as patches
        ellipse = patches.Ellipse((29.5, 3), width=5, height=5, edgecolor='r', linewidth=2,
                                  facecolor='none', linestyle='--', label='Bolt loose')
        ax.add_patch(ellipse)

    plt.scatter(xs, ys, color='black', marker='.', label='sensors', s=12)
    if is_bar:
        cbar = plt.colorbar(cs, ticks=np.arange(0,1.05,0.5)) if not is_damage else plt.colorbar(cs,ticks=np.linspace(0,np.ceil(max(p_score)),5))
        cbar.ax.tick_params(labelsize=16) 
    plt.xlim([-1,37])
    plt.ylim([-1,46])
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')


def localization_simu(model_name, data, args, case=22, group_fig=True, crack_text=True):
    train_data, test_data = train_test_split(data, train_size=0.7, random_state=24)
    save_name = f'{model_name}-{args.experiment}'
    if model_name == 'SPIRIT':
        from spirit.pipeline import Pipeline, SpiritStage
        def get_default_pipeline():
            spirit = SpiritStage(ispca=False,thresh=0.01,ebounds=(0.96,1.1), startm=52)
            pipeline = Pipeline()
            pipeline.append_stage(spirit)
            return pipeline
        pipeline = get_default_pipeline()
        input = {}
        mat_undmg = train_data.reshape(-1,2*args.n_sensors)
        mat_undmg2 = test_data.reshape(-1,2*args.n_sensors)

        input['data'] = mat_undmg.T
        output_undmg = pipeline.run(input)
        w_undmg = output_undmg['projection']

        input['data'] = mat_undmg2.T
        output_undmg2 = pipeline.run(input)
        w_undmg2 = output_undmg2['projection']
        
        loss_scores_baseline = np.array([np.sqrt((w_undmg2[i,0]-w_undmg[i,0])**2 + (w_undmg2[i,1]-w_undmg[i,1])**2) for i in range(2*args.n_sensors)])
    
    else:
        model, trainer = choose_model(model_name, train_data, test_data, args, False, 0)
        _, train_normloss = model_test_data(model, trainer, train_data)
        _, test_normloss = model_test_data(model, trainer, test_data)

        norm = get_normx(train_data)
        loss_scores_baseline = np.abs( np.mean(train_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm

    # =============== Test damaged cases
    os.makedirs('figs/localization', exist_ok=True)
    path = f'data/data-damaged-{args.experiment}/case{case}'
    filenames = os.listdir(path)
    length_list = find_length(filenames)
    lengths = np.sort(length_list)
    
    if group_fig:
        fig, ax = plt.subplots(1,len(lengths)+1,figsize=(len(lengths)*5,4.5))
        plt.subplot(1,len(lengths)+1,1) # plot undamaged first
        p_score = get_damage_score(loss_scores_baseline, loss_scores_baseline, args.n_sensors, alpha=args.alpha)
        plot_damage_score(p_score, None, is_crack=False, is_bar=args.is_bar, crack_text=crack_text)
        
    for i,length in enumerate(lengths):
        indexs = np.where(length_list==length)[0]
        for index in indexs:
            damage_data = np.array( load( filenames[index], folder=path ).reshape(-1,12,args.n_sensors*2) )
            
            # Localization metric
            if model_name == 'SPIRIT':
                mat_dmg = damage_data.reshape(-1,2*args.n_sensors)
                input['data'] = mat_dmg.T
                output_dmg = pipeline.run(input)
                w_dmg = output_dmg['projection']  
                loss_scores2 = np.array([np.sqrt((w_dmg[i,0]-w_undmg[i,0])**2 + (w_dmg[i,1]-w_undmg[i,1])**2) for i in range(2*args.n_sensors)])
                p_score = get_damage_score(loss_scores2, loss_scores_baseline, args.n_sensors, alpha=args.alpha)
                
            else:
                damage_mseloss, damage_normloss = model_test_data(model, trainer, damage_data)
                loss_scores_damage_test = np.abs( np.mean(damage_normloss,axis=0) - np.mean(test_normloss,axis=0) ) / norm
                p_score = get_damage_score(loss_scores_damage_test, loss_scores_baseline, args.n_sensors, alpha=args.alpha)

            # Plots for localization
            plt.subplot(1,len(lengths)+1,i+1+1) if group_fig else plt.subplots(figsize=(5, 5))
            plot_damage_score(p_score, filenames[index], is_crack=True, cmap='viridis', 
                                    is_bar=args.is_bar, crack_text=crack_text)
            if args.is_save and not group_fig:
                plt.savefig(f'figs/localization/{save_name}-case-{case}-{int(length*1e3)}mm.svg', dpi=1000, transparent=True)
                plt.close()
                
    if args.is_save and group_fig:
        plt.savefig(f'figs/localization/{save_name}-case-{case}-group.svg', dpi=1000, transparent=True)
    print(f'------figure saved at figs/localization/{save_name}-case-{case}-group.svg\n')
    plt.close()


def plot_damage_score(p_score, name, is_crack=True, cmap='viridis', is_bar=True, crack_text=True):
    def get_crack_info(name):
        idx1 = name.find('-x') 
        idx2 = name.find('-y')
        idx3 = name.find('-l')
        idx4 = name.find('-w')
        idx5 = name.find('-a')
        idx6 = name.find('-date')
        
        x = float(name[idx1+2:idx2]) # +2 to skip ordering '-x' itself
        y = float(name[idx2+2:idx3])
        l = float(name[idx3+2:idx4])
        w = float(name[idx4+2:idx5])
        angle = int(name[idx5+2:idx6])
    
        x12 = x + l/2*np.cos(np.deg2rad(angle))
        y12 = y + l/2*np.sin(np.deg2rad(angle))
        
        x1 = x12 - w/2*np.sin(np.deg2rad(angle))
        y1 = y12 + w/2*np.cos(np.deg2rad(angle))
        x2 = x12 + w/2*np.sin(np.deg2rad(angle))
        y2 = y12 - w/2*np.cos(np.deg2rad(angle))
        
        x34 = x - l/2*np.cos(np.deg2rad(angle))
        y34 = y - l/2*np.sin(np.deg2rad(angle))
        
        x4 = x34 - w/2*np.sin(np.deg2rad(angle))
        y4 = y34 + w/2*np.cos(np.deg2rad(angle))
        x3 = x34 + w/2*np.sin(np.deg2rad(angle))
        y3 = y34 - w/2*np.cos(np.deg2rad(angle))
        
        cpx = np.multiply([x1,x2,x3,x4,x1],100) #100 due to unit scale
        cpy = np.multiply([y1,y2,y3,y4,y1],100)
        return cpx, cpy, angle, l, w, x, y
    
    def get_geo_info_simu():
        px = [-30,30,30,10,-10,-30,-30];
        py = [-30,-30,10,30,30,10,-30];

        x0 = [0,0,1,6]
        y0 = [5,6,6,6]
        s0 = np.multiply(y0,7) + x0
        s1 = np.arange(4)
        index_s=s0-s1
        xv, yv = np.meshgrid(np.arange(-25,30,8),np.arange(-25,30,8))

        xs = xv.reshape(-1,1)
        ys = yv.reshape(-1,1)
        xs = np.delete(xs,s0)
        ys = np.delete(ys,s0)
        return px, py, index_s, xv, yv, xs, ys
    
    px, py, index_s, xv, yv, xs, ys = get_geo_info_simu()

    ss = np.insert(p_score, index_s, np.nan)
    score2d = ss.reshape(7,7)
    if not is_bar:
        vmax = max(10, np.max(p_score)) if is_crack else 1
        cs = plt.contourf(xv, yv, score2d, vmax=vmax, levels=np.linspace(0, vmax, 21) if not is_crack else 21, cmap=cmap)
    else:
        cs = plt.contourf(xv, yv, score2d, levels=np.linspace(0, 1.0, 21) if not is_crack else 21, cmap=cmap)
    plt.fill(px, py, label='geometry', color='grey', alpha=0.15)
    if is_crack:
        cpx, cpy, _, L, _, _, _ = get_crack_info(name)
        plt.fill(cpx, cpy, label='crack', color='r',)
        if crack_text:
            plt.title(f'$l$={L*1000:.0f}mm', fontsize=20, color='blue')
    if not is_crack and crack_text:
        plt.title('intact', fontsize=20, color='blue')
    plt.scatter(xs, ys, color='black', marker='.', label='sensors', s=12)
    if is_bar:
        cbar = plt.colorbar(cs,ticks=np.arange(0,1.05,0.5)) if not is_crack else plt.colorbar(cs,ticks=np.linspace(0,np.ceil(max(p_score)),5))
        cbar.ax.tick_params(labelsize=16) 
    plt.xlim([-33,33])
    plt.ylim([-35,36])
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')
    
    