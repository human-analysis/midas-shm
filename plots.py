# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : MIDAS-shm
# //   File        : plots.py
# //   Description : part of the plotting functions
# //
# //   Created On: 8/19/2024
# /////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def get_detection_metric_data_all(length, path):
    m1 = np.loadtxt(path + f'/IF-{length}-all.txt')
    m2 = np.loadtxt(path + f'/OCSVM-{length}-all.txt')
    m3 = np.loadtxt(path + f'/LODA-{length}-all.txt')
    m4 = np.loadtxt(path + f'/AE-{length}-all.txt')
    m5 = np.loadtxt(path + f'/MIAE-{length}-all.txt')
    return (m1,m2,m3,m4,m5)

def get_detection_metric_data_temp(args):
    path = args.dirs
    m1 = np.loadtxt(path + '/IF-0-all.txt')
    m2 = np.loadtxt(path + '/OCSVM-0-all.txt')
    m3 = np.loadtxt(path + '/LODA-0-all.txt')
    m4 = np.loadtxt(path + '/AE-0-all.txt')
    m5 = np.loadtxt(path + '/MIAE-0-all.txt')
    n0 = np.stack((m1,m2,m3,m4,m5))
    if not args.test_damage:
        m1 = np.loadtxt(path + f'/IF-temp-{args.temp}.txt')
        m2 = np.loadtxt(path + f'/OCSVM-temp-{args.temp}.txt')
        m3 = np.loadtxt(path + f'/LODA-temp-{args.temp}.txt')
        m4 = np.loadtxt(path + f'/AE-temp-{args.temp}.txt')
        m5 = np.loadtxt(path + f'/MIAE-temp-{args.temp}.txt')
    elif args.test_damage:
        m1 = np.loadtxt(path + f'/IF-damaged-temp-{args.temp}.txt')
        m2 = np.loadtxt(path + f'/OCSVM-damaged-temp-{args.temp}.txt')
        m3 = np.loadtxt(path + f'/LODA-damaged-temp-{args.temp}.txt')
        m4 = np.loadtxt(path + f'/AE-damaged-temp-{args.temp}.txt')
        m5 = np.loadtxt(path + f'/MIAE-damaged-temp-{args.temp}.txt')
    n1 = np.stack((m1,m2,m3,m4,m5))
    return n0, n1

def get_std(data):
    mean_for_each_data_point = np.mean(data, axis=0)
    std_above_mean_list = []
    std_below_mean_list = []
    for col in range(data.shape[1]):
        above_mean = data[data[:, col] > mean_for_each_data_point[col], col]
        below_mean = data[data[:, col] < mean_for_each_data_point[col], col]
        
        std_above_mean = np.std(above_mean, ddof=1)  # Sample standard deviation
        std_below_mean = np.std(below_mean, ddof=1)
        
        std_above_mean_list.append(std_above_mean)
        std_below_mean_list.append(std_below_mean)
    return np.array(std_below_mean_list), np.array(std_above_mean_list)
    
def detection_simu_noise(args, damage_level): # compare the 5 methods in detection
    cracks = [0.015, 0.03, 0.04]
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'AE', 'MIAE']
    metrics_task = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors       = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    
    d1 = get_detection_metric_data_all(cracks[0], path = args.dirs)
    d2 = get_detection_metric_data_all(cracks[1], path = args.dirs)
    d3 = get_detection_metric_data_all(cracks[2], path = args.dirs)
    
    bar_width = 0.08
    gap_width = 0.04
    pos_offset = -0.03
    x_positions = np.arange(len(metrics_task)) - (len(metrics_task) * (bar_width + gap_width) / 2)
    
    s = 3 #capsize for error bar
    error_color = '#5d5d5d'

    crack_sizes = [0.08, 0.2, 0.4]
    crack_names  = [f'Small crack ({crack_sizes[0]*10}cm)', 
                    f'Medium crack ({crack_sizes[1]*10}cm)', 
                    f'Large crack ({crack_sizes[2]*10}cm)']
    n_cracks = len(crack_names)
    n_cracks = 1
    data_zero_crack = get_detection_metric_data_all(0, path = args.dirs)
    
    length1 = 3.5 # plot width for no crack case
    length2 = 6+0.5 # plot width for crack cases
    fig, axs = plt.subplots(1, 1+n_cracks, figsize=(length1 + length2*n_cracks, 3.8))
    grid = plt.GridSpec(1,  1+n_cracks, width_ratios=[length1] + [length2]*n_cracks)  
    ax0 = plt.subplot(grid[0, 0])
    
    bars = plt.bar(methods_task, data_zero_crack, color=colors)
    ax0.set_xticks([])
    for bar, accuracy in zip(bars, data_zero_crack):
        if accuracy < 0.95:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{accuracy:.2f}', 
                     ha='center', va='bottom', fontsize=14)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.08, f'{accuracy:.2f}', 
                     ha='center', va='bottom', fontsize=14)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.set_ylim(0,1)
    ax0.set_xlabel('Accuracy', fontsize=16)
    ax0.set_title('No crack', fontsize=18)

    ### crack plotting
    j = damage_level-1 #index
    data = (d1,d2,d3)[j]
    ax = plt.subplot(grid[0, 1])
    ax.set_title(crack_names[j], fontsize=18)
    for i, method in enumerate(methods_task):
        ax.bar(x_positions + i * (bar_width + gap_width),
            np.mean(data[i], axis=0), width=bar_width, label=method, color=colors[i], 
            edgecolor='black', linewidth=1)
        asymmetric_error = np.stack(get_std(data[i]))
        ax.errorbar(x_positions + i * (bar_width + gap_width), np.mean(data[i], axis=0), 
                        yerr=asymmetric_error, fmt='.', capsize=s, color=error_color)
    
    ax.set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
    ax.set_xticklabels(metrics_task, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(0,1)
    
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='center left', 
                 bbox_to_anchor=(1, 0.5), 
                 fontsize=16, fancybox=True, frameon=True, ncol=1, 
                 columnspacing=3, handleheight=1)
    plt.tight_layout(w_pad=10)

    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/simu_noise.svg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Save as high-resolution PNG

def localization_simu_progress():
    # Localization is manually determined in a excel file in data folder
    df = pd.read_excel('data/localization.xlsx', header=None, index_col=None)
    data = df.iloc[1:, 1:4].to_numpy()
    data[data == 0] = 50 # remove those failed to locate 
    
    crack_lengths = np.unique(data).astype(int)
    m = np.zeros( (len(crack_lengths),  data.shape[1]) )
    
    for j in range(m.shape[1]):
        for i, crack_length in enumerate(crack_lengths):
            if i > 0:
                m[i,j] = np.count_nonzero(data[:,j] == crack_length) + m[i-1,j]
            else:
                m[i,j] = np.count_nonzero(data[:,j] == crack_length)
    
    m2 = np.insert(m[:-1,:], 0, np.zeros((1, 3)), axis=0)/37*100
    m2 = m2.astype(int)
    xx = np.insert(crack_lengths[:-1], 0, 0)/10
    
    m2 = m2/100 #change to decimals

    colors = ['#D7662A', '#1FBF8F', '#DD6CAF']
    fig, ax = plt.subplots(figsize=(12, 3.5))
    plt.plot(xx, m2[:,0], label='MIAE', linewidth=2,
             marker='*', markersize=10, color=colors[2]) #'#C15D98'
    plt.plot(xx, m2[:,1], '--', label='Autoencoder', linewidth=2, 
             marker='^', markersize=10, color=colors[1]) #'#1A9E76'
    plt.plot(xx, m2[:,2], '-.', label='SPIRIT', linewidth=2, 
             marker='x', markersize=10, color=colors[0]) #'#D7662A'
    
    plt.legend(fontsize=12, loc=2, frameon=False)
    plt.xticks(xx, [0,0.8,1,1.5,2,3,4], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Crack length (cm)', fontsize=14)
    plt.ylabel('Localization accuracy', fontsize=14)
    plt.grid('on')
    plt.tight_layout()
    
    os.makedirs('figs/localization', exist_ok=True)
    plt.savefig('figs/localization/progress.svg', dpi=1000)

def detection_simu_temp(temps, args):
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'AE', 'MIAE']
    metrics_task = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors       = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    
    args.test_damage = False
    args.temp = temps[0]
    _, undamaged1 = get_detection_metric_data_temp(args)
    args.temp = temps[1]
    _, undamaged2 = get_detection_metric_data_temp(args)
    
    args.test_damage = True
    args.temp = temps[2]
    _, damaged1 = get_detection_metric_data_temp(args)
    args.temp = temps[3]
    _, damaged2 = get_detection_metric_data_temp(args)
    
    bar_width = 0.08
    gap_width = 0.04
    pos_offset = -0.03
    x_positions = np.arange(len(metrics_task)) - (len(metrics_task) * (bar_width + gap_width) / 2)
    
    length1 = 3.5 # plot width for no crack case
    length2 = 6 # plot width for crack cases
    
    fig, axs = plt.subplots(1, 4, figsize=(length1*2 + length2*2, 3.5)) #3.8
    grid = plt.GridSpec(1, 4, width_ratios=[length1, length2]*2)
    
    data = (undamaged1, undamaged2)
    for i, j in enumerate([0,2]):
        data_zero_crack = data[i]
        ax0 = plt.subplot(grid[0, j])
        bars = plt.bar(methods_task, data_zero_crack, color=colors)
        ax0.set_xticks([])
        for bar, accuracy in zip(bars, data_zero_crack):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{accuracy:.2f}', 
                     ha='center', va='bottom', fontsize=14)
        ax0.tick_params(axis='both', which='major', labelsize=16)
        ax0.set_ylim(0,1)
        ax0.set_xlabel('Accuracy', fontsize=16)
        ax0.set_title(f'No crack ({temps[i]}$^\circ$C)', fontsize=18)

    data = (damaged1, damaged2)
    for ii, j in enumerate([1,3]):
        data_task1 = data[ii]
        ax = plt.subplot(grid[0, j])
        for i, method in enumerate(methods_task):
            ax.bar(x_positions + i * (bar_width + gap_width),
                data_task1[i], width=bar_width, label=method, color=colors[i], 
                edgecolor='black', linewidth=1)
        ax.set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
        ax.set_xticklabels(metrics_task, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f'Crack detection ({temps[ii+2]}$^\circ$C)', fontsize=18)
        ax.set_ylim(0,1)
    
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.13), 
                     fontsize=16, fancybox=True, frameon=True, ncol=len(methods_task), 
                     columnspacing=3)
    plt.tight_layout(w_pad=5+5)
    
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/simu_temp.svg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Save as high-resolution PNG

def detection_simu(args): # compare the 5 methods in detection
    crack_sizes = args.cracks_sizes#[0.015, 0.03, 0.04]
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'AE', 'MIAE']
    metrics_task = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors       = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    s = 3 #capsize for error bar
    error_color = '#5d5d5d'
    
    d1 = get_detection_metric_data_all(crack_sizes[0], path = args.dirs)
    d2 = get_detection_metric_data_all(crack_sizes[1], path = args.dirs)
    d3 = get_detection_metric_data_all(crack_sizes[2], path = args.dirs)
    data_zero_crack = get_detection_metric_data_all(0, path = args.dirs)
    
    bar_width = 0.08
    gap_width = 0.04
    pos_offset = -0.03
    x_positions = np.arange(len(metrics_task)) - (len(metrics_task) * (bar_width + gap_width) / 2)
    
    crack_names  = [f'Small crack ({crack_sizes[0]*10}cm)', 
                    f'Medium crack ({crack_sizes[1]*10}cm)', 
                    f'Large crack ({crack_sizes[2]*10}cm)']
    n_cracks = len(crack_names)
    
    length1 = 3.5 # plot width for no crack case
    length2 = 6 # plot width for crack cases
    fig, axs = plt.subplots(1, 4, figsize=(length1 + length2*n_cracks, 3.8))
    grid = plt.GridSpec(1, 4, width_ratios=[length1, length2, length2, length2])  
    ax0 = plt.subplot(grid[0, 0])
    
    bars = plt.bar(methods_task, data_zero_crack, color=colors)
    ax0.set_xticks([])
    for bar, accuracy in zip(bars, data_zero_crack):
        if accuracy < 0.93:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{accuracy:.2f}', 
                     ha='center', va='bottom', fontsize=14)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.01, f'{accuracy:.2f}', 
                     ha='center', va='bottom', fontsize=14)
            
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.set_ylim(0,1)
    ax0.set_xlabel('Accuracy', fontsize=16)
    ax0.set_title('No damage', fontsize=18)

    ### crack plotting
    for j in range(n_cracks):
        data = (d1,d2,d3)[j]
        ax = plt.subplot(grid[0, j+1])
        ax.set_title(crack_names[j], fontsize=18)
        for i, method in enumerate(methods_task):
            ax.bar(x_positions + i * (bar_width + gap_width),
                np.mean(data[i], axis=0), width=bar_width, label=method, color=colors[i], 
                edgecolor='black', linewidth=1)
            asymmetric_error = np.stack(get_std(data[i]))
            ax.errorbar(x_positions + i * (bar_width + gap_width), np.mean(data[i], axis=0), 
                            yerr=asymmetric_error, fmt='.', capsize=s, color=error_color)
        
        ax.set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
        ax.set_xticklabels(metrics_task, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(0,1)
    
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.13), 
                     fontsize=16, fancybox=True, frameon=True, ncol=len(methods_task), 
                     columnspacing=3)
    plt.tight_layout(w_pad=5)
    
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/simu.svg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Save as high-resolution PNG
    
def detection_beam_column(args):
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'AE', 'MIAE']
    metrics_task = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors       = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    
    dirs = args.dirs
    data_zero_crack = get_detection_metric_data_all(0, path=dirs)
    data_tasks = [get_detection_metric_data_all(levels, path=dirs) for levels in ['D0', 'D1', 'D2', 'D3'] ]
    # ====== plot cracks
    bar_width = 0.08
    gap_width = 0.04
    pos_offset = -0.03
    x_positions = np.arange(len(metrics_task)) - (len(metrics_task) * (bar_width + gap_width) / 2)
    
    n_cracks = len(data_tasks)
    
    length1 = 3.5 # plot width for no crack case
    length2 = 6.0 # plot width for crack cases
    fig, axs = plt.subplots(1, n_cracks+1, figsize=(length1 + length2*n_cracks, 4))
    grid = plt.GridSpec(1, n_cracks+1, width_ratios=[length1]+ [length2]*n_cracks)  
    
    # plot no damage acc
    ax0 = plt.subplot(grid[0, 0])
    bars = plt.bar(methods_task, data_zero_crack, color=colors)
    ax0.set_xticks([])
    for bar, accuracy in zip(bars, data_zero_crack):
        if accuracy>0.95:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.07, f'{accuracy:.2f}', 
                 ha='center', va='bottom', fontsize=14)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{accuracy:.2f}', 
                 ha='center', va='bottom', fontsize=14)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.set_ylim(0,1)
    ax0.set_xlabel('Accuracy', fontsize=16)
    ax0.set_title('D0', fontsize=18)

    # Plot for damages
    for j in range(n_cracks):
        ax = plt.subplot(grid[0, j+1])
        ax.set_title(f'D{j+1}', fontsize=18)
        for i, method in enumerate(methods_task):
            ax.bar(x_positions + i * (bar_width + gap_width),
                data_tasks[j][i], width=bar_width, label=method, color=colors[i], 
                edgecolor='black', linewidth=1)
            
        ax.set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
        ax.set_xticklabels(metrics_task, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(0,1)

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.13), 
                     fontsize=16, fancybox=True, frameon=True, ncol=len(methods_task), 
                     columnspacing=3)
    plt.tight_layout(w_pad=5)
    
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/beam_column.svg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Save as high-resolution PNG
    
def localization_simu_fewer_sensors_progress(args, dist_metric):
    metric = np.zeros( (len(args.cracks_sizes), len(dist_metric)) )
    for i,a in enumerate(dist_metric):
        a = np.where(a <= 13.5, 1, 0)
        metric[:,i] = sum(a,0)/len(a)
        
    colors = ['#D7662A', '#1FBF8F', '#DD6CAF']
    xx = np.array(args.cracks_sizes)*100
    markers = ['x', '^', '*']
    methods = ['SPIRIT', 'AE', 'MIAE'] #Mechanics-AE
    
    fig, axs = plt.subplots(1, 1, figsize=(4.8, 5))
    for i in range(len(methods)):
        plt.plot(xx, metric[:,i]*100/100, label=methods[i], linewidth=2,
                  marker=markers[i], markersize=10, color=colors[i]) #'#C15D98'
    plt.legend(fontsize=12, loc=2, frameon=False)
    plt.xticks(xx, [0.8, 1.5, 2, 3, 4], fontsize=14)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.xlabel('Crack length (cm)', fontsize=14)
    plt.ylabel('Localization accuracy', fontsize=14)
    plt.grid('on')
    plt.tight_layout()
    os.makedirs('figs/localization/', exist_ok=True)
    plt.savefig('figs/localization/progress_fewer_sensors.svg', dpi=1000)

def detection_simu_fewer_sensors(y, pick_n_sensor_list):
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'Autoencoder', 'MIAE']
    colors = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    subplot_titles = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUROC']
    
    x = pick_n_sensor_list
    import math
    xticks = range(math.floor(min(x)), math.ceil(max(x))+3)
    n_plots = len(subplot_titles)
    fig, axs = plt.subplots(1, n_plots, figsize=(3.8*n_plots, 3.8))
    for i in range(n_plots):
        plt.subplot(1, n_plots, i+1)
        for j, method in enumerate(methods_task): # ['ae', 'mechae', 'OCSVM', 'LODA', 'IF']
            y_mean = np.mean(y[:,j,:,i], axis=1)
            std_lower, std_upper = get_std(y[:,j,:,i].transpose())
            
            marker = 'o' if method == 'Mechanics-AE' else '^'
            plt.plot(x, y_mean, label=methods_task[j], linewidth=2, marker=marker, 
                      markersize=10, color=colors[j])
            plt.fill_between(x, y_mean-std_lower/5, y_mean+std_upper/5, alpha=0.4, 
                              color=colors[j])
            
        plt.xticks(xticks)
        plt.grid('on')
        plt.title(subplot_titles[i], fontsize=18)
        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16)
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=9)
        
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', 
                      bbox_to_anchor=(0.5, 1.12),  
                      fontsize=16, fancybox=True, frameon=True,
                      ncol=len(labels), labelspacing = 1)
    fig.supxlabel('Number of sensors', fontsize=18)
    plt.tight_layout(w_pad=2)
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/simu_fewer_sensors_metric.svg', dpi=1000, bbox_inches='tight')
