# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : Mechanics-AE for damage detection and localization
# //   File        : plots.py
# //   Description : part of the plotting functions
# //
# //   Created On: 3/7/2024
# /////////////////////////////////////////////////////////////////


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import os

def get_detection_metric_data_all(length):
    path = 'data/metric-detection-simu-all/'
    m1 = np.loadtxt(path + f'IF-{length}-all.txt')
    m2 = np.loadtxt(path + f'OCSVM-{length}-all.txt')
    m3 = np.loadtxt(path + f'LODA-{length}-all.txt')
    m4 = np.loadtxt(path + f'ae-{length}-all.txt')
    m5 = np.loadtxt(path + f'mechae-{length}-all.txt')
    return (m1,m2,m3,m4,m5)

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

    
def compare_simu_detection_plot(): # compare the 5 methods in detection
    try:
        plt.style.use(['aa', 'nature'])
    except:
        print("Warning: science plot style not found. Default plot style will be used.")
    
    cracks = [0.015, 0.03, 0.04]
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'Autoencoder', 'Mechanics-AE']
    metrics_task = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors       = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    
    d1 = get_detection_metric_data_all(cracks[0])
    d2 = get_detection_metric_data_all(cracks[1])
    d3 = get_detection_metric_data_all(cracks[2])
    
    bar_width = 0.08
    gap_width = 0.04
    pos_offset = -0.03
    x_positions = np.arange(len(metrics_task)) - (len(metrics_task) * (bar_width + gap_width) / 2)
    
    s = 3 #capsize for error bar
    error_color = '#5d5d5d'

    fig, axs = plt.subplots(1, 3, figsize=(6*3, 3.8))
    # Plot for Task 1
    axs[0].set_title('Small crack', fontsize=18)
    for i, method in enumerate(methods_task):
        axs[0].bar(x_positions + i * (bar_width + gap_width),
            np.mean(d1[i], axis=0), width=bar_width, label=method, color=colors[i], 
            edgecolor='black', linewidth=1)
        
        asymmetric_error = np.stack(get_std(d1[i]))
        axs[0].errorbar(x_positions + i * (bar_width + gap_width), np.mean(d1[i], axis=0), 
                        yerr=asymmetric_error, fmt='.', capsize=s, color=error_color)
    
    axs[0].set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
    axs[0].set_xticklabels(metrics_task, fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_ylim(0,1)
    
    # Plot for Task 2
    axs[1].set_title('Medium crack', fontsize=18)
    for i, method in enumerate(methods_task):
        axs[1].bar(x_positions + i * (bar_width + gap_width),
            np.mean(d2[i], axis=0), width=bar_width, label=method, color=colors[i], 
            edgecolor='black', linewidth=1)
        
        asymmetric_error = np.stack(get_std(d2[i]))
        axs[1].errorbar(x_positions + i * (bar_width + gap_width), np.mean(d2[i], axis=0), 
                        yerr=asymmetric_error, fmt='.', capsize=s, color=error_color)
        
    axs[1].set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
    axs[1].set_xticklabels(metrics_task, fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].set_ylim(0,1)
    
    # Plot for Task 3
    axs[2].set_title('Large crack', fontsize=18)
    for i, method in enumerate(methods_task):
        axs[2].bar(x_positions + i * (bar_width + gap_width),
            np.mean(d3[i], axis=0), width=bar_width, label=method, color=colors[i], 
            edgecolor='black', linewidth=1)
        
        asymmetric_error = np.stack(get_std(d3[i]))
        axs[2].errorbar(x_positions + i * (bar_width + gap_width), np.mean(d3[i], axis=0), 
                        yerr=asymmetric_error, fmt='.', capsize=s, color=error_color)
        
    axs[2].set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
    axs[2].set_xticklabels(metrics_task, fontsize=16)
    axs[2].tick_params(axis='both', which='major', labelsize=16)
    axs[2].set_ylim(0,1)
    
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.13), 
                     fontsize=16, fancybox=True, frameon=True, ncol=len(methods_task), 
                     columnspacing=3)
    plt.tight_layout(w_pad=5)
    
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/mechae-ae_bar_plot.svg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Save as high-resolution PNG
    print('------figure saved at figs/detection/mechae-ae_bar_plot.svg')
    plt.close()

def localization_progress():
    try:
        plt.style.use(['aa', 'nature'])
    except:
        print("Warning: science plot style not found. Default plot style will be used.")
    
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

    colors = ['#D7662A', '#1FBF8F', '#DD6CAF']
    fig, ax = plt.subplots(figsize=(12, 3.5))
    plt.plot(xx, m2[:,0], label='Mechanics-AE', linewidth=2,
             marker='*', markersize=10, color=colors[2]) #'#C15D98'
    plt.plot(xx, m2[:,1], '--', label='Autoencoder', linewidth=2, 
             marker='^', markersize=10, color=colors[1]) #'#1A9E76'
    plt.plot(xx, m2[:,2], '-.', label='SPIRIT', linewidth=2, 
             marker='x', markersize=10, color=colors[0]) #'#D7662A'
    
    plt.legend(fontsize=12, loc=2, frameon=False)
    plt.xticks(xx, [0,0.8,1,1.5,2,3,4], fontsize=14)
    plt.yticks(fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel('Crack length (cm)', fontsize=14)
    plt.ylabel('Localizated cases', fontsize=14)
    plt.grid('on')
    plt.tight_layout()
    
    os.makedirs('figs/localization', exist_ok=True)
    plt.savefig('figs/localization/localization-progress.svg', dpi=1000)
    print('------figure saved at figs/localization/localization-progress.svg')
    plt.close()

def get_detection_metric_data(experiment):
    path = f'data/metric-detection-{experiment}/'
    m1 = np.loadtxt(path + 'IF.txt')
    m2 = np.loadtxt(path + 'OCSVM.txt')
    m3 = np.loadtxt(path + 'LODA.txt')
    m4 = np.loadtxt(path + 'ae.txt')
    m5 = np.loadtxt(path + 'mechae.txt')
    data_task = {
        'Isolation Forest': m1,
        'OCSVM': m2,
        'LODA': m3,
        'Autoencoder': m4,
        'Mechanics-AE': m5,}
    return data_task
    
def crack_bc_detection_plot():
    methods_task = ['Isolation Forest', 'OCSVM', 'LODA', 'Autoencoder', 'Mechanics-AE']
    metrics_task = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors       = ['#A5A5A5', '#6C77DD', '#DDD26C', '#6CDD99', '#DD6CAF']
    
    data_task1 = get_detection_metric_data('crack')
    data_task2 = get_detection_metric_data('bc')
    
    bar_width = 0.08
    gap_width = 0.04
    pos_offset = -0.03
    x_positions = np.arange(len(metrics_task)) - (len(metrics_task) * (bar_width + gap_width) / 2)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    # Plot for Task 1
    axs[0].set_title('Crack detection', fontsize=18)
    for i, method in enumerate(methods_task):
        axs[0].bar(x_positions + i * (bar_width + gap_width),
            data_task1[method], width=bar_width, label=method, color=colors[i], 
            edgecolor='black', linewidth=1)
    axs[0].set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
    axs[0].set_xticklabels(metrics_task, fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_ylim(0,1)
    
    # Plot for Task 2
    axs[1].set_title('Bolt loose detection', fontsize=18)
    for i, method in enumerate(methods_task):
        axs[1].bar(x_positions + i * (bar_width + gap_width),
            data_task2[method], width=bar_width, label=method, color=colors[i], 
            edgecolor='black', linewidth=1)
    axs[1].set_xticks([pos + pos_offset for pos in range(len(metrics_task))])
    axs[1].set_xticklabels(metrics_task, fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].set_ylim(0,1)
    
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.12), 
                     fontsize=16, fancybox=True, frameon=True, ncol=len(methods_task), 
                     columnspacing=3)
    plt.tight_layout(w_pad=6)
    
    os.makedirs('figs/detection', exist_ok=True)
    plt.savefig('figs/detection/crack-bc_bar_plot.svg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Save as high-resolution PNG
    print('------figure saved at figs/detection/crack-bc_bar_plot.svg')
    plt.close()

