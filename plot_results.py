#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 02:43:23 2020

@author: porthos
"""


import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def draw_multi_lr():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/20_10_20-9_00[lr0.00001]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/20_10_20-6_00[lr0.0001]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/20_10_20-18_30[lr0.001]' + '/metrics.json')
    metrics_4 = load_json_arr(experiment_folder + '/20_10_20-15_00[lr0.01]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'] #blue, orange, green, red
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1], 
        [x['total_loss'] for x in metrics_1], color='#D62728')
    plt.plot(
        [x['iteration'] for x in metrics_2], 
        [x['total_loss'] for x in metrics_2], color='#FF7F0E')
    plt.plot(
        [x['iteration'] for x in metrics_3], 
        [x['total_loss'] for x in metrics_3], color='#2CA02C')
    plt.plot(
        [x['iteration'] for x in metrics_4], 
        [x['total_loss'] for x in metrics_4], color='#1F77B4')
    plt.legend(['lr=0.00001', 'lr=0.0001', 'lr=0.001', 'lr=0.01'], loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total training loss')
    fig_1.tight_layout()
    # plt.title('Training Loss')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_loss_multi_lr.pdf')
    plt.show()

def draw_multi_ap50():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/20_10_20-9_00[lr0.00001]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/20_10_20-6_00[lr0.0001]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/20_10_20-18_30[lr0.001]' + '/metrics.json')
    metrics_4 = load_json_arr(experiment_folder + '/20_10_20-15_00[lr0.01]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'] #blue, orange, green, red
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']%300==299], 
        [x['bbox/AP50'] for x in metrics_1 if x['iteration']%300==299], color='#D62728')
    plt.plot(
        [x['iteration'] for x in metrics_2 if x['iteration']%300==299], 
        [x['bbox/AP50'] for x in metrics_2 if x['iteration']%300==299], color='#FF7F0E')
    plt.plot(
        [x['iteration'] for x in metrics_3 if x['iteration']%300==299], 
        [x['bbox/AP50'] for x in metrics_3 if x['iteration']%300==299], color='#2CA02C')
    plt.plot(
        [x['iteration'] for x in metrics_4 if x['iteration']%300==299], 
        [x['bbox/AP50'] for x in metrics_4 if x['iteration']%300==299], color='#1F77B4')
    plt.legend(['lr=0.00001', 'lr=0.0001', 'lr=0.001', 'lr=0.01'], loc='lower right')
    plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_multi_lr.pdf')
    plt.show()

def draw_train_val_best_lr():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/20_10_20-18_30[lr0.001]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/20_10_20-6_00[lr0.0001]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1], 
        [x['total_loss'] for x in metrics_1], color=colors[2])
    plt.plot(
        [x['iteration'] for x in metrics_1], 
        [x['total_val_loss'] for x in metrics_1], color= colors[4])
    plt.plot(
        [x['iteration'] for x in metrics_2], 
        [x['total_loss'] for x in metrics_2], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_2], 
        [x['total_val_loss'] for x in metrics_2], color= colors[7])
    plt.legend(['training loss (lr: 0.001)', 'validation loss (lr: 0.001)', 'training loss (lr: 0.0001)', 'validation loss (lr: 0.0001)'], 
               loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total loss')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_val_loss_lr0.001-0.0001.pdf')
    plt.show()
    
def draw_lr_schedule_model():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics = load_json_arr(experiment_folder + '/21_10_20-11_00[lr1_0.0001-lr2_0.00001(@3k)]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'] #blue, orange, green, red
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics], 
        [x['total_loss'] for x in metrics], color=colors[0])
    plt.plot(
        [x['iteration'] for x in metrics], 
        [x['total_val_loss'] for x in metrics], color= colors[1])
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total loss')
    fig_1.tight_layout()
    # plt.title('lr scheduled model losses')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_val_loss_lr_sch.pdf')
    
    fig_2 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics if x['iteration']%100==99], 
        [x['bbox/AP50'] for x in metrics if x['iteration']%100==99], color='#D62728')

    # plt.legend(['lr=0.00001', 'lr=0.0001', 'lr=0.001', 'lr=0.01'], loc='lower right')
    plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    fig_2.tight_layout()
    # plt.title('lr scheduled model val AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_lr_sch.pdf')
    plt.grid(b=True, linestyle=':')
    plt.show()

if __name__ == '__main__':
    draw_multi_lr()
    draw_multi_ap50()
    draw_train_val_best_lr()
    draw_lr_schedule_model()

