#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 02:43:23 2020

@author: porthos
"""


import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['font.size'] = '13'
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
    plt.legend(['lr = $1\mathrm{e}{-5}$', 'lr = $1\mathrm{e}{-4}$', 'lr = $1\mathrm{e}{-3}$', 'lr = $1\mathrm{e}{-2}$'], loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total training loss')
    fig_1.tight_layout()
    # plt.title('Training Loss')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_loss_multi_lr.pdf')
    plt.show()

def draw_multi_lr_ap50_bbox():
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
    plt.legend(['lr = $1\mathrm{e}{-5}$', 'lr = $1\mathrm{e}{-4}$', 'lr = $1\mathrm{e}{-3}$', 'lr = $1\mathrm{e}{-2}$'], loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP_{50}$')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_multi_lr_bbox.pdf')
    plt.show()

def draw_multi_lr_ap50_key():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/20_10_20-9_00[lr0.00001]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/20_10_20-6_00[lr0.0001]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/20_10_20-18_30[lr0.001]' + '/metrics.json')
    metrics_4 = load_json_arr(experiment_folder + '/20_10_20-15_00[lr0.01]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'] #blue, orange, green, red
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']%300==299], 
        [x['keypoints/AP50'] for x in metrics_1 if x['iteration']%300==299], color='#D62728')
    plt.plot(
        [x['iteration'] for x in metrics_2 if x['iteration']%300==299], 
        [x['keypoints/AP50'] for x in metrics_2 if x['iteration']%300==299], color='#FF7F0E')
    plt.plot(
        [x['iteration'] for x in metrics_3 if x['iteration']%300==299], 
        [x['keypoints/AP50'] for x in metrics_3 if x['iteration']%300==299], color='#2CA02C')
    plt.plot(
        [x['iteration'] for x in metrics_4 if x['iteration']%300==299], 
        [x['keypoints/AP50'] for x in metrics_4 if x['iteration']%300==299], color='#1F77B4')
    plt.legend(['lr = $1\mathrm{e}{-5}$', 'lr = $1\mathrm{e}{-4}$', 'lr = $1\mathrm{e}{-3}$', 'lr = $1\mathrm{e}{-2}$'], loc='center right')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP^{kp}_{50}$')
    # plt.ylabel('Accuracy')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_multi_lr_key.pdf')
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
    plt.legend(['training loss (lr $1\mathrm{e}{-3}$)', 'validation loss (lr $1\mathrm{e}{-3}$)', 'training loss (lr $1\mathrm{e}{-4}$)', 'validation loss (lr $1\mathrm{e}{-4}$)'], 
               loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total loss')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_val_loss_lr0.001-0.0001.pdf')
    plt.show()
    
def draw_train_val_warmup_lr_sch():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/21_10_20-2_00[no_warmup-lr0.001]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/20_10_20-18_30[lr0.001]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_1 if x['iteration']<=3000], color=colors[3])
    plt.plot(
        [x['iteration'] for x in metrics_2 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_2 if x['iteration']<=3000], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_3 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_3 if x['iteration']<=3000], color=colors[7])
    
    plt.legend(['with decay [no warmup]', 'without decay [no warmup]', 'without decay [with warmup]'], loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total training loss')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_loss_warmup_lr_sch.pdf')
    plt.show()
 
def draw_ap50_warmup_lr_sch():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/21_10_20-2_00[no_warmup-lr0.001]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/20_10_20-18_30[lr0.001]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    line_1 = plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], label='$AP_{50}$ with decay [no warmup]', color=colors[3], linestyle='-')
    line_2 = plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], label='$AP_{50}$ without decay [no warmup]', color=colors[1], linestyle='-')
    line_3 = plt.plot(
        [x['iteration'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], label='$AP_{50}$ without decay [with warmup]', color=colors[7], linestyle='-')
    line_4 = plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], label='$AP^{kp}_{50}$ with decay [no warmup]', color=colors[3], linestyle='--')
    line_5 = plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], label='$AP^{kp}_{50}$ without decay [no warmup]', color=colors[1], linestyle='--')
    line_6 = plt.plot(
        [x['iteration'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], label='$AP^{kp}_{50}$ without decay [with warmup]', color=colors[7], linestyle='--')
    
    # plt.legend(['$AP_{50}$ with decay [no warmup]', '$AP_{50}$ without decay [no warmup]', '$AP_{50}$ without decay [with warmup]',
    #             '$AP^{kp}_{50}$ with decay [no warmup]', '$AP^{kp}_{50}$ without decay [no warmup]', '$AP^{kp}_{50}$ without decay [with warmup]'], 
    #            loc='lower right')
    l1 = plt.legend(handles=[line_1[0], line_2[0], line_3[0]], loc='center left')
    plt.gca().add_artist(l1)
    l2 = plt.legend(handles=[line_4[0], line_5[0], line_6[0]], loc='lower right')
    # plt.gca().add_artist(l2)
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP$')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_warmup_lr_sch.pdf')
    plt.show()
 
# def draw_train_val_lr_sch():
#     experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
#     metrics_1 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')
#     metrics_2 = load_json_arr(experiment_folder + '/21_10_20-2_00[no_warmup-lr0.001]' + '/metrics.json')

#     colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
#     fig_1 = plt.figure()
#     plt.plot(
#         [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
#         [x['total_loss'] for x in metrics_1 if x['iteration']<=3000], color=colors[3])
#     plt.plot(
#         [x['iteration'] for x in metrics_2 if x['iteration']<=3000], 
#         [x['total_loss'] for x in metrics_2 if x['iteration']<=3000], color=colors[5])
#     plt.legend(['decay @ 1000 iter.', 'without decay'], 
#                loc='upper right')
#     plt.xlabel('Iterations')
#     plt.ylabel('Total training loss')
#     fig_1.tight_layout()
#     # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
#     plt.grid(b=True, linestyle=':')
#     # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_val_lr_sch.pdf')
#     plt.show()
    
# def draw_ap50_lr_sch():
#     experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
#     metrics_1 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')
#     metrics_2 = load_json_arr(experiment_folder + '/21_10_20-2_00[no_warmup-lr0.001]' + '/metrics.json')
    
#     colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
#     fig_1 = plt.figure()
#     plt.plot(
#         [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
#         [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[3], linestyle='-')
#     plt.plot(
#         [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
#         [x['bbox/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5], linestyle='-')
#     plt.plot(
#         [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
#         [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[3], linestyle='--')
#     plt.plot(
#         [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
#         [x['keypoints/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5], linestyle='--')
#     plt.legend(['$AP_{50}$ with decay @ 1000 iter.', '$AP_{50}$ without decay', '$AP^{kp}_{50}$ with decay @ 1000 iter.', '$AP^{kp}_{50}$ without decay'], loc='lower right')
#     plt.xlabel('Iterations')
#     # plt.ylabel('Accuracy')
#     fig_1.tight_layout()
#     plt.grid(b=True, linestyle=':')
#     # plt.title('Validation AP50')
#     # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_lr_sch.pdf')
#     plt.show()
    
def draw_train_loss_adam_sgd():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/23_10_20-3_00[adam-lr0.0001-no_warmup-wd]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/21_10_20-23_00[adam-lr0.001-wd]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')


    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_1 if x['iteration']<=3000], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_2 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_2 if x['iteration']<=3000], color=colors[5])
    plt.plot(
        [x['iteration'] for x in metrics_3 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_3 if x['iteration']<=3000], color=colors[3])
    plt.legend(['Adam [lr $1\mathrm{e}{-4}$]', 'Adam [lr $1\mathrm{e}{-4}$]', 'SGD [lr $1\mathrm{e}{-3}$]'], 
               loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total training loss')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_loss_adam_sgd.pdf')
    plt.show()
    
def draw_ap50_adam_sgd():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/23_10_20-3_00[adam-lr0.0001-no_warmup-wd]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/21_10_20-23_00[adam-lr0.001-wd]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5])
    plt.plot(
        [x['iteration'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[3])
    plt.legend(['$AP_{50}$ Adam [lr $1\mathrm{e}{-4}$]', '$AP_{50}$ Adam [lr $1\mathrm{e}{-3}$]', '$AP_{50}$ SGD [lr $1\mathrm{e}{-3}$]'], loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP_{50}$')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_adam_sgd_bbox.pdf')

    fig_2 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[1], linestyle='--')
    plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5], linestyle='--')
    plt.plot(
        [x['iteration'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[3], linestyle='--')
    plt.legend(['$AP^{kp}_{50}$ Adam [lr $1\mathrm{e}{-4}$]', '$AP^{kp}_{50}$ Adam [lr $1\mathrm{e}{-3}$]', '$AP^{kp}_{50}$ SGD [lr $1\mathrm{e}{-3}$]'], loc='center right')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP^{kp}_{50}$')
    fig_2.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_adam_sgd_key.pdf')
    plt.show()

def draw_best_model():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[2])
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[0])

    plt.legend(['$AP_{50}$', '$AP^{kp}_{50}$'], loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP$')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_best_model.pdf')

    fig_2 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_1 if x['iteration']<=3000], color=colors[3])
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['total_val_loss'] for x in metrics_1 if x['iteration']<=3000], color=colors[4])
    plt.legend(['total loss', 'validation loss'], loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total Loss')
    fig_2.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_val_loss_best_model.pdf')
    plt.show()
    
    fig_3 = plt.figure()
    plt.semilogy(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['loss_box_reg'] for x in metrics_1 if x['iteration']<=3000], color=colors[7])
    plt.semilogy(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['loss_cls'] for x in metrics_1 if x['iteration']<=3000], color=colors[4])
    plt.semilogy(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['loss_keypoint'] for x in metrics_1 if x['iteration']<=3000], color=colors[2])
    plt.legend(['Box regression loss', 'Classification loss', 'Keypoint loss'], loc='center right')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    fig_3.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.title('Validation AP50')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/loss_comp_best_model.pdf')
    plt.show()
    

def draw_ap50_pretrained():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/23_10_20-5_00[pretrain-lr0.001-no_warmup_wd]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')

    
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5])
    plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5], linestyle='--')
    plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[1], linestyle='--')
    plt.legend(['Pretrained $AP_{50}$', 'Trained from scratch $AP_{50}$', 'Pretrained $AP^{kp}_{50}$', 'Trained from scratch $AP^{kp}_{50}$'], loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP$')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_pretrained.pdf')
    plt.show()
    
def draw_ap50_model_size():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/22_10_20-23_00[res34-lr0.001-iter3000-mile1000-no_warmup]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/24_10_20-3_00[res101-batch1-lr0.001-3000iter-sch1000-no_warmup]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[3])
    plt.plot(
        [x['iteration'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['bbox/AP50'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5])
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[1], linestyle='--')
    plt.plot(
        [x['iteration'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_2 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[3], linestyle='--')
    plt.plot(
        [x['iteration'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], 
        [x['keypoints/AP50'] for x in metrics_3 if ((x['iteration']%300==299) and (x['iteration']<=3000))], color=colors[5], linestyle='--')
    plt.legend(['res34 $AP_{50}$', 'res50 $AP_{50}$', 'res101 $AP_{50}$', 'res34 $AP1^{kp}_{50}$', 'res50 $AP1^{kp}_{50}$', 'res101 $AP1^{kp}_{50}$'], loc='center left')
    plt.xlabel('Iterations')
    plt.ylabel('Validation $AP$')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_model_size.pdf')
    plt.show()

def draw_train_loss_model_size():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/22_10_20-23_00[res34-lr0.001-iter3000-mile1000-no_warmup]' + '/metrics.json')
    metrics_2 = load_json_arr(experiment_folder + '/22_10_20-16_30[sch1000-lr0.001-no_warmup]' + '/metrics.json')
    metrics_3 = load_json_arr(experiment_folder + '/24_10_20-3_00[res101-batch1-lr0.001-3000iter-sch1000-no_warmup]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_1 if x['iteration']<=3000], color=colors[1])
    plt.plot(
        [x['iteration'] for x in metrics_2 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_2 if x['iteration']<=3000], color=colors[3])
    plt.plot(
        [x['iteration'] for x in metrics_3 if x['iteration']<=3000], 
        [x['total_loss'] for x in metrics_3 if x['iteration']<=3000], color=colors[5])
    plt.legend(['res34', 'res50', 'res101'], 
               loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total training loss')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_loss_model_size.pdf')
    plt.show()

def draw_long_train():
    experiment_folder = '/home/porthos/masters_thesis/cuboid_detection/results/final_dataset_results/final'
    metrics_1 = load_json_arr(experiment_folder + '/23_10_20-15_00[long_train-iter25000-lr0.001-mile1000-no_warmup-wd]' + '/metrics.json')

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1], 
        [x['total_loss'] for x in metrics_1], color=colors[0])
    # plt.legend(['res34', 'res50', 'res101'], 
               # loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Total training loss')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/train_loss_long_train.pdf')
    
    fig_2 = plt.figure()
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299))], 
        [x['bbox/AP50'] for x in metrics_1 if ((x['iteration']%300==299))], color=colors[0])
    plt.plot(
        [x['iteration'] for x in metrics_1 if ((x['iteration']%300==299) )], 
        [x['keypoints/AP50'] for x in metrics_1 if ((x['iteration']%300==299))], color=colors[0], linestyle='--')
    plt.legend(['Training $AP_{50}$', 'Training $AP1^{kp}_{50}$'], loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Training $AP$')
    fig_2.tight_layout()
    axes = plt.gca()
    axes.set_ylim([0,100])
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/ap50_long_train.pdf')  
    plt.show()
  
def draw_dataset():
    val_values_box = [68.03, 70.24, 74.75, 73.13, 76.2]
    val_values_key = [26.09, 27.9, 31.24, 33.84, 34.37]
    x_perc = [20, 40, 60, 80,    100]

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#8C564B', '#7F7F7F', '#9467BD', '#17BECF'] #blue, orange, green, red, brown, grey, purple, cian
    fig_1 = plt.figure()
    plt.plot(x_perc, val_values_box, color=colors[0])
    plt.plot(x_perc, val_values_key, color=colors[3])
    plt.legend(['$AP_{50}$', '$AP1^{kp}_{50}$'], loc='center right')
    plt.plot(np.unique(x_perc), np.poly1d(np.polyfit(x_perc, val_values_key , 1))(np.unique(x_perc)), color=colors[3], linestyle=':')
    plt.plot(np.unique(x_perc), np.poly1d(np.polyfit(x_perc, val_values_box , 1))(np.unique(x_perc)), color=colors[0], linestyle=':')
    # plt.plot(x_perc, val_values_key, color=colors[0], linestyle='--')
    plt.xticks(x_perc, labels=['20\%', '40\%', '60\%', '80\%', '100\%'])
    plt.xlabel('Dataset size')
    plt.ylabel('Validation $AP$')
    fig_1.tight_layout()
    # plt.title('Total train-val losses at 0.001 and 0.0001 lr')
    plt.grid(b=True, linestyle=':')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/results/dataet_size.pdf')
    
    
    
if __name__ == '__main__':
    draw_multi_lr()
    draw_multi_lr_ap50_bbox()
    draw_multi_lr_ap50_key()
    draw_train_val_warmup_lr_sch()
    draw_ap50_warmup_lr_sch()
    draw_train_val_best_lr()
    draw_train_loss_adam_sgd()
    draw_ap50_adam_sgd()
    # draw_train_val_lr_sch()
    # draw_ap50_lr_sch()
    draw_best_model()
    draw_ap50_pretrained()
    draw_ap50_model_size()
    draw_train_loss_model_size()
    draw_long_train()
    draw_dataset()