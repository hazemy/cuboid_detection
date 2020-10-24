#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:18:16 2020

@author: porthos
"""

from annot_processor import merge_annot_files, get_unique, amend_dir, remove_faulty
from format_converter import convert2detectron_format
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def get_total (annot_file):
    total_ins = 0
    total_pos = 0
    total_keypts_vis = 0
    total_keypts_hid = 0
    total_keypts_out = 0
    instances_num = []
    annot_time = []
    for entry in annot_file:
        instances = entry['cubes']
        total_ins += len(instances)
        if instances:
            total_pos += 1
            instances_num.append(len(instances))
            annot_time.append(entry['time'])
            for cube in instances:
                for keypt in cube:
                    if keypt[2] == 2:
                        total_keypts_vis += 1
                    if keypt[2] == 1:
                        total_keypts_hid += 1
                    if keypt[2] == 0:
                        total_keypts_out += 1   
    return total_pos, total_ins, instances_num, total_keypts_vis, total_keypts_hid, total_keypts_out, annot_time
  
    
def vis_get_total(total_pos, total_ins, instances_num, total_keypts_vis, total_keypts_hid, total_keypts_out):
    print('Total num of cuboid instances is: {}'.format(total_ins))
    print('Total num of pos images is: {}'.format(total_pos))
    print('Total num of visible keypts is: {}'.format(total_keypts_vis))
    print('Total num of hidden keypts is: {}'.format(total_keypts_hid))
    print('Total num of out keypts is: {}'.format(total_keypts_out))
    print('Max num of instances per image is: {}'.format(max(instances_num)))
    instances_num_np = np.asarray(instances_num)
    fig_1 = plt.figure()
    n, bins, _ =  plt.hist(instances_num_np, bins=range(1,11,1), log=True, align='left', histtype='bar', ec='black', color='#1F77B4')
    # plt.boxplot(instances_num_np)
    plt.xlabel('Number of instances')
    plt.ylabel('Number of images')
    plt.title('Instances per image') 
    fig_1.tight_layout()
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/dataset/hist_inst_per_img.pdf')
    plt.show()

    # fig_2 = plt.figure()
    labels = ['Visible (6527)', 'Hidden (1909)', 'Out (216)']
    vals = [6527, 1909, 216]
    # pct = ['75.3', '22.3', '2.4']
    data = [total_keypts_vis, total_keypts_hid, total_keypts_out]
    explode = (0.0, 0.0, 0.0)  # only "explode" the 2nd slice
    fig2, ax1 = plt.subplots(subplot_kw=dict(aspect="equal"))
    colors = ['#2CA02C', '#1F77B4', '#D62728']
    wedges, texts, autotexts = ax1.pie(data, explode=explode, labels=labels, 
                                       autopct=make_autopct(vals), 
                                       textprops=dict(color="w"), shadow=True, startangle=90, 
                                       colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle. 
    ax1.set_title('Visibility of Dataset Keypoints')
    ax1.legend(wedges, labels,
          title="Visibility",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=11, weight="bold")
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/dataset/vis_pie.pdf')
    plt.show()


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f} \% '.format(p=pct)
    return my_autopct


def get_mean_pixel(dataset_list):
    sum_r = 0
    sum_g = 0
    sum_b = 0
    total_pixels = 0
    for entry in dataset_list:
        img_path = entry['file_name']
        img = cv2.imread(img_path) # in RGB format
        # height, width = img.shape[:2]
        val_r = np.reshape(img[:, :, 0], -1) # convert red channel to 1D array
        val_g = np.reshape(img[:, :, 1], -1) # convert green channel to 1D array
        val_b = np.reshape(img[:, :, 2], -1) # convert blue channel to 1D array
        sum_r += np.sum(val_r)
        sum_g += np.sum(val_g)
        sum_b += np.sum(val_b)
        total_pixels += len(val_r)
    print('Mean pixel value for Red ch is: {}'.format(sum_r/total_pixels))
    print('Mean pixel value for Green ch is: {}'.format(sum_g/total_pixels))
    print('Mean pixel value for Blue ch is: {}'.format(sum_b/total_pixels))


def get_areas(dataset_list):   
    heights = []
    widths = []
    areas = []
    heights_pos = []
    widths_pos = []
    areas_imgs_pos = []
    ratio_img_size = []
    areas_boxes = []
    total_ins = 0
    for entry in dataset_list:
        heights.append(entry['height'])
        widths.append(entry['width'])
        areas.append(heights[-1]*widths[-1])
        if entry['annotations']:
            img_height = entry['height']
            img_width = entry['width']
            heights_pos.append(img_height)
            widths_pos.append(img_width)
            img_area = img_height*img_width
            areas_imgs_pos.append(img_area)
            for annot in entry['annotations']:
                box_area = annot['bbox'][2]*annot['bbox'][3]
                areas_boxes.append(box_area)
                ratio = box_area/img_area
                ratio_img_size.append(ratio)
                total_ins += 1
                # if ratio < 0.01:
                #     print(entry['file_name']) 
    return heights_pos, widths_pos, heights, widths, ratio_img_size, areas_imgs_pos, areas, total_ins

    
def vis_get_areas(heights_pos, widths_pos, heights, widths, ratio_img_size, areas_imgs_pos, areas, total_ins):
    print('Min image height is: {}'.format(min(heights_pos)))
    print('Min image width is: {}'.format(min(widths_pos)))
    # print('Min area for image: {}'.format((dataset_list[areas_imgs_pos.index(min(areas_imgs_pos))])['file_name']))
    print('Max image height is: {}'.format(max(heights_pos)))
    print('Max image width is: {}'.format(max(widths_pos)))
    # print('Max area for image: {}'.format((dataset_list[areas_imgs_pos.index(max(areas_imgs_pos))])['file_name']))
    print('Mean image height is: {}'.format(np.mean(np.asarray(heights))))
    print('Mean image width is: {}'.format(np.mean(np.asarray(widths))))
    ratio_img_size_np = np.asarray(ratio_img_size)
    percent_img_size = np.multiply(ratio_img_size, 100)
    fig_1 = plt.figure()
    n, bins, _ =  plt.hist(np.ceil(percent_img_size), bins=range(1,101,1), align='left', 
                           histtype='bar', ec='black')
    plt.xlabel('Percent of image size')
    plt.ylabel('Number of instances')
    plt.title('Histogram of instance size')
    fig_1.tight_layout()
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/dataset/hist_img_size.pdf')
    plt.show()

    # percent_range = np.asarray(range(0,100,10))
    # left_of_first_bin = percent_range.min() - 5
    # right_of_last_bin = percent_range.max() + 5
    # n, bins, _ = plt.hist(percent_img_size, np.arange(left_of_first_bin, right_of_last_bin, 10))
    fig_2 = plt.figure()
    n_np = np.asarray(n)
    percent_instances = np.multiply(np.divide(n_np, total_ins), 100)
    plt.plot(bins[1:], percent_instances, marker='.', markersize=7, color='#1F77B4')
    plt.xlabel('Percent of image size')
    plt.ylabel('Percent of instances')
    plt.title('Instance size')
    fig_2.tight_layout()
    
    fig_3, ax_3 = plt.subplots()
    n, bins, _ = plt.hist(areas, bins=10, log=True, align='left', histtype='bar', ec='black')
    # print(bins)
    ax_3.set_xlabel('Image area (pixels)')
    ax_3.set_ylabel('Number of images')
    ax_3.set_title('Area of images')
    # ax_3.set_xticks(np.arange((500000), (14000000), (1000000)))
    # ax_3.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
    fig_3.tight_layout()
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/dataset/hist_img_area.pdf')
    plt.show()
    # bins=[500000, 1500000, 2500000, 3500000, 4500000, 5500000, 6500000, 7500000, 8500000, 9500000, 10500000]


def get_annotator_stats(annot_files_dir_list):
    annotator_total_pos = []
    annotator_total_inst = []
    annotators_time = []
    annotators_time_combined = []
    for annot_file in annot_files_dir_list:
        with open(annot_file) as f:
            annot_list = json.load(f)
        total_pos, total_ins, instances_num, _, _, _, annot_time = get_total(annot_list)
        annotator_total_pos.append(total_pos)
        annotator_total_inst.append(total_ins)
        annotators_time.append(annot_time)
        annotators_time_combined.extend(annot_time)
    return annotator_total_pos, annotator_total_inst, annotators_time, annotators_time_combined


def vis_get_annotator_stats(annotator_total_pos, annotator_total_inst, annotators_time, annotators_time_combined):
    print(annotator_total_inst)
    print(annotator_total_pos)
    labels = ['Annot. 1', 'Annot. 2', 'Annot. 3', 'Annot. 4', 'Annot. 5']
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    width = 0.35
    rects1 = ax.bar(x - width/2, annotator_total_pos, width, label='Positive images', color='#1F77B4')
    rects2 = ax.bar(x + width/2, annotator_total_inst, width, label='Instances', color='#FF7F0E')
    ax.set_xlabel('Annotator number')
    ax.set_ylabel('Total labeled')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Annotators workload') 
    ax.set_ylim([0,580])
    ax.legend()  
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/dataset/annot_workload.pdf')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.boxplot(annotators_time, 0, '')
    ax2.set_xticklabels(labels)
    # fig2.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
    #                 hspace=0.4, wspace=0.3)
    ax2.set_title('Annotation time') 
    ax2.set_xlabel('Annotator number')
    ax2.set_ylabel('Time (sec)')
    # plt.savefig('/home/porthos/masters_thesis/writing/figures/dataset/annot_time.pdf')
    plt.show()
    
    fig3, ax3 = plt.subplots()
    plot_out = ax3.boxplot(annotators_time_combined, 0)
    ax3.set_xticklabels(labels)
    total_time = 0
    output = [item.get_ydata() for item in plot_out['fliers']]
    outliers = np.asarray(output)[0]
    for time in annotators_time_combined:
        # if time not in outliers:
        total_time += time
    mean_img_time_all = (total_time)/(len(annotators_time_combined)-len(outliers))
    print('Total time for all annotators is: {}'.format(total_time))
    print('Mean image time for all annotators is: {}'.format(mean_img_time_all))
    

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')    


def get_entry(img_path, dataset_list):
    for entry in dataset_list:
        if entry['file_name'] == img_path:
            return entry




if __name__ =='__main__':
    annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_hazem_mod.json'
    annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_ammar.json'
    annot_file_dir_3 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_pablo.json'
    annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_anas.json'
    annot_file_dir_5 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_leonie.json'
    images_dir = '/home/porthos/masters_thesis/datasets/final_dataset/images'
    annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4, annot_file_dir_5]
    annot_files_merged = merge_annot_files(annot_files_dir_list) 
    unique, duplicates_ids, duplicates_index = get_unique(annot_files_merged)
    annot_file_amended = amend_dir(unique)
    annot_file_corrected = remove_faulty(annot_file_amended)
    
    total_pos, total_ins, instances_num, total_keypts_vis, total_keypts_hid, total_keypts_out, _ = get_total(annot_file_corrected)
    vis_get_total(total_pos, total_ins, instances_num, total_keypts_vis, total_keypts_hid, total_keypts_out)
    
    dataset_list = convert2detectron_format(annot_file_corrected, images_dir)
    get_mean_pixel(dataset_list)
    
    heights_pos, widths_pos, heights, widths, ratio_img_size, areas_imgs_pos, areas, total_ins = get_areas(dataset_list)
    vis_get_areas(heights_pos, widths_pos, heights, widths, ratio_img_size, areas_imgs_pos, areas, total_ins)
    
    annotator_total_pos, annotator_total_inst, annotators_time, annotators_time_combined = get_annotator_stats(annot_files_dir_list)
    vis_get_annotator_stats(annotator_total_pos, annotator_total_inst, annotators_time, annotators_time_combined)
    
    img_query = '/home/porthos/masters_thesis/datasets/final_dataset/images/1791076.jpg'
    img_data = get_entry(img_query, dataset_list)

    