import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd() 
data_path = join(cwd,'')
savedir = './'
dataset_list = ['base', 'val', 'novel']#

#if not os.path.exists(savedir):
#    os.makedirs(savedir)
print(data_path)
cl = -1
folderlist = []

datasetmap = {'base':'train','val':'val','novel':'test'}

for dataset in dataset_list:
    filelists = {}
    density_lists = {}
    folderlist = []
    print(data_path + datasetmap[dataset])
  #  with open(data_path + datasetmap[dataset] + ".csv", "r") as lines:
    class_folders = os.listdir(data_path+'/images/' + datasetmap[dataset])
    for i, c in enumerate(class_folders):

        class_name = c
        # if not class_name in filelists[dataset]:
        folderlist.append(class_name)
        filelists[class_name] = []
        fnames = listdir( join(data_path, 'images', datasetmap[dataset], class_name) )
        fnames = list(join( data_path, 'images', datasetmap[dataset], class_name, fname) for fname in fnames)     
        filelists[class_name] = fnames

        gt_densities = listdir( join(data_path, 'gt_density', datasetmap[dataset], class_name) )
        gt_densities = list(join( data_path, 'gt_density', datasetmap[dataset],  class_name, gt_density) for gt_density in gt_densities)     
        density_lists[class_name] = gt_densities

    annotations = listdir( join(data_path, 'annotations', datasetmap[dataset]) )    
    annotations = list(join( data_path, 'annotations', datasetmap[dataset], annotation) for annotation in annotations)


    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"class_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in list(filelists.values())])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"annotation_names": [')
    fo.writelines(['"%s",' % item.replace("\\","\\\\")  for item in annotations])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"density_names": [')
    fo.writelines(['"%s",' %  item  for item in list(density_lists.values())])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
