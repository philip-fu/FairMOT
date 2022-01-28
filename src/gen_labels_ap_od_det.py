import os
import glob
import os.path as osp
import numpy as np
import pandas as pd
import shutil
import tqdm
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

data_dir='/home/jupyter/dataset/ap_od/20210520_dynamic/'
label_root = osp.join(data_dir, 'labels_with_ids')
dataset_file = '/home/jupyter/FairMOT/src/data/ap_od_dynamic_all.train'



# clean existing
print('Removing exisiting files in {} and {}'.format(dataset_file, label_root))
if osp.exists(dataset_file):
    os.remove(dataset_file)
    
if osp.exists(label_root) and osp.isdir(label_root):
    shutil.rmtree(label_root)

mkdirs(label_root)

print('Scanning label files...')
label_files = glob.glob(os.path.join(data_dir, 'labels/*.txt'))
img_file_w_objs = []


def parse_annotation(label_file):
    img_width = 640.
    img_height = 640.
    
    
    data = pd.read_csv(label_file, delimiter=' ', header=None, usecols=[0,4,5,6,7], names=['label', 'xmin', 'ymin', 'xmax', 'ymax'])
    data = data[data['label'].isin(['item', 'holding_hand', 'empty_hand'])]
    if len(data) == 0:
        return None

    data['x'] = (data['xmin'] + data['xmax']) / 2. / img_width
    data['y'] = (data['ymin'] + data['ymax']) / 2. / img_height
    data['w'] = (data['xmax'] - data['xmin']) / img_width
    data['h'] = (data['ymax'] - data['ymin']) / img_height
    data['label'].replace({'item': 0, 'holding_hand': 1, 'empty_hand': 1}, inplace=True)
    data['t_id'] = '-1'
    
    label_file_new = label_file.replace('labels/', 'labels_with_ids/')
    data[['label', 't_id', 'x', 'y', 'w', 'h']].to_csv(label_file_new, header=None, index=None, sep=' ')
    
    return label_file.replace('labels/', 'images/').replace('.txt', '.png')
    


#for label_file in tqdm.tqdm(label_files):
#    img_file_w_obj = parse_annotation(label_file)
#    img_file_w_objs.append(img_file_w_obj)
    
p = Pool(multiprocessing.cpu_count() * 4 - 1)
img_file_w_objs = list(tqdm.tqdm(p.imap(parse_annotation, label_files), total=len(label_files)))
p.close()
p.join()

    
    

with open(dataset_file, 'a') as f:
    for img_file_w_obj in img_file_w_objs:
        if img_file_w_obj is not None:
            f.write(img_file_w_obj)
            f.write('\n')
            