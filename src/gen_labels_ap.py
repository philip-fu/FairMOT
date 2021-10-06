import os.path as osp
import os
import numpy as np
import shutil


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/home/jupyter/dataset/ap/images/train'
label_root = '/home/jupyter/dataset/ap/labels_with_ids/train'
dataset_file = '/home/jupyter/FairMOT/src/data/ap.train'

# clean existing
if osp.exists(dataset_file):
    os.remove(dataset_file)
    
if osp.exists(label_root) and osp.isdir(label_root):
    shutil.rmtree(label_root)

mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root) if not s.startswith('.')]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    frames_w_objs = []
    for fid, tid, x, y, w, h, mark, _, _, _ in gt:
        if mark == 0:
            continue
            
        fid = int(fid)
        frames_w_objs.append(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
            
    frames_w_objs = list(set(frames_w_objs))
    with open(dataset_file, 'a') as f:
        for frame_w_objs in frames_w_objs:
            image_filename = osp.join(seq_root, seq, 'img1', str(frame_w_objs).zfill(6) + '.jpg')
            f.write(image_filename)
            f.write('\n')
            
    
