from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from lib.tracker.multitracker import JDETracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
from lib.tracking_utils.eval import trackeval
import lib.datasets.dataset.jde as datasets

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if True: #tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, scores=online_scores, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=True, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.avi'.format(seq))
            cmd_str = 'ffmpeg -y -f image2 -i "{}/%05d.jpg" -c:v copy "{}"'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

    eval_config['LOG_ON_ERROR'] = osp.join(result_root, 'error.log')
    eval_config['PLOT_CURVES'] = False
    dataset_config['GT_FOLDER'] = data_root
    dataset_config['SEQMAP_FOLDER'] = osp.join(data_root, '../../seqmaps')
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['TRACKERS_FOLDER'] = osp.join(result_root, '..')
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['BENCHMARK'] = 'ap'

    dataset_config['EVAL_PROD'] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)

    """
    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
    """


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if opt.test_ap:
        seqs_str = '''201907251018_darshan_lane48
                      201907251032_darshan_lane48
                      201907251047_darshan_lane47
                      201907251053_darshan_lane47
                      201907251056_darshan_lane47
                      201907251100_darshan_lane47
                      201907251103_darshan_lane47
                      201907251115_darshan_lane46
                      201907251118_darshan_lane46
                      201907251121_darshan_lane46
                      '''

        #seqs_str = '''201907251018_darshan_lane48'''
        """
        seqs_str = '''fp_20210908_967_27_1631111760000-1631111850000
                      fp_20210908_967_28_1631119740000-1631119760000
                      fp_20210908_967_30_1631105760000-1631105820000
                      fp_20210908_967_33_1631111940000-1631112060000
                      fp_20210908_967_46_1631105880000-1631105940000
                      fp_20210908_967_46_1631117820000-1631117880000
                      fp_20210908_967_47_1631105580000-1631105660000
                      fp_20210908_1000_44_1631072950000-1631072960000
                      fp_20210908_2656_45_1631052655000-1631052690000
                      fp_20210908_2656_48_1631054700000-1631054760000
                      fp_20210908_2656_50_1631053500000-1631053540000
                      fp_20210908_3648_33_1631118210000-1631118260000
                      fp_20210908_3648_37_1631114670000-1631114730000
                      fp_20210908_3648_39_1631117100000-1631117160000
                      fp_20210908_3648_43_1631114010000-1631114060000
                      fp_20210908_3648_46_1631123790000-1631123835000
                      fp_20210908_3648_47_1631116350000-1631116440000
                      fp_20210908_3648_50_1631113980000-1631114055000
                      fp_20210908_3763_35_1631128500000-1631128560000
                      fp_20210908_3763_37_1631118700000-1631118730000
                      fp_20210908_3763_45_1631121420000-1631121480000
                      fp_20210908_3763_45_1631127180000-1631127240000
                      fp_20210908_3763_51_1631118420000-1631118450000
                      fp_20210908_6243_1_1631125530000-1631125560000
                   '''
        """
        data_root = os.path.join(opt.data_dir, 'ap/images/train')
    else:
        print('Only test data for AP is available. Exiting.')
        exit(1)
    seqs = [seq.strip() for seq in seqs_str.split()]

    '''
    import wandb
    wandb.init(project="ap-tracking-fairmot", entity="philip-fu")
    wandb.config = {
        "arch": opt.arch,
        "model": opt.load_model,
        "match_thres": opt.match_thres,
        "appearance_weight": opt.appearance_weight,
        "conf_thres": opt.conf_thres,
        "high_conf_thres": opt.high_conf_thres,
        "byte_track": opt.byte_track,
        "handle_occlusion": opt.handle_occlusion
    }
    '''

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='fairmot-yolov5',
         show_image=False,
         save_images=False,
         save_videos=True)
