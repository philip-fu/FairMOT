import sys
import os.path as osp

from lib.tracking_utils.eval import trackeval

data_root = 'C:/Users/Philip Fu/datasets/ap/images/train'

if __name__ == "__main__":
    exp_name = sys.argv[1]

    result_root = osp.join(data_root, '..', 'results')


    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

    eval_config['LOG_ON_ERROR'] = osp.join(result_root, 'error.log')
    eval_config['PLOT_CURVES'] = False
    dataset_config['GT_FOLDER'] = data_root
    dataset_config['SEQMAP_FOLDER'] = osp.join(data_root, '../../seqmaps')
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['TRACKERS_FOLDER'] = osp.join(result_root)
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['BENCHMARK'] = 'ap'
    dataset_config['TRACKERS_TO_EVAL'] = [exp_name] if len(exp_name) > 0 else None 

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