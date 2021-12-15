import tensorflow as tf
import numpy as np
import cv2

from lib.utils.post_process import ctdet_post_process


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh



def _tranpose_and_gather_feat(feat, ind):
    feat = np.transpose(feat, (0, 2, 3, 1))
    feat = np.reshape(feat, (feat.shape[0], -1, feat.shape[3]))
    feat = feat[:, ind, :]
    feat = np.squeeze(feat, 0)
    return feat

def _postprocess_dets(dets, inp_size=(608, 1088), img_size=(1080, 1920), num_classes=1, down_ratio=4):
    inp_height, inp_width = inp_size
    height, width = img_size
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}

    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]

def _merge_dets(detections, num_classes=1, max_per_image=500):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

class JdeDla34ConvTF(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_saved_model(model_filepath = self.model_filepath)
        #self.load_graph(model_filepath = self.model_filepath)

        self.height = 608
        self.width = 1088
        self.max_per_image = 500
        self.num_classes = 1
        self.down_ratio = 4

    def load_saved_model(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(allow_soft_placement=True))

        tf.compat.v1.saved_model.load(self.sess, [tf.saved_model.SERVING], model_filepath, import_scope='')

        #all_tensors = [tensor for op in graph.get_operations() for tensor in op.values()]
        #for t in all_tensors:
        #    print(t)
        
        self.input_tensor = graph.get_tensor_by_name("serving_default_input:0") #Tensor("serving_default_input:0", shape=(1, 3, 608, 1088), dtype=float32)
        output_hm_tensor = graph.get_tensor_by_name("PartitionedCall:1") #Tensor("PartitionedCall:1", shape=(1, 1, 152, 272), dtype=float32)
        output_reg_tensor = graph.get_tensor_by_name("PartitionedCall:4") #Tensor("PartitionedCall:4", shape=(1, 2, 152, 272), dtype=float32)
        output_id_tensor = graph.get_tensor_by_name("PartitionedCall:2") #Tensor("PartitionedCall:2", shape=(1, 128, 152, 272), dtype=float32)
        output_wh_tensor = graph.get_tensor_by_name("PartitionedCall:5") #Tensor("PartitionedCall:5", shape=(1, 4, 152, 272), dtype=float32)
        output_dets = graph.get_tensor_by_name("PartitionedCall:0") #Tensor("PartitionedCall:0", shape=(1, 500, 6), dtype=float32)
        output_inds = graph.get_tensor_by_name("PartitionedCall:3") #Tensor("PartitionedCall:3", shape=(1, 500), dtype=int64)
        
        self.output_tensors = [output_hm_tensor, output_reg_tensor, output_id_tensor, output_wh_tensor, output_dets, output_inds]

    def _preprocessing(self, img):

        # Padded resize
        img, _, _, _ = letterbox(img, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img

    

    def _postprocessing(self, output, img_ori):
        img_height, img_width = img_ori.shape[0:2]
        hm, reg, id_feature, wh, dets, inds = output
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        dets = _postprocess_dets(dets, 
                                 inp_size=(self.height, self.width), 
                                 img_size=(img_height, img_width), 
                                 num_classes=self.num_classes, 
                                 down_ratio=self.down_ratio)
        dets = _merge_dets([dets])[1]

        """
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        #nms
        dets = torch.from_numpy(dets).float()
        remain_inds = ops.nms(dets[:, 0:4], dets[:, 4], iou_threshold=self.opt.nms_thres).cpu().detach().numpy()
        dets = dets[remain_inds, :].cpu().detach().numpy()
        id_feature = id_feature[remain_inds, :]
        """
        


    def infer(self, img):
        # Know your output node name
        img_ori = img.copy()
        img = self._preprocessing(img)
        output = self.sess.run(self.output_tensors, feed_dict = {self.input_tensor: [img]})

        self._postprocessing(output, img_ori)

        return output


if __name__ == "__main__":
    model_filepath = 'models/dla34conv_ap_all_ds_20'
    model = JdeDla34ConvTF(model_filepath)

    img = cv2.imread('images/test_ap/000040.jpg')
    pred = model.infer(img)