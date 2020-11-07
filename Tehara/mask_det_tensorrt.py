import logging
import time

import cv2 as cv
import numpy as np
from tensorrt_uff import WeightType
from tensorrt_uff import Tensorrt

class MaskDet_TensorRT:
    """
    This class will load the mask model and predict for given images
    """
    classes = ['yes', 'no']
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]

    DEFAULT_CONFIGS_TENSORRT = {
        'tensorrt':
            {
                'workspace_size': 10000000,
                'gpu_id': 0,
                'fp16_mode': False,
                'weight_type': WeightType.UFF
            },
        'batch_size': 1,
        'inputs': ['image_tensor_x0'],
        'outputs': ['model_1/dense_2/Softmax'],
        'input_shape': (96, 96, 3)}

    def __init__(self, weights_file, configs=DEFAULT_CONFIGS_TENSORRT):
        '''
        #Limit the GPU memory of this model using tensorflow backend
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
        set_session(tf.Session(config=config))
        '''
        self.weights_file = weights_file
        self.configs = configs
        self.input_shape = configs['input_shape']
        self.model = None

    def init(self):
        self.load_model(self.weights_file)

    def load_model(self, weights_file):
        self.model = zml_tensorrt(weights_file, self.input_shape[2], self.input_shape[0],
                                  self.input_shape[1], self.configs['inputs'],
                                  self.configs['outputs'], self.configs['tensorrt']['workspace_size'],
                                  self.configs['tensorrt']['gpu_id'], self.configs['batch_size'],
                                  self.configs['tensorrt']['fp16_mode'], self.configs['tensorrt']['weight_type'])
        self.batch_size = self.configs['batch_size']
        self.measure_model_delay()

    def pre_process(self, x):
        img_rz = cv.resize(x, (self.input_shape[1], self.input_shape[0])).astype('float32')
        img_rz /= 255.0

        for i in range(3):
            img_rz[:, :, i] -= MaskDet_TensorRT.IMAGE_NET_MEAN[i]
            img_rz[:, :, i] /= MaskDet_TensorRT.IMAGE_NET_STD[i]

        img_rz = np.expand_dims(img_rz, axis=0)
        img_rz = np.transpose(img_rz, (0, 3, 1, 2))
        img_rz = img_rz.ravel()

        return img_rz

    def batch_pre_process(self, x_list):
        self.no_imgs = len(x_list)
        img_rz_batch = np.zeros((self.batch_size, self.input_shape[1] * self.input_shape[0] * self.input_shape[2]),
                                dtype=np.float32)
        for j, x in enumerate(x_list):
            img_rz = cv.resize(x, (self.input_shape[1], self.input_shape[0])).astype('float32')
            img_rz /= 255.0

            for i in range(3):
                img_rz[:, :, i] -= MaskDet_TensorRT.IMAGE_NET_MEAN[i]
                img_rz[:, :, i] /= MaskDet_TensorRT.IMAGE_NET_STD[i]

            img_rz = np.transpose(img_rz, (2, 0, 1))
            img_rz = img_rz.ravel()
            img_rz_batch[j] = img_rz

        return img_rz_batch

    def get_class_confidence(self, prediction):
        """
        Take the prediction and decide the which class it belonfs ro
        :param prediction: Original prediction from the model
        :return: Class and the score
        """
        max_prob = int(np.argmax(prediction, axis=0))
        pred_sorted = np.sort(prediction)
        score = pred_sorted[1] / pred_sorted[0]
        return MaskDet_TensorRT.classes[max_prob], score

    def get_batch_class_confidence(self, batch_prediction, batch_size):
        batch_prediction = batch_prediction[0].reshape(batch_size, len(MaskDet_TensorRT.classes))
        batch_prediction = batch_prediction[:self.no_imgs]
        max_probs = np.argmax(batch_prediction, axis=1)
        cls_list = [MaskDet_TensorRT.classes[max_prob] for max_prob in max_probs]
        batch_pred_sorted = np.sort(batch_prediction)
        scores_list = batch_pred_sorted[:,-1]/(batch_pred_sorted[:,-2]+1e-10)

        return cls_list, scores_list

    def predict(self, img):
        """
        Predict an image
        :param img: Image
        :return: Prediction
        """
        start_time = time.time()
        img_rz = self.pre_process(img)
        pred = np.squeeze(self.net_forward(img_rz))
        cls, conf = self.get_class_confidence(pred)
        logging.info('Mask prediction {}[{}], time {} ms'.format(cls, conf,
                                                                 1000 * (time.time() - start_time)))
        return cls, conf
        
    def batch_predict(self, x_list):
        start_time = time.time()
        img_rz_batch = self.batch_pre_process(x_list)
        batch_pred = self.model.predict(img_rz_batch.ravel(), batch_size=self.batch_size)
        cls_list, scores_list = self.get_batch_class_confidence(batch_pred, batch_size=self.batch_size)

        logging.info('Mask batch prediction {}[{}], time {} ms'.format(cls_list, scores_list,
                                                                      1000 * (time.time() - start_time)))
        return cls_list, scores_list


if __name__ == '__main__':
    test_img = './lady.jpg'
    BATCH_SIZE = 5
    img = cv.imread(test_img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_list = [img] * BATCH_SIZE
    model_path = './trt_mask_v2.uff'

    tensorrt_model1 = MaskDet_TensorRT(model_path)
    tensorrt_model1.init()
    pred = tensorrt_model1.predict(img)

    configs = {
        'tensorrt':
            {
                'workspace_size': 10000000,
                'gpu_id': 0,
                'fp16_mode': False,
                'weight_type': WeightType.UFF
            },
        'batch_size': BATCH_SIZE,
        'inputs': ['image_tensor_x0'],
        'outputs': ['model_1/dense_2/Softmax'],
        'input_shape': (96, 96, 3)}

    tensorrt_model2 = MaskDet_TensorRT(model_path, configs)
    tensorrt_model2.init()
    batch_pred = tensorrt_model2.batch_predict(img_list)

    print('pred: {} \nbatch_pred: {}'.format(pred, batch_pred))
