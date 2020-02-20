from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os


class TLClassifier(object):
    def __init__(self):
        
        self.true_path = os.path.dirname(os.path.realpath('models/'))

        self.init_classifier()
        self.init_graph()
       
        self.match_dict = {0: TrafficLight.GREEN,
                           1: TrafficLight.RED,
                           2: TrafficLight.YELLOW,
                           3: TrafficLight.UNKNOWN}

    def get_classification(self, image):

        self.localize_obj(image)
        if self.img_out is None:
            #print('Didnt find traffic lights')
            return self.match_dict[3]
        self.classify_img()
        return self.match_dict[self.state]
        
    def localize_obj(self,img):
        
        # net was trained in bgr colorspace
        self.img_out = None
        self.img = img
        # shape of (1,?,?,3)
        input_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), axis=0)

        with self.dg.as_default():
            (detection_boxes, detection_scores, detection_classes,num_detections) = self.sess.run(
                [self.box_t, self.score_t, self.class_t, self.num_t],
                feed_dict={self.img_t: input_img})
            
        for obs in zip(detection_boxes[0], detection_classes[0], detection_scores[0]):
            # did we observe traffic lights with high certainty?
            if obs[1] == 10 and obs[2] >= .5:
                # get box and img for classification
                box = obs[0]
                x_min = int(box[0] * self.img.shape[0])
                x_max = int(box[2] * self.img.shape[0])
                y_min = int(box[1] * self.img.shape[1])
                y_max = int(box[3] * self.img.shape[1])
                self.img_out = cv2.resize(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)[x_min:x_max,y_min:y_max,:],(14,32))
                break

    def classify_img(self):
        with self.class_graph.as_default():
            self.state = np.argmax(self.classifier.predict(self.img_out.reshape(1,32,14,3)))

    def init_graph(self):
        self.path = self.true_path + "/light_classification/models/frozen_inference_graph.pb"
        self.dg = tf.Graph()
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(self.path, 'rb') as f:
                gdef.ParseFromString(f.read())
                tf.import_graph_def(gdef, name="")
            self.sess = tf.Session(graph=self.dg)
            self.img_t = self.dg.get_tensor_by_name('image_tensor:0')
            self.box_t = self.dg.get_tensor_by_name('detection_boxes:0')
            self.score_t = self.dg.get_tensor_by_name('detection_scores:0')
            self.class_t = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_t = self.dg.get_tensor_by_name('num_detections:0')
    
    def init_classifier(self):
        self.classifier = load_model(self.true_path + '/light_classification/models/model.h5')
        self.class_graph = tf.get_default_graph()
