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
        self.class_graph = tf.get_default_graph()
        self.init_graph()
       
        self.state = 3
        self.last_state = None
        self.img_c = 0
        self.match_dict = {0: TrafficLight.GREEN,
                           1: TrafficLight.RED,
                           2: TrafficLight.YELLOW,
                           3: TrafficLight.UNKNOWN}

        self.skip_classification = True


    def get_classification(self, image):
        self.last_state = self.state
        self.state = 3
        self.localize_obj(image)
        #return TrafficLight.UNKNOWN
        if self.skip_classification:
            self.classify_img()
        #print(self.state)
        #cv2.imwrite(self.true_path + '/light_classification/imgs/' + '{}.png'.format(self.img_c), image)
        self.img_c += 1
        return self.match_dict[self.state]

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
        
    def localize_obj(self,img):
        
        self.img_out_c = None
        self.skip_classification = True
        # net was trained in bgr colorspace
        self.img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # shape of (1,?,?,3)
        input_img = np.expand_dims(img, axis=0)

        with self.dg.as_default():
            (detection_boxes, detection_scores, detection_classes,num_detections) = self.sess.run(
                [self.box_t, self.score_t, self.class_t, self.num_t],
                feed_dict={self.img_t: input_img})
            
        self.boxes = detection_boxes
        self.scores = detection_scores
        self.classes = detection_classes
        self.obs = num_detections
        self.update_img()
    
    def update_img(self):
        for obs in zip(self.boxes[0], self.classes[0], self.scores[0]):
            # did we observe traffic lights with high certainty?
            if obs[1] == 10 and obs[2] >= .5:
                # get box and img
                box = obs[0]
                x_min = int(box[0] * self.img.shape[0])
                x_max = int(box[2] * self.img.shape[0])
                y_min = int(box[1] * self.img.shape[1])
                y_max = int(box[3] * self.img.shape[1])
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                
                '''
                to check if boxes are correct
                print('Box')
                print(x_min, x_max, y_min, y_max)
                '''
                
                
                if not self.img_out_c:
                    self.img_out = cv2.resize(self.img[x_min:x_max,y_min:y_max,:],(14,32))
                    cv2.imwrite(self.true_path + '/light_classification/imgs/' + '{}_2.png'.format(self.img_c), self.img_out)
                    self.img_out_c = True
                    break
        if not self.img_out_c:
            self.state = 3
            self.skip_classification = False


    def classify_img(self):
        
        with self.class_graph.as_default():
            self.state = np.argmax(self.classifier.predict(self.img_out.reshape(1,32,14,3)))
        print('Classification: ', self.state)