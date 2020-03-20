# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

result_path = 'mAP-master/input/images-optional/wider_face_val/'
img_root_path = 'predict_pic/wider_face_val/'
predict_result = 'mAP-master/input/detection-results/wider_face_val/'

def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)
class YOLO(object):
    _defaults = {
        "model_path": 'logs/wider_best/yolov3_best_val_result_2.h5',
        "anchors_path": 'model_data/anchors/yolo_anchors_WIDER_train.txt',
        "classes_path": 'datasets/wider_dataset/wider_classes.txt',
        "score" : 0.2,
        "iou" : 0.5,
        "model_image_size" : (608, 608),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path, num_anchors, num_classes))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
		
        pic_filename=os.path.basename(img_path)
        portion=os.path.splitext(pic_filename)
        if portion[1]=='.jpg':
            txt_result=predict_result+portion[0]+'.txt'
        print('txt_result The path is：'+txt_result)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            print("The updated coordinate label is ：", (left, top), (right, bottom))
            
            with open(txt_result, 'a')as new_f:
                new_f.write(str(label) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            #draw.rectangle(
                #[tuple(text_origin), tuple(text_origin + label_size)],
                #fill=self.colors[c])
            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image
    def detect_image_frame(self, image):
        start = timer()
        face_number = 0
        confidence_score = 0

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            face_number = face_number + 1
            confidence_score = confidence_score + score
			
            draw = ImageDraw.Draw(image)
            label = '{} {:.2f}'.format(predicted_class, score)
            
            label_size = draw.textsize(label, font)


            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        process_time = end - start
        
        if confidence_score > 0:
             avg_conf_score = confidence_score / face_number
        else:
            avg_conf_score = 0
        print(process_time)
        print("Avg. Conf. Score : "+str(avg_conf_score))

        return image, face_number, avg_conf_score

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path=0, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0)
    vid.set(3,1280)

    vid.set(4,720)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", str(output_path), str(video_FourCC), str(video_fps), str(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    max_face_number=13
    def_face_number = []
    det_face_number=0
	
    process_time_val = []
    face_detected = []
    average_conf_score = []
    prev_time = timer()
	
    total_frame = 0
    total_process_time = 0
    total_avg_conf_score = 0
    fps_val = []
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        image = Image.fromarray(frame)
        start = timer()
        image, face_number, avg_conf_score = yolo.detect_image_frame(image)
        end = timer()
		
        face_detected.append(face_number)
        def_face_number.append(max_face_number)
        average_conf_score.append(avg_conf_score)
        total_avg_conf_score = total_avg_conf_score + avg_conf_score
		
        process_time = end - start
        if total_frame > 1:
            process_time_val.append(process_time)
        total_process_time = process_time + total_process_time
        
        total_frame = total_frame + 1

        if total_process_time > 1800: #stop after 30 minutes
            break
		
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            tmp_avg_conf_score = total_avg_conf_score / total_frame
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps) + "  Avg. Conf. Score: "+ str(round(tmp_avg_conf_score, 2))+"  Detected Faces: "+str(face_number)
            fps_val.append(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    avg_process_time = total_process_time / total_frame
    avg_conf_score = total_avg_conf_score / total_frame
			
    print("Latest FPS : "+str(curr_fps))
    print("Total Process Time : "+str(total_process_time))
    print("Total Frame : "+str(total_frame))
    print("Average Process Time / Frame : "+str(avg_process_time))
    print("Average Conf. Score / Frame : "+str(avg_conf_score))
    yolo.close_session()
	
    plt.plot(process_time_val)
    plt.title('Process Time Per Frame')
    plt.xlabel('Frame')
    plt.legend(['Process Time / Frame'], loc='upper left')
    plt.show()
	
    plt.plot(fps_val)
    plt.title('Frame per Second Performance')
    plt.xlabel('Seconds')
    plt.legend(['FPS'], loc='upper left')
    plt.show()
	
    plt.plot(average_conf_score)
    plt.title('Average Conf. Score')
    plt.xlabel('Frame')
    plt.legend(['Avg. Conf. Score'], loc='upper left')
    plt.show()
	
    plt.plot(face_detected)
    plt.plot(def_face_number)
    plt.title('Detected Faces')
    plt.xlabel('Frame')
    plt.ylabel('Face Number')
    plt.legend(['Detected Faces','Maximum Real Faces'], loc='upper left')
    plt.show()
	
def detect_img_for_test(yolo):
    # Traverse each picture under the path
    global img_path
    for img_path in iterbrowse(img_root_path):

        print('img_path The path is：'+img_path)
        image = Image.open(img_path)
        image = image.convert("RGB")
        filename=os.path.basename(img_path)
        print('filename'+filename)
       
        r_image = yolo.detect_image(image)

        if r_image==None:
            continue
        # r_image.show()  # Display first, then save
        r_image.save(result_path+filename)

    yolo.close_session()


if __name__ == '__main__':
    detect_img_for_test(YOLO())

