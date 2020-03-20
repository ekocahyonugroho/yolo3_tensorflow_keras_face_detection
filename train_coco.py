#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""

import numpy as np
import keras.backend as K
import keras.callbacks as KCall
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

#VISUALIZE TRAIN RESULTS
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(0)

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

import pickle
from pickle import load

def _main():
    annotation_path = 'datasets/coco_dataset/train2017.txt'  # data
    val_annotation_path = 'datasets/coco_dataset/val2017.txt'  # data
    classes_path = 'datasets/coco_dataset/coco_classes.txt'  # category class

    log_dir = 'logs/coco/'  # log folder
    
    pretrained_path = 'model_data/yolo_weights.h5'  # Pre-training model
    #pretrained_path = log_dir + 'trained_weights_final.h5'  # Pre-training model
    anchors_path = 'configs/yolo_anchors.txt'  # anchors

    class_names = get_classes(classes_path)  # Category list
    num_classes = len(class_names)  # Number of categories
    anchors = get_anchors(anchors_path)  # Anchors list

    input_shape = (416, 416)  # Multiple of 32, input image original 416 x 416
    LR_unfreeze = 0.0001
    LR_freeze = 0.001

    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2,
                         weights_path=pretrained_path)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    # val_loss error. coba diganti dengan val_loss:2.f sebelumnya 3.f. ternyata val_loss error karena dataset yang terlalu kecil sehingga tidak bisa meghitung total loss
    #checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    checkpoint = ModelCheckpoint(log_dir + 'trained_weights_last_checkpoint_epoch_{epoch:03d}.h5',
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=False, period=1)  # Store only weights，
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)  # Reduce learning rate when evaluation indicators are not improving
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)  # Test set accuracy rate, stop before falling
    csv_logger = CSVLogger(log_dir+"training_log.csv", separator=';', append=True) # Save training history in CSV File

    val_split = .2  # Training and verification ratio
    batch_size = 8 # batch size
    with open(annotation_path) as f:
        lines = f.readlines()
    with open(val_annotation_path) as v:
        val_lines = v.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    #num_val = int(len(lines) * val_split)  # Number of verification sets
    num_val = len(val_lines)
    #num_train = len(lines) - num_val  # Number of training sets
    num_train = len(lines)

    """
    Think of the target as an input, form a multi-input model, write loss as a layer, as the final output, when building the model,
     Just define the output of the model as loss, and when compile,
     Set loss directly to y_pred (because the output of the model is loss, so y_pred is loss),
     Ignore y_true. When training, y_true just throws an array of shapes into it.
    """

    if True:
        model.compile(optimizer=Adam(lr=LR_freeze), loss={
            # use custom yolo_loss Lambda layer. tadi ditambahin metrics=['accuracy']
            'yolo_loss': lambda y_true, y_pred: y_pred})  # Loss function

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            #validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_data=data_generator_wrapper(val_lines[:num_val], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=5,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, csv_logger])
        model.save_weights(log_dir + 'trained_weights_freezed_layers.h5')  # Store the final parameters, and then pass the callback storage during the training process


    if True:  # All training. tadi ditambahin metrics=['accuracy']
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=LR_unfreeze),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            #validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_data=data_generator_wrapper(val_lines[:num_val], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=10,
                            initial_epoch=5,
                            callbacks=[logging, checkpoint, csv_logger, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/darknet53_weights.h5'):
    K.clear_session()  # Clear session
    image_input = Input(shape=(None, None, 3))  # Picture input format
    h, w = input_shape  # size
    num_anchors = len(anchors)  # Number of anchors

    # Three scales of YOLO, the number of anchors per scale, the number of categories + the border 4 + confidence 1
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)  # model
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:  # Load pre-training model
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)  # Load parameters, skip errors
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False  # Close the training of the other layers
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss,
                        output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5})(model_body.output + y_true)  # Behind the input, the front is the output
    model = Model([model_body.input] + y_true, model_loss)  # Models, inputs, and outputs

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)  # Get pictures and boxes. Choose data augmentation True for random, False for changing scaling/flip/etc
            image_data.append(image)  # add pictures
            box_data.append(box)  # Add box
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)  # Truth
        yield [image_data] + y_true, np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
    For condition checking
    """
    n = len(annotation_lines)  # Number of lines to annotate the image
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def visualize_training(history, title):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    #training_accuracy = history.history['y_pred_mean']
    #test_accuracy = history.history['val_y_pred_mean']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel(title+' Epoch')
    plt.ylabel(title+' Loss')
    plt.show()

    #plt.plot(epoch_count, training_accuracy, 'r--')
    #plt.plot(epoch_count, test_accuracy, 'b-')
    #plt.legend(['y_pred_mean', 'val_y_pred_mean'])
    #plt.xlabel(title+' Epoch')
    #plt.ylabel(title+' y_pred_mean')
    #plt.show()

def visualize_training_history(oldhstry, hstry):
    oldhstry['loss'].extend(hstry.history['loss'])
    oldhstry['val_loss'].extend(hstry.history['val_loss'])

    # Plotting the Loss vs Epoch Graphs
    plt.plot(oldhstry['loss'])
    plt.plot(oldhstry['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def y_pred_mean(y_true, y_pred):
    return K.mean(y_pred)

if __name__ == '__main__':
    _main()
