from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
from lasagne.layers import ConcatLayer, InputLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import ElemwiseSumLayer

from lasagne.nonlinearities import softmax, linear, rectify
import lasagne.nonlinearities


from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer


import os
import cv2
import numpy
import time
import pickle
import _pickle as cPickle
import math

import theano
import theano.tensor as T

from random import shuffle

from lasagne.layers.shape import PadLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import datetime

from tensorflow import image as ti
import tensorflow as tf
from skimage import transform as st

clipDepth = 16
img_rows = 96
img_cols = 96
numberOfClasses = 27
channels = 4

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

mapping = {
    "Swiping Up": "Previous Series",
    "Swiping Down": "Next Series",
    "Swiping Left": "Previous Image",
    "Swiping Right": "Next Image",
    "Sliding Two Fingers Up": "Browse Up",
    "Sliding Two Fingers Down": "Browse Down",
    "Sliding Two Fingers Left": "Browse Left",
    "Sliding Two Fingers Right": "Browse Right",
    "Rolling Hand Backward": "Play Series In Reverse",
    "Rolling Hand Forward": "Play Series",
    "Stop Sign": "Stop Playing Series",
    "Pushing Hand Away": "Stop Playing Series",
    "Pushing Two Fingers Away": "Increase Brightness",
    "Pulling Two Fingers In": "Decrease Brightness",
    "Turning Hand Counterclockwise": "Rotate Counter-clockwise",
    "Turning Hand Clockwise": "Rotate Clockwise",
    "Zooming In With Full Hand": "Zoom In",
    "Zooming In With Two Fingers": "Zoom In",
    "Zooming Out With Full Hand": "Zoom Out",
    "Zooming Out With Two Fingers": "Zoom Out",
    "Thumb Up": "Thumb Up (Unlock)",
    "Thumb Down": "Thumb Down (Lock)",
    "Doing other things": "No System Action",
    "No gesture": "No System Action",
    "Drumming Fingers": "No System Action",
    "Pulling Hand In": "No System Action",
    "Shaking Hand": "No System Action",
}


def map_gesture(labels):
    mapped_labels = []
    for lbl in labels:
        mapped_labels.append(mapping[lbl])

    return mapped_labels


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(24, 12)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=1.0)
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # print(cm)
    cm_sum = numpy.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = numpy.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%" % (p)  #'%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%" % (p)  #'%.1f%%\n%d' % (p, c)
    col_labels = [
        "PrS",
        "NS",
        "PrI",
        "NI",
        "BU",
        "BD",
        "BL",
        "BR",
        "PlSIR",
        "PlS",
        "SPlS",
        "IB",
        "DB",
        "RCC",
        "RC",
        "ZI",
        "ZO",
        "TUU",
        "TDL",
        "NSA",
    ]
    cm = pd.DataFrame(cm_perc, index=labels, columns=col_labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    # plt.xticks(rotation=45)
    # plt.ylabel('ylabel', fontsize=30)
    fig, ax = plt.subplots(figsize=figsize)
    # ax.set_xticklabels(rotation=45)
    # sns.set(font_scale=1.4)
    g = sns.heatmap(cm, annot=annot, fmt="", ax=ax)
    # g.set_xticklabels(g.get_xticklabels(), rotation = 45)
    # sns.set(font_scale=1.5)
    plt.savefig(filename)


def scale_frame(img, space_scale):
    if space_scale > 1.0:
        scale_out_rgb = (
            st.rescale(img, scale=space_scale, mode="constant") * 255
        ).astype(numpy.uint8)
        start_row = int((scale_out_rgb.shape[0] - img.shape[0]) / 2)
        start_col = int((scale_out_rgb.shape[1] - img.shape[1]) / 2)
        return scale_out_rgb[
            start_row : start_row + img.shape[0], start_col : start_col + img.shape[1]
        ]
    elif space_scale < 1.0:
        scale_out_rgb = (
            st.rescale(img, scale=space_scale, mode="constant") * 255
        ).astype(numpy.uint8)
        start_row = int((img.shape[0] - scale_out_rgb.shape[0]) / 2)
        start_col = int((img.shape[1] - scale_out_rgb.shape[1]) / 2)
        padded_rgb = numpy.zeros((img.shape[0], img.shape[1], 3), numpy.uint8)
        padded_rgb[
            start_row : start_row + scale_out_rgb.shape[0],
            start_col : start_col + scale_out_rgb.shape[1],
        ] = scale_out_rgb
        return padded_rgb
    else:
        return img


def get_representative_frames(frames, clipDepth=16, time_scale=1.0, space_scale=1.0):
    # print(frames)

    scaled_frames = []
    start = int((len(frames) - time_scale * len(frames)) / 2)
    end = int(time_scale * len(frames))
    for i in range(start, end):
        index = i
        if index < 0:
            index = 0
        if index >= len(frames):
            index = len(frames) - 1
        scaled_frames.append(frames[index])

    frame_interval = 1.0 * len(scaled_frames) / clipDepth
    rep_frames = []
    for i in range(0, clipDepth):
        # print(scaled_frames[int(math.floor(frame_interval*i))])
        img = cv2.imread(scaled_frames[int(math.floor(frame_interval * i))])
        img = cv2.resize(img, (img_rows, img_cols))
        img = scale_frame(img, space_scale)


        b_channel, g_channel, r_channel = cv2.split(img)
        d_channel = numpy.zeros((img_rows, img_cols), numpy.uint8)
        d_channel[:] = 255

        # d_frames.append(d_channel)
        img_RGBD = cv2.merge((b_channel, g_channel, r_channel, d_channel))

        rep_frames.append(img_RGBD)
    return numpy.rollaxis(numpy.array(rep_frames) / numpy.float32(256), 3, 0)
    # return(numpy.array(rep_frames))


def build_c3d_model(input_var=None, for_training=True):
    net = {}
    # net['input'] = InputLayer((None, 3, 16, 112, 112), input_var=input_var)
    net["input"] = InputLayer(
        (None, channels, clipDepth, img_rows, img_cols), input_var=input_var
    )  # with depth channel

    # ----------- 1st layer group ---------------
    net["conv1a"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["input"],
            64,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            flip_filters=False,
        )
    )
    net["pool1"] = MaxPool3DDNNLayer(
        net["conv1a"], pool_size=(1, 2, 2), stride=(1, 2, 2)
    )

    # ------------- 2nd layer group --------------
    net["conv2a"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["pool1"],
            128,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    net["pool2"] = MaxPool3DDNNLayer(
        net["conv2a"], pool_size=(2, 2, 2), stride=(2, 2, 2)
    )

    # ----------------- 3rd layer group --------------
    net["conv3a"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["pool2"],
            256,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    net["conv3b"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["conv3a"],
            256,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    net["pool3"] = MaxPool3DDNNLayer(
        net["conv3b"], pool_size=(2, 2, 2), stride=(2, 2, 2)
    )

    # ----------------- 4th layer group --------------
    net["conv4a"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["pool3"],
            512,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    net["conv4b"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["conv4a"],
            512,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    net["pool4"] = MaxPool3DDNNLayer(
        net["conv4b"], pool_size=(2, 2, 2), stride=(2, 2, 2)
    )

    # ----------------- 5th layer group --------------
    net["conv5a"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["pool4"],
            512,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    net["conv5b"] = lasagne.layers.batch_norm(
        Conv3DDNNLayer(
            net["conv5a"],
            512,
            (3, 3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
    )
    # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
    net["pad"] = PadLayer(net["conv5b"], width=[(0, 1), (0, 1)], batch_ndim=3)
    net["pool5"] = MaxPool3DDNNLayer(
        net["pad"], pool_size=(2, 2, 2), pad=(0, 0, 0), stride=(2, 2, 2)
    )
    net["fc6-1"] = DenseLayer(
        net["pool5"], num_units=4096, nonlinearity=lasagne.nonlinearities.rectify
    )
    net["fc6_dropout"] = DropoutLayer(net["fc6-1"], p=0.1)
    net["fc7-1"] = DenseLayer(
        net["fc6_dropout"], num_units=4096, nonlinearity=lasagne.nonlinearities.rectify
    )
    net["fc7_dropout"] = DropoutLayer(net["fc7-1"], p=0.1)

    if for_training:
        # net['fc8-1']  = DenseLayer(net['fc7_dropout'], num_units=10, nonlinearity=None)
        # net['prob']  = NonlinearityLayer(net['fc8-1'], softmax)
        net["fc8-1"] = DenseLayer(
            net["fc7_dropout"], num_units=numberOfClasses, nonlinearity=None
        )
        net["prob"] = NonlinearityLayer(net["fc8-1"], softmax)

        # Load pretrained model
        print("Loading the trained c3d network")
        with numpy.load("network/validation_9452.npz") as f:
            param_values = [f["arr_%d" % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net["prob"], param_values)
    return net["prob"]


######################################################################################################################################
totalPredictionTime = 0.0
totalpredictions = 0
totalLoss = 0

# Prepare Theano variables for inputs and targets
tensor5 = T.TensorType("float32", (False,) * 5)
input_var = tensor5("inputs")
target_var = T.ivector("targets")


# Build the network
print(
    "================================================================================================"
)
print("For Evaluation ")
start_time = time.time()
network = build_c3d_model(input_var=input_var)
print("Building the network took " + str(time.time() - start_time) + " seconds")

print("Compiling Prediction Function")
start_time = time.time()
prediction = lasagne.layers.get_output(network, deterministic=True)
predict_function = theano.function([input_var], prediction)
print("Compiling prediction function " + str(time.time() - start_time) + " seconds")


testFile = open("jester-v1-validation.csv", "r")
labelsFile = open("jester-v1-labels.csv", "r")
predictionsFile = open("jester-v1-predictions.csv", "w")
missesFile = open("misses.csv", "w")
labels = []
for line in labelsFile:
    labels.append(line.strip())

own_labels = [
    "Previous Series",
    "Next Series",
    "Previous Image",
    "Next Image",
    "Browse Up",
    "Browse Down",
    "Browse Left",
    "Browse Right",
    "Play Series In Reverse",
    "Play Series",
    "Stop Playing Series",
    "Increase Brightness",
    "Decrease Brightness",
    "Rotate Counter-clockwise",
    "Rotate Clockwise",
    "Zoom In",
    "Zoom Out",
    "Thumb Up (Unlock)",
    "Thumb Down (Lock)",
    "No System Action",
]

# GESTURE_COUNT=numpy.zeros([len(labels), len(labels)])
GESTURE_COUNT = numpy.zeros([len(own_labels), len(own_labels)])
labelsFile.close()

# For confusion matrix
y_true = []
y_pred = []

for line in testFile:
    tokens = line.strip().split(";")
    sample = tokens[0]
    tag = labels.index(tokens[1])

    frames = []
    for image in sorted(os.listdir(os.path.join("20bn-jester-v1", sample))):
        frames.append(os.path.join("20bn-jester-v1", sample, image))

    dataX = []
    # dataX.append(get_representative_frames(frames, time_scale=1.0, space_scale=0.8))
    # dataX.append(get_representative_frames(frames, time_scale=1.0, space_scale=0.9))
    dataX.append(get_representative_frames(frames, time_scale=1.0))
    # dataX.append(get_representative_frames(frames, time_scale=1.0, space_scale=1.1))
    # dataX.append(get_representative_frames(frames, time_scale=1.0, space_scale=1.2))

    # dataX.append(get_representative_frames(frames, time_scale=0.9, space_scale=0.8))
    # dataX.append(get_representative_frames(frames, time_scale=0.9, space_scale=0.9))
    # dataX.append(get_representative_frames(frames, time_scale=0.9, space_scale=1.0))
    # dataX.append(get_representative_frames(frames, time_scale=0.9, space_scale=1.1))
    # dataX.append(get_representative_frames(frames, time_scale=0.9, space_scale=1.2))

    # dataX.append(get_representative_frames(frames, time_scale=1.1, space_scale=0.8))
    # dataX.append(get_representative_frames(frames, time_scale=1.1, space_scale=0.9))
    # dataX.append(get_representative_frames(frames, time_scale=1.1, space_scale=1.0))
    # dataX.append(get_representative_frames(frames, time_scale=1.1, space_scale=1.1))
    # dataX.append(get_representative_frames(frames, time_scale=1.1, space_scale=1.2))

    # dataX.append(get_representative_frames(frames, time_scale=1.4))
    # dataX.append(get_representative_frames(frames, time_scale=1.3))
    # dataX.append(get_representative_frames(frames, time_scale=1.2))
    # dataX.append(get_representative_frames(frames, time_scale=1.1))
    # dataX.append(get_representative_frames(frames, time_scale=0.9))
    # dataX.append(get_representative_frames(frames, time_scale=0.8))
    # dataX.append(get_representative_frames(frames, time_scale=0.7))
    # dataX.append(get_representative_frames(frames, time_scale=0.6))
    dataX = numpy.array(dataX)

    # dataX = numpy.expand_dims(numpy.rollaxis( numpy.array(representative_frames) /numpy.float32(256), 3, 0), axis=0)
    # dataX = numpy.expand_dims(numpy.rollaxis( representative_frames /numpy.float32(256), 3, 0), axis=0)
    before_prediction = time.time()
    result = predict_function(dataX)
    # print(result.shape)
    totalPredictionTime = totalPredictionTime + (time.time() - before_prediction)
    totalpredictions = totalpredictions + 1
    # network_prediction = numpy.argmax(result)
    # totalLoss = totalLoss + (math.log(result[0][network_prediction]) * -1)
    ave_result = result.mean(axis=0)
    network_prediction = numpy.argmax(ave_result)
    totalLoss = totalLoss + (math.log(ave_result[network_prediction]) * -1)

    # MAP
    network_prediction = own_labels.index(mapping[labels[network_prediction]])
    tag = own_labels.index(mapping[labels[tag]])

    GESTURE_COUNT[network_prediction][tag] = GESTURE_COUNT[network_prediction][tag] + 1
    predictionsFile.write(sample + ";" + labels[network_prediction] + "\n")
    if totalpredictions % 500 == 0:
        print("Total Predictions  " + str(totalpredictions).zfill(5))
    if totalpredictions >= 15000:
        break
    if network_prediction != tag:
        missesFile.write(
            sample
            + ";"
            + labels[tag]
            + ";"
            + labels[network_prediction]
            + ";"
            + str(ave_result[network_prediction])
            + "\n"
        )
    y_true.append(labels[tag])
    y_pred.append(labels[network_prediction])


TOTAL_PER_CLASS = sum(GESTURE_COUNT)
TOTAL_SA = TOTAL_CORRECT = 0
i = 0
print("")
while i < len(own_labels):
    print(
        own_labels[i]
        + " = "
        + str(GESTURE_COUNT[i][i])
        + "/"
        + str(TOTAL_PER_CLASS[i])
        + "("
        + str(round(GESTURE_COUNT[i][i] * 100.0 / TOTAL_PER_CLASS[i], 2))
        + "%)"
    )
    TOTAL_CORRECT = TOTAL_CORRECT + GESTURE_COUNT[i][i]
    if i < len(own_labels) - 1:
        TOTAL_CORRECT_SA = TOTAL_CORRECT
    i = i + 1
print("")
print(
    "OVERALL = "
    + str(TOTAL_CORRECT)
    + "/"
    + str(sum(TOTAL_PER_CLASS))
    + "("
    + str(round(TOTAL_CORRECT * 100.0 / sum(TOTAL_PER_CLASS), 2))
    + "%)"
)
print("Loss = " + str(round(totalLoss / totalpredictions, 4)))


print(
    "System Action Accuracy = "
    + str(TOTAL_CORRECT_SA)
    + "/"
    + str(sum(TOTAL_PER_CLASS[: len(own_labels) - 1]))
    + "("
    + str(
        round(TOTAL_CORRECT_SA * 100.0 / sum(TOTAL_PER_CLASS[: len(own_labels) - 1]), 2)
    )
    + "%)"
)

# Non System Action Performance/
TP = GESTURE_COUNT[len(own_labels) - 1][len(own_labels) - 1]
print(
    "Non System Action Precision = "
    + str(TP)
    + "/"
    + str(sum(GESTURE_COUNT[len(own_labels) - 1]))
    + "("
    + str(round(TP * 100.0 / sum(GESTURE_COUNT[len(own_labels) - 1]), 2))
    + "%)"
)
print(
    "Non System Action Recall = "
    + str(TP)
    + "/"
    + str(sum(GESTURE_COUNT[:, len(own_labels) - 1]))
    + "("
    + str(round(TP * 100.0 / sum(GESTURE_COUNT[:, len(own_labels) - 1]), 2))
    + "%)"
)

testFile.close()
predictionsFile.close()
missesFile.close()

own_labels = [
    "Previous Series",
    "Next Series",
    "Previous Image",
    "Next Image",
    "Browse Up",
    "Browse Down",
    "Browse Left",
    "Browse Right",
    "Play Series In Reverse",
    "Play Series",
    "Stop Playing Series",
    "Increase Brightness",
    "Decrease Brightness",
    "Rotate Counter-clockwise",
    "Rotate Clockwise",
    "Zoom In",
    "Zoom Out",
    "Thumb Up (Unlock)",
    "Thumb Down (Lock)",
    "No System Action",
]
cm_analysis(
    map_gesture(y_true), map_gesture(y_pred), "confusion_matrix.png", own_labels
)
