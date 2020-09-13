import lasagne
from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
from lasagne.layers import ConcatLayer
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


from lasagne.layers.shape import PadLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer

GESTURE_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

clipDepth = 16
img_rows = 96
img_cols = 96
numberOfClasses = 10
channels = 3


def build_c3d_model(input_var=None):
    """
    Builds C3D model
    Returns
    -------
    dict
        A dictionary containing the network layers, where the output layer is at key 'prob'
    """
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
    net["fc6_dropout"] = DropoutLayer(net["fc6-1"], p=0.5)
    net["fc7-1"] = DenseLayer(
        net["fc6_dropout"], num_units=4096, nonlinearity=lasagne.nonlinearities.rectify
    )
    net["fc7_dropout"] = DropoutLayer(net["fc7-1"], p=0.5)

    net["fc8-1"] = DenseLayer(net["fc7_dropout"], num_units=10, nonlinearity=None)
    net["prob"] = NonlinearityLayer(net["fc8-1"], softmax)

    # Load pretrained model
    # print ('Loading the trained c3d network')
    with numpy.load("network/current_network.npz") as f:
        param_values = [f["arr_%d" % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net["prob"], param_values)

    return net["prob"]


# load data
def measurePerf(rgbDir, depthDir):

    dataX = []
    dataY = []

    totalPredictionTime = 0.0
    totalpredictions = 0

    # Prepare Theano variables for inputs and targets
    tensor5 = T.TensorType("float32", (False,) * 5)
    input_var = tensor5("inputs")
    target_var = T.ivector("targets")

    # Build the network
    print(
        "================================================================================================"
    )
    print("For Dataset " + rgbDir)
    start_time = time.time()
    network = build_c3d_model(input_var=input_var)
    print("Building the network took " + str(time.time() - start_time) + " seconds")

    print("Compiling Prediction Function")
    start_time = time.time()
    prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_function = theano.function([input_var], prediction)
    print("Compiling prediction function " + str(time.time() - start_time) + " seconds")
    GESTURE_COUNT = numpy.zeros([len(GESTURE_CLASSES), len(GESTURE_CLASSES)])

    # process Videos
    for root, dirs, files in os.walk(rgbDir):
        for file in files:
            gestureClip = os.path.join(root, file).replace("\\", "/")
            tag = str(int(file.split(".")[0].split("_")[10]) % 10)
            frames = []
            gray_frames = []
            cap = cv2.VideoCapture(gestureClip)

            # Get all frames from rgb video clip
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.resize(
                        frame, (img_rows, img_cols), interpolation=cv2.INTER_AREA
                    )
                    frames.append(frame)
                    gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    break
            cap.release()

            # Get all frames from depth video clip
            depthClip = gestureClip.replace("_rgb", "_d").replace("M_", "K_")
            depthFrames = []
            cap = cv2.VideoCapture(depthClip)

            # Get all frames from depth video clip
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.resize(
                        frame, (img_rows, img_cols), interpolation=cv2.INTER_AREA
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    depthFrames.append(frame)
                else:
                    break
            cap.release()
            # print depthClip + ' ' + str(len(depthFrames))

            # Extract representative frames
            frame_interval = 1.0 * len(frames) / clipDepth
            representative_frames = []
            i = 0
            while i < clipDepth:
                # representative_frames.append(frames[int(math.floor(frame_interval*i))])

                # For depth data
                b_channel, g_channel, r_channel = cv2.split(
                    frames[int(math.floor(frame_interval * i))]
                )
                d_channel = depthFrames[int(math.floor(frame_interval * i))]
                y_channel = gray_frames[int(math.floor(frame_interval * i))]
                img_RGBD = cv2.merge((b_channel, g_channel, r_channel))  # , d_channel))
                representative_frames.append(img_RGBD)

                i = i + 1

            testsetInput = []
            testsetInput.append(
                numpy.rollaxis(numpy.array(representative_frames), 3, 0)
                / numpy.float32(256)
            )
            testsetInput = numpy.array(testsetInput)

            before_prediction = time.time()
            result = predict_function(testsetInput)
            totalPredictionTime = totalPredictionTime + (
                time.time() - before_prediction
            )
            totalpredictions = totalpredictions + 1
            network_prediction = numpy.argmax(result)
            GESTURE_COUNT[network_prediction][GESTURE_CLASSES.index(tag)] = (
                GESTURE_COUNT[network_prediction][GESTURE_CLASSES.index(tag)] + 1
            )

    # Print results; exclude background class
    print(GESTURE_COUNT)
    TOTAL_PER_CLASS = sum(GESTURE_COUNT)
    TOTAL_CORRECT = 0
    i = 0
    while i < len(GESTURE_CLASSES):
        print(
            GESTURE_CLASSES[i]
            + " = "
            + str(GESTURE_COUNT[i][i])
            + "/"
            + str(TOTAL_PER_CLASS[i])
            + "("
            + str(round(GESTURE_COUNT[i][i] * 100.0 / TOTAL_PER_CLASS[i], 2))
            + "%)"
        )
        TOTAL_CORRECT = TOTAL_CORRECT + GESTURE_COUNT[i][i]
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
    print(
        "Average prediction time for dataset: "
        + str(totalPredictionTime / totalpredictions)
    )


# START HERE
# measurePerf('D:\\Documents\\MS CS\\Thesis\\lasagne-models\\c3d\\train_rgb')
# measurePerf('D:\\Documents\\MS CS\\Thesis\\lasagne-models\\c3d\\test_rgb')

measurePerf("data/train_rgb", "data/train_d")
measurePerf("data/test_rgb", "data/test_d")
