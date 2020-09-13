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

    net["fc8-1"] = DenseLayer(
        net["fc7_dropout"], num_units=numberOfClasses, nonlinearity=None
    )
    net["prob"] = NonlinearityLayer(net["fc8-1"], softmax)

    return net["prob"]


# load data
def load_dataset(rgbDir):

    dataX = []
    dataY = []

    # process RGB Videos
    n = 0
    for root, dirs, files in os.walk(rgbDir):
        for file in files:
            gestureClip = os.path.join(root, file).replace("\\", "/")
            frames = []
            gray_frames = []
            cap = cv2.VideoCapture(gestureClip)

            # Get all frames from video clip
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

            # Get all frames from corresponding depth video clip
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
            # print gestureClip + ' ' + str(len(frames))
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
                # d_channel = numpy.zeros((img_rows,img_cols), numpy.uint8)
                # d_channel[:] = 255
                y_channel = gray_frames[int(math.floor(frame_interval * i))]
                img_RGBD = cv2.merge(
                    (b_channel, g_channel, r_channel)
                )  # ,  y_channel))#,  d_channel))
                representative_frames.append(img_RGBD)
                i = i + 1

            # Add clip to dataset
            dataX.append(numpy.rollaxis(numpy.array(representative_frames), 3, 0))
            dataY.append(int(file.split(".")[0].split("_")[10]) % 10)
            n = n + 1
        # if n==5 :
        #     break

    return (
        numpy.array(dataX) / numpy.float32(256),
        numpy.array(dataY, dtype=numpy.int32),
    )


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx : start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Trains the model
def train(
    trainsetInput,
    trainsetLabel,
    testsetInput,
    testsetLabel,
    num_epochs=100,
    minibatch=5,
    convergenceTrainLoss=0.001,
):

    # Prepare Theano variables for inputs and targets
    tensor5 = T.TensorType("float32", (False,) * 5)
    input_var = tensor5("inputs")
    target_var = T.ivector("targets")

    # Build the network
    start_time = time.time()
    network = build_c3d_model(input_var=input_var)
    print("Building the network took " + str(time.time() - start_time) + " seconds")

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    l2_penalty = (
        lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2
        )
        * 1e-4
    )
    loss = loss.mean() + l2_penalty
    train_acc = T.mean(
        T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX
    )
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.001, momentum=0.9
    )

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(
        T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX
    )

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    start_time = time.time()
    train_fn = theano.function(
        [input_var, target_var], [loss, train_acc], updates=updates
    )

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    print("Compiling functions " + str(time.time() - start_time) + " seconds")
    # Compile a second function computing the validation loss and accuracy:

    # Finally, launch the training loop.
    print("Starting training...")

    # best_acc
    best_acc = 0

    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        for batch in iterate_minibatches(
            trainsetInput, trainsetLabel, minibatch, shuffle=True
        ):
            inputs, targets = batch
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(
            testsetInput, testsetLabel, minibatch, shuffle=False
        ):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Best model saving
        if val_acc > best_acc:
            print("Saving weights of best model so far")
            # pickle.dump(network, open('current_network.pkl','w'), protocol=pickle.HIGHEST_PROTOCOL)
            numpy.savez(
                "network/current_network.npz",
                *lasagne.layers.get_all_param_values(network)
            )
            best_acc = val_acc

        # Then we print the results for this epoch:
        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time
            )
        )
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print(
            "  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100)
        )
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        # Declare convergence
        if (1.0 * train_err / train_batches) < convergenceTrainLoss:
            print("Stopping training. Network already converged.")
            break


    print("Training complete")


# START HERE
print("Loading train data")

trainX, trainY = load_dataset("data/train_rgb")
print("Loading test data")
testX, testY = load_dataset("data/test_rgb")
print(trainX.shape)
print(trainY.shape)
train(
    trainsetInput=trainX,
    trainsetLabel=trainY,
    testsetInput=testX,
    testsetLabel=testY,
    minibatch=30,
    num_epochs=50,
)
