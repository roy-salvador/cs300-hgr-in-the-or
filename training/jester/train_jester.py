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


from lasagne import utils
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
import random

from lasagne.layers.shape import PadLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import datetime
import imutils

clipDepth = 16
img_rows = 96
img_cols = 96
numberOfClasses = 27
channels = 4

LR = theano.shared(lasagne.utils.floatX(0.01))
ITER_NUMBER = theano.shared(lasagne.utils.floatX(0))


def build_c3d_model(input_var=None, for_training=True):

    net = {}
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
    net["fc6_dropout"] = DropoutLayer(net["fc6-1"], p=0.8)
    net["fc7-1"] = DenseLayer(
        net["fc6_dropout"], num_units=4096, nonlinearity=lasagne.nonlinearities.rectify
    )
    net["fc7_dropout"] = DropoutLayer(net["fc7-1"], p=0.8)

    if for_training:

        net["fc9-1"] = DenseLayer(
            net["fc7_dropout"], num_units=numberOfClasses, nonlinearity=None
        )
        net["prob"] = NonlinearityLayer(net["fc9-1"], softmax)

        # Load pretrained model
        #print("Loading the trained c3d network")
        #with numpy.load("network/last_network.npz") as f:
        #    param_values = [f["arr_%d" % i] for i in range(len(f.files))]


        lasagne.layers.set_all_param_values(net["prob"], param_values)



    return net["prob"]


def perform_validation(
    predict_function, validation_file="jester-v1-validation.csv", n_samples=15000
):
    labelsFile = open("jester-v1-labels.csv", "r")
    labels = []
    for line in labelsFile:
        labels.append(line.strip())
    labelsFile.close()

    print("**********************************")
    print("Performing validation for " + str(n_samples) + " samples")
    start_time = time.time()
    correctpredictions = 0
    totalpredictions = 0
    totalLoss = 0
    testFile = open(validation_file, "r")
    for line in testFile:
        tokens = line.strip().split(";")
        sample = tokens[0]
        tag = labels.index(tokens[1])

        frames = []
        for image in sorted(os.listdir(os.path.join("20bn-jester-v1", sample))):
            frames.append(os.path.join("20bn-jester-v1", sample, image))

        # Get representative Frame
        frame_interval = 1.0 * len(frames) / clipDepth
        # rgb_frames = []
        # d_frames = []
        representative_frames = []
        i = 0
        while i < clipDepth:
            img = cv2.imread(frames[int(math.floor(frame_interval * i))])
            img = cv2.resize(img, (img_rows, img_cols))
            # rgb_frames.append(img)
            b_channel, g_channel, r_channel = cv2.split(img)

            d_channel = numpy.zeros((img_rows,img_cols), numpy.uint8)
            d_channel[:] =255

            # d_frames.append(d_channel)
            img_RGBD = cv2.merge((b_channel, g_channel, r_channel, d_channel))
            representative_frames.append(img_RGBD)

            i = i + 1

        dataX = numpy.expand_dims(
            numpy.rollaxis(
                numpy.array(representative_frames) / numpy.float32(256), 3, 0
            ),
            axis=0,
        )
        result = predict_function(dataX)
        network_prediction = numpy.argmax(result)
        totalpredictions = totalpredictions + 1
        totalLoss = totalLoss + (math.log(result[0][network_prediction]) * -1)
        if tag == network_prediction:
            correctpredictions = correctpredictions + 1
        if totalpredictions >= n_samples:
            break

    print("  validation loss:\t\t" + str(round(totalLoss / totalpredictions, 6)))
    print(
        "  validation accuracy:\t\t"
        + str(round(correctpredictions * 100.0 / totalpredictions, 2))
        + "%"
    )
    testFile.close()
    print("Validation took " + str(time.time() - start_time) + " seconds")
    print("**********************************")
    return round(correctpredictions * 100.0 / totalpredictions, 2)



############################ Batch iterator ustilizing Data augmentation
def iterate_minibatches_with_augmentation(
    rgbd_clips, n_clips_at_a_time=800, batchsize=40
):

    file_count = 0
    while file_count < len(rgbd_clips):
        n = 0
        dataX = []
        dataY = []

        while n < n_clips_at_a_time and file_count < len(rgbd_clips):

            with numpy.load(rgbd_clips[file_count]) as data:
                aug_rgb_clips = data["rgb"]
                # aug_depth_clips = data['depth']
                # print(aug_rgb_clips.shape)

            tagClass = int(
                os.path.basename(rgbd_clips[file_count].split("_")[1].split(".")[0])
            )
            # print(tagClass)
            # aug_rgbd_clips, aug_depth_clips = video_augmentation.augment_video(frames, depthFrames, img_rows, img_cols, clipDepth, (tagClass==16))
            # representative_frames = []

            for i in range(0, len(aug_rgb_clips)):
                representative_frames = []

                p = random.uniform(0.8, 1.5)
                invert = random.randint(0, 2) == 0
                sharpen = random.randint(0, 2) == 0
                angle = random.uniform(-10, 10) % 360
                color_channel_i = random.randint(0, 5)
                finalClass = tagClass
                flip = random.randint(0, 1) == 1
                if finalClass in ([0, 1, 6, 7, 14, 15]):
                    if flip:
                        if finalClass == 0:
                            finalClass = 1
                        elif finalClass == 1:
                            finalClass = 0
                        elif finalClass == 6:
                            finalClass = 7
                        elif finalClass == 7:
                            finalClass = 6
                        elif finalClass == 14:
                            finalClass = 15
                        elif finalClass == 15:
                            finalClass = 14

                for j in range(0, clipDepth):

                    current_img = aug_rgb_clips[i][j]

                    # flip
                    if flip:
                        current_img = cv2.flip(current_img, 1)

                    # color inversion
                    if invert:
                        current_img = cv2.bitwise_not(current_img)

                    # sharpen
                    if sharpen:
                        kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        current_img = cv2.filter2D(current_img, -1, kernel)

                    # random constant color channel
                    if color_channel_i < 3:
                        current_img[:, :, color_channel_i] = random.randint(0, 255)

                    # random rotate
                    current_img = imutils.rotate_bound(current_img, angle)

                    # random lower esolution
                    if p < 1.0:
                        current_img = cv2.resize(
                            current_img,
                            (
                                int(current_img.shape[0] * p),
                                int(current_img.shape[1] * p),
                            ),
                        )

                    current_img = cv2.resize(
                        current_img,
                        (aug_rgb_clips[i][j].shape[0], aug_rgb_clips[i][j].shape[1]),
                    )

                    b_channel, g_channel, r_channel = cv2.split(current_img)
                    # d_channel = aug_depth_clips[i][j]
                    # d_channel = numpy.zeros((img_rows,img_cols), numpy.uint8)
                    # d_channel[:] =255

                    img_RGBD = cv2.merge(
                        (b_channel, g_channel, r_channel)
                    )  # , d_channel))
                    representative_frames.append(img_RGBD)

                # Add clip to dataset
                dataX.append(numpy.rollaxis(numpy.array(representative_frames), 3, 0))
                # tagClass = int(file.split('_')[3])
                dataY.append(finalClass)
            n = n + 1
            file_count = file_count + 1

        dataX = numpy.array(dataX) / numpy.float32(256)
        dataY = numpy.array(dataY, dtype=numpy.int32)
        indices = numpy.arange(len(dataX))
        numpy.random.shuffle(indices)
        for start_idx in range(0, len(dataX) - batchsize + 1, batchsize):
            excerpt = indices[start_idx : start_idx + batchsize]
            yield dataX[excerpt], dataY[excerpt]


# Trains the model
def train(
    trainsetInput,
    trainsetLabel,
    testsetInput,
    testsetLabel,
    num_epochs=100,
    minibatch=5,
    convergenceTrainLoss=0.0001,
    using_segmented=False,
    train_dir="train.scaled.jester",
    clips_per_subepoch=800,
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
    prediction = lasagne.layers.get_output(network, input_var)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    l1_penalty = (
        lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l1
        )
        * 0
    )
    l2_penalty = (
        lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2
        )
        * 1e-6
    )
    # l2_penalty = 0
    loss = loss.mean() + l2_penalty + l1_penalty
    train_acc = T.mean(
        T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX
    )

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=LR, momentum=0.9
    )
    # updates = lasagne.updates.sgd(loss, params, learning_rate=LR)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    """
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    """
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    start_time = time.time()
    train_fn = theano.function(
        [input_var, target_var],
        [loss, train_acc, l1_penalty + l2_penalty],
        updates=updates,
    )
    prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_function = theano.function([input_var], prediction)

    # val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    print("Compiling functions " + str(time.time() - start_time) + " seconds")
    print("Network has parameter count of " + str(lasagne.layers.count_params(network)))
    # Compile a second function computing the validation loss and accuracy:

    # Finally, launch the training loop.
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    print("Starting training...")
    perform_validation(predict_function)

    # best_acc
    best_acc = 0
    best_loss = 1000
    early_stopping_count = 0
    iteration = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        if using_segmented:
            seg_indicator = "seg"
        else:
            seg_indicator = ""

        # shuffle order of training clips
        rgbd_clips = []
        # for root, dirs, files in os.walk(train_dir):
        #    for file in files:
        # if '_' + seg_indicator + 'rgbd_' in file :
        #        rgbd_clips.append(os.path.join(root, file).replace('\\', '/'))
        # shuffle(rgbd_clips)

        if os.path.isfile("epoch.savepoint"):
            fSavepoint = open("epoch.savepoint", "r")
            for line in fSavepoint:
                start_index = int(line.strip())
            fSavepoint.close()
            fEpochList = open("epoch.list", "r")
            for line in fEpochList:
                rgbd_clips.append(line.strip())
            fEpochList.close()
        else:
            start_index = 0
            fEpochList = open("epoch.list", "w")
            for root, dirs, files in os.walk(train_dir):
                for file in files:
                    # if '_' + seg_indicator + 'rgbd_' in file :
                    sample = os.path.join(root, file).replace("\\", "/")
                    fEpochList.write(sample + "\n")
                    rgbd_clips.append(sample)
            shuffle(rgbd_clips)
            fEpochList.close()

        train_err = 0
        train_acc = 0
        train_penalty = 0
        train_batches = 0

        # In each sub-epoch, we do a pass with 90 training clips of the training data:
        for k in range(start_index, len(rgbd_clips), clips_per_subepoch):
            early_stopping_count = early_stopping_count + 1
            if k + clips_per_subepoch > len(rgbd_clips):
                last_index = len(rgbd_clips)
            else:
                last_index = k + clips_per_subepoch

            train_clips = rgbd_clips[k:last_index]

            start_time = time.time()
            # for batch in iterate_minibatches(trainsetInput, trainsetLabel, minibatch, shuffle=True):
            for batch in iterate_minibatches_with_augmentation(train_clips):
                inputs, targets = batch
                err, acc, penalty = train_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_penalty += penalty
                train_batches += 1

                ITER_NUMBER.set_value(ITER_NUMBER.get_value() + 1)

            # Validation
            iteration = iteration + 1
            if iteration % 10 == 0:
                acc = perform_validation(predict_function)
                if acc > best_acc:
                    print("Saving weights of best model so far")
                    numpy.savez(
                        "network/current_network.npz",
                        *lasagne.layers.get_all_param_values(network)
                    )
                    best_acc = acc
                else:
                    LR.set_value(LR.get_value() / 10)
                    print("Annealing learning rate...")
            elif iteration % 1 == 0:
                perform_validation(predict_function, n_samples=1000)

            # Best model saving
            numpy.savez(
                "network/last_network.npz",
                *lasagne.layers.get_all_param_values(network)
            )

            # Then we print the results for this sub epoch:
            currentDT = datetime.datetime.now()
            print(str(currentDT))
            print(
                "Epoch {} of {} took {:.3f}s".format(
                    round(epoch + (last_index / len(rgbd_clips)), 2),
                    num_epochs,
                    time.time() - start_time,
                )
            )  #
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  training penalty:\t\t{:.6f}".format(train_penalty / train_batches))
            # print("  training penalty:\t\t{:.6f}".format(LR.get_value())
            print("  training clr:  \t\t{:.8f}".format(LR.get_value()))
            # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print(
                "  training accuracy:\t\t{:.3f} %".format(
                    train_acc / train_batches * 100
                )
            )


            # Epoch savepoint
            fSavepoint = open("epoch.savepoint", "w")
            fSavepoint.write(str(last_index))
            fSavepoint.close()

            if (1.0 * train_err / train_batches) < convergenceTrainLoss:
                print("Stopping training. Network already converged.")
                break


        # Declare convergence
        if (1.0 * train_err / train_batches) < convergenceTrainLoss:
            break
        if round(100.0 * train_err / train_batches, 2) == 100.00:
            break
        # Early Stopping
        # if early_stopping_count >= 15  :
        #    break

        # Delete epoch savepoint files
        os.remove("epoch.savepoint")
        os.remove("epoch.list")

    print("Training complete")


#### Start here
dataX = []
dataY = []
train(
    trainsetInput=dataX,
    trainsetLabel=dataY,
    testsetInput=dataX,
    testsetLabel=dataY,
    minibatch=40,
    num_epochs=100,
)
