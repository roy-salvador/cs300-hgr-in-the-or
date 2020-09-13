import lasagne
from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import softmax, linear, rectify
import lasagne.nonlinearities
from lasagne.layers.shape import PadLayer
import theano
import theano.tensor as T


import os
import cv2
import numpy
import time
import pickle
import _pickle as cPickle
import math




class DemoDNN:


    def __init__(self):
        '''
        Initializes the netwrok
        '''
    
        self.img_rows = 96
        self.img_cols = 96
        self.channels = 4
        self.depth = 16
        self.num_classes = 17
        self.jester_num_classes = 27
        #self.weights_path = "network/raw_after_pink2_epoch62.npz" 
        self.weights_path = "network/jester.npz" 
        self.jester_action_classes=['Swiping Left', 'Swiping Right',  'Swiping Down', 'Swiping Up',
                                    'Pushing Hand Away', 'Pulling Hand In',
                                    'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 'Sliding Two Fingers Down',  'Sliding Two Fingers Up',
                                    'Pushing Two Fingers Away', 'Pulling Two Fingers In', 
                                    'Rolling Hand Forward', 'Rolling Hand Backward',
                                    'Turning Hand Clockwise', 'Turning Hand Counterclockwise', 
                                    'Zooming In With Full Hand', 'Zooming Out With Full Hand', 'Zooming In With Two Fingers', 'Zooming Out With Two Fingers', 
                                    'Thumb Up', 'Thumb Down',  
                                    'Shaking Hand', 'Stop Sign', 'Drumming Fingers', 
                                    'No gesture', 'Doing other things' ]
        
        
        # Prepare Theano variables for inputs and targets
        tensor5 = T.TensorType('float32', (False,)*5)
        self.input_var = tensor5('inputs')

        # Build the network
        start_time = time.time()
        self.network = self.build_c3d_model(input_var=self.input_var)
        print ('Building the network took ' + str(time.time() - start_time) + ' seconds')

        # Compiling Prediction Function
        start_time = time.time()
        self.prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.prediction_function = theano.function([self.input_var], self.prediction)
        print ('Compiling prediction function took ' + str(time.time() - start_time) + ' seconds')
        

    def build_c3d_model(self, input_var=None):
        '''
        Builds C3D model
        Returns
        -------
        dict
            A dictionary containing the network layers, where the output layer is at key 'prob'
        '''
        net = {}
        #net['input'] = InputLayer((None, 3, 16, 112, 112), input_var=input_var)
        net['input'] = InputLayer((None, self.channels, self.depth, self.img_rows, self.img_cols), input_var=input_var) # with depth channel

        # ----------- 1st layer group ---------------
        net['conv1a'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False))
        net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

        # ------------- 2nd layer group --------------
        net['conv2a'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 3rd layer group --------------
        net['conv3a'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        net['conv3b'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        net['pool3']  = MaxPool3DDNNLayer(net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 4th layer group --------------
        net['conv4a'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        net['conv4b'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        net['pool4']  = MaxPool3DDNNLayer(net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 5th layer group --------------
        net['conv5a'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        net['conv5b'] = lasagne.layers.batch_norm(Conv3DDNNLayer(net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify))
        # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
        net['pad']    = PadLayer(net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
        net['pool5']  = MaxPool3DDNNLayer(net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
        net['fc6-1']  = DenseLayer(net['pool5'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
        net['fc6_dropout'] = DropoutLayer(net['fc6-1'], p=0.1)
        net['fc7-1']  = DenseLayer(net['fc6_dropout'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
        net['fc7_dropout'] = DropoutLayer(net['fc7-1'], p=0.1)
        
        net['fc8-1']  = DenseLayer(net['fc7_dropout'], num_units=self.jester_num_classes, nonlinearity=None)
        net['prob']  = NonlinearityLayer(net['fc8-1'], softmax) 
        
        # Load pretrained model
        print ('Loading the trained c3d network')
        with numpy.load(self.weights_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net['prob'], param_values)

        return net['prob']
        
    
    def predict(self, representative_frames) :
        '''
        Predicts given a sample - a list of RGBD frames
        Returns
        -------
        numpy array
            A numpy array containing the network prediction
        '''
        sample = []
        sample.append(numpy.rollaxis( numpy.array(representative_frames), 3, 0)/numpy.float32(256))
        sample = numpy.array(sample)
        jester_result = self.prediction_function(sample) 
       
        return jester_result
        