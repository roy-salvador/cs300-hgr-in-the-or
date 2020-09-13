# Instructions
1. Download and extract the Sheffield Kinect Gesture (SKIG) dataset. 
2. Create `data/train_rgb`, `data/train_d`, `data/test_rgb`, `data/test_d` directories
3. Place the video clip samples in appropriate directories
	* `data/train_rgb` - rgb clips in the training set
	* `data/train_d` - depth clips in the training set
	* `data/test_rgb` - depth clips in the test set
	* `data/test_d` - depth clips in the test set
4. Run train_skig.py to start training
5. Run evaluate_skig.py to measure performance of the trained network on the test set
