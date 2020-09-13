# Instructions
1. Download our Dynamic Hand Gesture Dataset
2. Place training video clip samples under `train/users` directory, place test video clips under `test/users`
3. Run `video_augmentation.py` to generate augmented network samples on disk
4. Start training the network by running `train_jester.py`. Weights of the most recent and best performing iteration of the network are placed in `network/last_network.npz` and `network/current_network.npz` respectively.
5. If you want to continue training, initialize the weights with your network by uncommenting and indicating the network's filename in [lines 196-200](https://github.com/roy-salvador/cs300-hgr-in-the-or/blob/master/training/own/train_demo_network.py#L196) of `train_demo_network.py`
6. Performance is output at the end of the training. Alternatively you can just load your network then invoke performance measuring just like in [lines 699-700](https://github.com/roy-salvador/cs300-hgr-in-the-or/blob/master/training/own/train_demo_network.py#L699) in `train_demo_network.py`
