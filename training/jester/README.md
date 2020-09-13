# Instructions
1. Download and extract Jester Dataset
2. Prepare the network samples with space-scaled augmentation (scale in and out) to disk by running `prepare_jester.py`
3. Perform time-scaled, random brightness and contrast data augmentation using `prepare_tscaled.py`
4. Create noised version of the samples for further augmentation with `create_noised_samples.py`
5. Start training the network by running `train_jester.py`. Weights of the most recent and best performing iteration of the network are placed in `network/last_network.npz` and `network/current_network.npz` respectively
6. If you want to continue training, initialize the weights with your network by uncommenting and indicating the network's filename in [lines 175-178](https://github.com/roy-salvador/cs300-hgr-in-the-or/blob/master/training/jester/train_jester.py#L175) of `train_jester.py`
7. Measure the performance of the network on the Jester validation set using `evaluate_jester.py` 
8. Measure the performance of the Jester Mapped System Actions on the Jester validation set using `evaluate_jester_mapped.py`. Mapping of the gestures can be found [here](https://github.com/roy-salvador/cs300-hgr-in-the-or/blob/master/training/jester/evaluate_jester_mapped.py#L52). 
