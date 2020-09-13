import os
import numpy
from skimage import util as su
import time
import random

TRAIN_DIR = "train.scaled.jester"
suffix = ".noised"

# Recreate noised files for those samples seen by the network within the last epoch
fSavepoint = open("epoch.savepoint", "r")
for line in fSavepoint:
    last_index = int(line.strip())
fSavepoint.close()


i = 0
files = []
fList = open("epoch.list", "r")
for line in fList:
    files.append(line)
fList.close()
random.shuffle(files)

for line in files:
    file = line.strip()
    # for file in os.listdir(TRAIN_DIR) :
    if suffix not in file:

        noisy_file = os.path.join(file + suffix + ".npz")
        if os.path.isfile(noisy_file):
            file_mod_time = os.stat(noisy_file).st_mtime
        else:
            file_mod_time = time.time() - 864000

        if (
            time.time() - file_mod_time < 86400 * 3
        ):  # False :#os.path.isfile(noisy_file) :
            None
        else:
            # Perform noise augmentation
            with numpy.load(os.path.join(file)) as data:
                rgb_clips = data["rgb"]
                # depth_clips = data['depth']

            noised_rgb_clips = []
            # noised_depth_clips = []
            for i in range(0, len(rgb_clips)):
                rep_rgb_clip = []
                for frame in rgb_clips[i]:
                    rep_rgb_clip.append(
                        (su.random_noise(frame) * 255).astype(numpy.uint8)
                    )
                noised_rgb_clips.append(numpy.array(rep_rgb_clip))
                # noised_depth_clips.append(depth_clips[i])

            rgb = numpy.array(noised_rgb_clips)
            # depth = numpy.array(noised_depth_clips)

            numpy.savez(noisy_file, rgb=rgb)  # , depth=depth)

            print("Recreated noisy sample for " + file + " " + noisy_file)
        # else :
        # 	print('Already exists ' +  noisy_file)
    if i >= last_index:
        break
