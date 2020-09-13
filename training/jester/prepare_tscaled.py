import os
import cv2
import math
import numpy
import random
import tensorflow as tf
from tensorflow import image as ti
import time
from skimage import transform as st

img_rows = 96
img_cols = 96

# Random Contrast
def random_contrast(frames, min_contrast_factor=0.5, max_contrast_factor=2.0):
    contrast_factor = random.uniform(min_contrast_factor, max_contrast_factor)

    adjusted_frames = []
    for frame in frames:
        with tf.Session() as sess:
            processed_frame = ti.adjust_contrast(frame, contrast_factor)
            adjusted_frames.append(processed_frame.eval(session=sess))
        tf.reset_default_graph()
    return numpy.array(adjusted_frames)


# Random Brightness
def random_brightness(frames, max_brightness_factor=0.2):
    brightness_factor = random.uniform(
        max_brightness_factor * -1, max_brightness_factor
    )

    adjusted_frames = []
    for frame in frames:
        with tf.Session() as sess:
            processed_frame = ti.adjust_brightness(frame, brightness_factor)
            adjusted_frames.append(processed_frame.eval(session=sess))
        tf.reset_default_graph()
    return numpy.array(adjusted_frames)


# Randomspatial scaling
def random_spatial_scaling(frames, min_scale=0.4, max_scale=1.2):
    scale_factor = random.uniform(min_scale, max_scale)
    # scale_rgb = (st.rescale(frames[0], scale=scale_out_factor, mode='constant')* 255).astype(numpy.uint8)

    adjusted_frames = []
    for i, frame in enumerate(frames):
        scale_out_rgb = (
            st.rescale(frame, scale=scale_factor, mode="constant") * 255
        ).astype(numpy.uint8)
        if i == 0:
            start_row = int(
                random.uniform(0, abs(scale_out_rgb.shape[0] - frame.shape[0]))
            )
            start_col = int(
                random.uniform(0, abs(scale_out_rgb.shape[1] - frame.shape[1]))
            )

        if scale_factor >= 1.0:
            adjusted_frames.append(
                scale_out_rgb[
                    start_row : start_row + frame.shape[0],
                    start_col : start_col + frame.shape[1],
                ]
            )
        else:
            padded_rgb = numpy.zeros((frame.shape[0], frame.shape[1], 3), numpy.uint8)
            padded_rgb[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_rgb
            adjusted_frames.append(padded_rgb)
    return numpy.array(adjusted_frames)


# Time scaling
def get_representative_frames(frames, clipDepth=16, time_scale=1.0):
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
    rgb_frames = []
    for i in range(0, clipDepth):
        # print(scaled_frames[int(math.floor(frame_interval*i))])
        img = cv2.imread(scaled_frames[int(math.floor(frame_interval * i))])
        img = cv2.resize(img, (img_rows, img_cols))
        rgb_frames.append(img)
    return numpy.array(rgb_frames)


# Random Clipping
def get_random_frames(frames, clipDepth=16):
    # print(frames)
    unchosen = int(clipDepth / len(frames))
    marker = numpy.zeros(len(frames))
    marker[:] = unchosen

    while sum(marker) < clipDepth:
        index = random.randint(0, len(marker) - 1)
        while marker[index] != unchosen:
            index = (index + 1) % len(marker)
        marker[index] = marker[index] + 1
    # print(marker)

    rgb_frames = []
    for i in range(0, len(marker)):
        while marker[i] > 0:
            # print(frames[i])
            img = cv2.imread(frames[i])
            img = cv2.resize(img, (img_rows, img_cols))
            rgb_frames.append(img)
            marker[i] = marker[i] - 1
    return numpy.array(rgb_frames)


trainFile = open("jester-v1-train.csv", "r")
labelsFile = open("jester-v1-labels.csv", "r")
labels = []
for line in labelsFile:
    labels.append(line.strip())

file_list = []
for line in trainFile:
    file_list.append(line)
random.shuffle(file_list)
# print(file_list)

for j, line in enumerate(file_list):
    tokens = line.strip().split(";")
    sample = tokens[0]
    tag = labels.index(tokens[1])
    # print(sample)
    # sampleFile = os.path.join('train.scaled.jester', sample + '_' + str(tag) + '.npz')
    if os.path.isfile(
        os.path.join("train.scaled.jester", sample + "_" + str(tag) + ".tscaled.npz")
    ):
        file_mod_time = os.stat(
            os.path.join(
                "train.scaled.jester", sample + "_" + str(tag) + ".tscaled.npz"
            )
        ).st_mtime
        if (
            time.time() - file_mod_time < 86400 * 3
        ):  # False :#os.path.isfile(noisy_file) :
            # print(str(j) + ' ' + sample + ' already processed')
            continue

    frames = []
    for image in sorted(os.listdir(os.path.join("20bn-jester-v1", sample))):
        frames.append(os.path.join("20bn-jester-v1", sample, image))

    rgb = []
    for time_scale in [
        random.uniform(0.7, 0.8),
        random.uniform(0.8, 0.9),
        random.uniform(0.9, 1.0),
        random.uniform(1.0, 1.1),
        random.uniform(1.1, 1.2),
        random.uniform(1.2, 1.3),
    ]:

        rgb.append(
            random_contrast(
                random_brightness(
                    random_spatial_scaling(
                        get_representative_frames(frames, time_scale=time_scale)
                    )
                )
            )
        )
    rgb.append(
        random_contrast(
            random_brightness(random_spatial_scaling(get_random_frames(frames)))
        )
    )
    rgb = numpy.array(rgb)
    # print(rgb.shape)i
    print(
        str(j)
        + " Created "
        + os.path.join("train.scaled.jester", sample + "_" + str(tag) + ".tscaled.npz")
    )
    numpy.savez(
        os.path.join("train.scaled.jester", sample + "_" + str(tag) + ".tscaled.npz"),
        rgb=rgb,
    )

    # break

trainFile.close()
