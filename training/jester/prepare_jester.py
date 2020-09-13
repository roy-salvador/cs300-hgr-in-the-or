import os
import math
import cv2
import numpy
from skimage import transform as st

clipDepth = 16
img_rows = 96
img_cols = 96


def augment_video(
    rgb_clip, depth_clip, img_rows=img_rows, img_cols=img_cols, frame_length=clipDepth
):
    augmented_rgb_clips = []
    augmented_depth_clips = []

    augmented_rgb_clips.append(numpy.array(rgb_clip))
    augmented_depth_clips.append(numpy.array(depth_clip))

    add_rgb_clips = []
    add_depth_clips = []

    for i in range(0, len(augmented_rgb_clips)):
        rep_rgb_clip_scale12 = []
        rep_depth_clip_scale12 = []
        rep_rgb_clip_scale11 = []
        rep_depth_clip_scale11 = []
        rep_rgb_clip_scale09 = []
        rep_depth_clip_scale09 = []
        rep_rgb_clip_scale08 = []
        rep_depth_clip_scale08 = []
        rep_rgb_clip_scale07 = []
        rep_depth_clip_scale07 = []
        rep_rgb_clip_scale06 = []
        rep_depth_clip_scale06 = []

        for j in range(0, len(augmented_rgb_clips[i])):
            # scale by 120%
            scale_out_factor = 1.2
            scale_out_rgb = (
                st.rescale(
                    augmented_rgb_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            scale_out_depth = (
                st.rescale(
                    augmented_depth_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            start_row = int(
                (scale_out_rgb.shape[0] - augmented_rgb_clips[i][j].shape[0]) / 2
            )
            start_col = int(
                (scale_out_rgb.shape[1] - augmented_rgb_clips[i][j].shape[1]) / 2
            )
            rep_rgb_clip_scale12.append(
                scale_out_rgb[
                    start_row : start_row + augmented_rgb_clips[i][j].shape[0],
                    start_col : start_col + augmented_rgb_clips[i][j].shape[1],
                ]
            )
            rep_depth_clip_scale12.append(
                scale_out_depth[
                    start_row : start_row + augmented_rgb_clips[i][j].shape[0],
                    start_col : start_col + augmented_rgb_clips[i][j].shape[1],
                ]
            )

            # scale by 110%
            scale_out_factor = 1.1
            scale_out_rgb = (
                st.rescale(
                    augmented_rgb_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            scale_out_depth = (
                st.rescale(
                    augmented_depth_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            start_row = int(
                (scale_out_rgb.shape[0] - augmented_rgb_clips[i][j].shape[0]) / 2
            )
            start_col = int(
                (scale_out_rgb.shape[1] - augmented_rgb_clips[i][j].shape[1]) / 2
            )
            rep_rgb_clip_scale11.append(
                scale_out_rgb[
                    start_row : start_row + augmented_rgb_clips[i][j].shape[0],
                    start_col : start_col + augmented_rgb_clips[i][j].shape[1],
                ]
            )
            rep_depth_clip_scale11.append(
                scale_out_depth[
                    start_row : start_row + augmented_rgb_clips[i][j].shape[0],
                    start_col : start_col + augmented_rgb_clips[i][j].shape[1],
                ]
            )

            # scale by 90%
            scale_out_factor = 0.9
            scale_out_rgb = (
                st.rescale(
                    augmented_rgb_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            scale_out_depth = (
                st.rescale(
                    augmented_depth_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            start_row = int(
                (augmented_rgb_clips[i][j].shape[0] - scale_out_rgb.shape[0]) / 2
            )
            start_col = int(
                (augmented_rgb_clips[i][j].shape[1] - scale_out_rgb.shape[1]) / 2
            )
            padded_rgb = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                    3,
                ),
                numpy.uint8,
            )
            padded_rgb[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_rgb
            padded_depth = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                ),
                numpy.uint8,
            )
            padded_depth[:] = 255
            padded_depth[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_depth
            rep_rgb_clip_scale09.append(padded_rgb)
            rep_depth_clip_scale09.append(padded_depth)

            scale_out_factor = 0.8
            scale_out_rgb = (
                st.rescale(
                    augmented_rgb_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            scale_out_depth = (
                st.rescale(
                    augmented_depth_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            start_row = int(
                (augmented_rgb_clips[i][j].shape[0] - scale_out_rgb.shape[0]) / 2
            )
            start_col = int(
                (augmented_rgb_clips[i][j].shape[1] - scale_out_rgb.shape[1]) / 2
            )
            padded_rgb = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                    3,
                ),
                numpy.uint8,
            )
            padded_rgb[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_rgb
            padded_depth = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                ),
                numpy.uint8,
            )
            padded_depth[:] = 255
            padded_depth[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_depth
            rep_rgb_clip_scale08.append(padded_rgb)
            rep_depth_clip_scale08.append(padded_depth)

            scale_out_factor = 0.7
            scale_out_rgb = (
                st.rescale(
                    augmented_rgb_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            scale_out_depth = (
                st.rescale(
                    augmented_depth_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            start_row = int(
                (augmented_rgb_clips[i][j].shape[0] - scale_out_rgb.shape[0]) / 2
            )
            start_col = int(
                (augmented_rgb_clips[i][j].shape[1] - scale_out_rgb.shape[1]) / 2
            )
            padded_rgb = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                    3,
                ),
                numpy.uint8,
            )
            padded_rgb[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_rgb
            padded_depth = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                ),
                numpy.uint8,
            )
            padded_depth[:] = 255
            padded_depth[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_depth
            rep_rgb_clip_scale07.append(padded_rgb)
            rep_depth_clip_scale07.append(padded_depth)

            scale_out_factor = 0.6
            scale_out_rgb = (
                st.rescale(
                    augmented_rgb_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            scale_out_depth = (
                st.rescale(
                    augmented_depth_clips[i][j], scale=scale_out_factor, mode="constant"
                )
                * 255
            ).astype(numpy.uint8)
            start_row = int(
                (augmented_rgb_clips[i][j].shape[0] - scale_out_rgb.shape[0]) / 2
            )
            start_col = int(
                (augmented_rgb_clips[i][j].shape[1] - scale_out_rgb.shape[1]) / 2
            )
            padded_rgb = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                    3,
                ),
                numpy.uint8,
            )
            padded_rgb[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_rgb
            padded_depth = numpy.zeros(
                (
                    augmented_rgb_clips[i][j].shape[0],
                    augmented_rgb_clips[i][j].shape[1],
                ),
                numpy.uint8,
            )
            padded_depth[:] = 255
            padded_depth[
                start_row : start_row + scale_out_rgb.shape[0],
                start_col : start_col + scale_out_rgb.shape[1],
            ] = scale_out_depth
            rep_rgb_clip_scale06.append(padded_rgb)
            rep_depth_clip_scale06.append(padded_depth)

        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale12))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale12))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale11))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale11))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale09))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale09))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale08))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale08))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale07))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale07))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale06))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale06))

    augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
    augmented_depth_clips = augmented_depth_clips + add_depth_clips

    return numpy.array(augmented_rgb_clips), numpy.array(augmented_depth_clips)


trainFile = open("jester-v1-train.csv", "r")
labelsFile = open("jester-v1-labels.csv", "r")
labels = []
for line in labelsFile:
    labels.append(line.strip())


for line in trainFile:
    tokens = line.strip().split(";")
    sample = tokens[0]
    tag = labels.index(tokens[1])
    print(sample)
    sampleFile = os.path.join("train.scaled.jester", sample + "_" + str(tag) + ".npz")
    if os.path.isfile(sampleFile):
        if os.path.getsize(sampleFile) > 2949582:
            continue

    # print(tag)

    frames = []
    for image in sorted(os.listdir(os.path.join("20bn-jester-v1", sample))):
        frames.append(os.path.join("20bn-jester-v1", sample, image))

    # Get representative Frame
    frame_interval = 1.0 * len(frames) / clipDepth
    rgb_frames = []
    d_frames = []
    i = 0
    while i < clipDepth:
        img = cv2.imread(frames[int(math.floor(frame_interval * i))])
        img = cv2.resize(img, (img_rows, img_cols))
        rgb_frames.append(img)
        # b_channel, g_channel, r_channel = cv2.split(img)

        d_channel = numpy.zeros((img_rows, img_cols), numpy.uint8)
        d_channel[:] = 255

        d_frames.append(d_channel)
        # img_RGBD = cv2.merge((b_channel, g_channel, r_channel, d_channel))
        # representative_frames.append(img_RGBD)

        i = i + 1
    # print(numpy.array(rgb_frames).shape)
    rgb, depth = augment_video(rgb_frames, d_frames)
    numpy.savez(
        os.path.join("train.scaled.jester", sample + "_" + str(tag) + ".npz"),
        rgb=rgb,
        depth=depth,
    )
    # print(sample)
    # print(rgb.shape)
    # print(depth.shape)

    # break


labelsFile.close()
trainFile.close()

print(labels)
