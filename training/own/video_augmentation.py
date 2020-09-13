from skimage import transform as st
from skimage import util as su
from tensorflow import image as ti

import cv2
import numpy
import random
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


def augment_video(
    rgb_clip,
    depth_clip,
    img_rows,
    img_cols,
    frame_length,
    sequence_segmentation_only=True,
):

    assert len(rgb_clip) == len(depth_clip)
    print(len(rgb_clip))

    if sequence_segmentation_only:
        print("16 class detected")

    augmented_rgb_clips = []
    augmented_depth_clips = []

    # sequence segmentation. extract each combination of consecutive frames
    for i in range(0, len(rgb_clip) - frame_length + 1):
        rep_rgb_clip = []
        rep_depth_clip = []
        for j in range(i, i + frame_length):
            rep_rgb_clip.append(rgb_clip[j])
            rep_depth_clip.append(depth_clip[j])
        augmented_rgb_clips.append(numpy.array(rep_rgb_clip))
        augmented_depth_clips.append(numpy.array(rep_depth_clip))

    # scaling augmentation
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

            # scale by 90%
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

        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale12))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale12))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale11))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale11))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale09))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale09))
        add_rgb_clips.append(numpy.array(rep_rgb_clip_scale08))
        add_depth_clips.append(numpy.array(rep_depth_clip_scale08))
    augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
    augmented_depth_clips = augmented_depth_clips + add_depth_clips

    # return numpy.array(augmented_rgb_clips), numpy.array(augmented_depth_clips)
    ###################################################################################

    # brightness augmentation. apply to rgb clips only
    max_brightness_factor = 0.2
    add_rgb_clips = []
    add_depth_clips = []
    for i in range(0, len(augmented_rgb_clips)):
        brightness_factor = random.uniform(
            max_brightness_factor * -1, max_brightness_factor
        )
        rep_rgb_clip = []
        for frame in augmented_rgb_clips[i]:
            with tf.Session() as sess:
                processed_frame = ti.adjust_brightness(frame, brightness_factor)
                rep_rgb_clip.append(processed_frame.eval(session=sess))
            tf.reset_default_graph()
        add_rgb_clips.append(numpy.array(rep_rgb_clip))
        add_depth_clips.append(augmented_depth_clips[i])
    augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
    augmented_depth_clips = augmented_depth_clips + add_depth_clips

    # contrast augmentation. apply to rgb clips only
    min_contrast_factor = 0.5
    max_contrast_factor = 2.0
    add_rgb_clips = []
    add_depth_clips = []
    for i in range(0, len(augmented_rgb_clips)):
        contrast_factor = random.uniform(min_contrast_factor, max_contrast_factor)
        rep_rgb_clip = []
        for frame in augmented_rgb_clips[i]:
            with tf.Session() as sess:
                processed_frame = ti.adjust_contrast(frame, contrast_factor)
                rep_rgb_clip.append(processed_frame.eval(session=sess))
            tf.reset_default_graph()
        add_rgb_clips.append(numpy.array(rep_rgb_clip))
        add_depth_clips.append(augmented_depth_clips[i])
    augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
    augmented_depth_clips = augmented_depth_clips + add_depth_clips

    # noise augmentation. apply to rgb clips only
    add_rgb_clips = []
    add_depth_clips = []
    for i in range(0, len(augmented_rgb_clips)):
        rep_rgb_clip = []
        for frame in augmented_rgb_clips[i]:
            rep_rgb_clip.append((su.random_noise(frame) * 255).astype(numpy.uint8))
            # print(su.random_noise(frame))
        add_rgb_clips.append(numpy.array(rep_rgb_clip))
        add_depth_clips.append(augmented_depth_clips[i])
    augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
    augmented_depth_clips = augmented_depth_clips + add_depth_clips

    return numpy.array(augmented_rgb_clips), numpy.array(augmented_depth_clips)
    ###################################################################################

    if not sequence_segmentation_only:
        # translational shift augmentation. up to 1/10 th pixel dimensions only
        add_rgb_clips = []
        add_depth_clips = []
        for i in range(0, len(augmented_rgb_clips)):
            pad_left = random.randint(1, int(img_cols / 10))
            pad_right = random.randint(1, int(img_cols / 10))
            pad_top = random.randint(1, int(img_rows / 10))
            pad_bottom = random.randint(1, int(img_rows / 10))

            upward_rgb_clip = []
            upward_depth_clip = []
            downward_rgb_clip = []
            downward_depth_clip = []
            rightward_rgb_clip = []
            rightward_depth_clip = []
            leftward_rgb_clip = []
            leftward_depth_clip = []

            for j in range(0, len(augmented_rgb_clips[i])):
                with tf.Session() as sess:
                    padded_frame_rgb = ti.pad_to_bounding_box(
                        augmented_rgb_clips[i][j],
                        pad_top,
                        pad_left,
                        img_rows + pad_bottom + pad_top,
                        img_cols + pad_right + pad_left,
                    )
                    padded_frame_depth = ti.pad_to_bounding_box(
                        cv2.cvtColor(augmented_depth_clips[i][j], cv2.COLOR_GRAY2BGR),
                        pad_top,
                        pad_left,
                        img_rows + pad_bottom + pad_top,
                        img_cols + pad_right + pad_left,
                    )

                    # shift upward
                    processed_frame_rgb = ti.crop_to_bounding_box(
                        padded_frame_rgb,
                        pad_top + pad_bottom,
                        pad_left,
                        img_rows,
                        img_cols,
                    )
                    processed_frame_depth = ti.crop_to_bounding_box(
                        padded_frame_depth,
                        pad_top + pad_bottom,
                        pad_left,
                        img_rows,
                        img_cols,
                    )
                    upward_rgb_clip.append(processed_frame_rgb.eval(session=sess))
                    depth_frame_gray = cv2.cvtColor(
                        processed_frame_depth.eval(session=sess), cv2.COLOR_BGR2GRAY
                    )
                    # depth frame pad with background pixel
                    depth_frame_gray[
                        img_rows - pad_bottom - 1 : img_rows, :
                    ] = numpy.max(depth_frame_gray)
                    upward_depth_clip.append(depth_frame_gray)

                    # shift downward
                    processed_frame_rgb = ti.crop_to_bounding_box(
                        padded_frame_rgb, 0, pad_left, img_rows, img_cols
                    )
                    processed_frame_depth = ti.crop_to_bounding_box(
                        padded_frame_depth, 0, pad_left, img_rows, img_cols
                    )
                    downward_rgb_clip.append(processed_frame_rgb.eval(session=sess))
                    depth_frame_gray = cv2.cvtColor(
                        processed_frame_depth.eval(session=sess), cv2.COLOR_BGR2GRAY
                    )
                    # depth frame pad with background pixel
                    depth_frame_gray[0:pad_top, :] = numpy.max(depth_frame_gray)
                    downward_depth_clip.append(depth_frame_gray)

                    # shift right
                    processed_frame_rgb = ti.crop_to_bounding_box(
                        padded_frame_rgb, pad_top, 0, img_rows, img_cols
                    )
                    processed_frame_depth = ti.crop_to_bounding_box(
                        padded_frame_depth, pad_top, 0, img_rows, img_cols
                    )
                    rightward_rgb_clip.append(processed_frame_rgb.eval(session=sess))
                    depth_frame_gray = cv2.cvtColor(
                        processed_frame_depth.eval(session=sess), cv2.COLOR_BGR2GRAY
                    )
                    # depth frame pad with background pixel
                    depth_frame_gray[:, 0:pad_left] = numpy.max(depth_frame_gray)
                    rightward_depth_clip.append(depth_frame_gray)

                    # shift left
                    processed_frame_rgb = ti.crop_to_bounding_box(
                        padded_frame_rgb,
                        pad_top,
                        pad_left + pad_right,
                        img_rows,
                        img_cols,
                    )
                    processed_frame_depth = ti.crop_to_bounding_box(
                        padded_frame_depth,
                        pad_top,
                        pad_left + pad_right,
                        img_rows,
                        img_cols,
                    )
                    leftward_rgb_clip.append(processed_frame_rgb.eval(session=sess))
                    depth_frame_gray = cv2.cvtColor(
                        processed_frame_depth.eval(session=sess), cv2.COLOR_BGR2GRAY
                    )
                    # depth frame pad with background pixel
                    depth_frame_gray[
                        :, img_cols - pad_right - 1 : img_cols
                    ] = numpy.max(depth_frame_gray)
                    leftward_depth_clip.append(depth_frame_gray)
                tf.reset_default_graph()

            add_rgb_clips.append(numpy.array(upward_rgb_clip))
            add_depth_clips.append(numpy.array(upward_depth_clip))
            add_rgb_clips.append(numpy.array(downward_rgb_clip))
            add_depth_clips.append(numpy.array(downward_depth_clip))
            add_rgb_clips.append(numpy.array(rightward_rgb_clip))
            add_depth_clips.append(numpy.array(rightward_depth_clip))
            add_rgb_clips.append(numpy.array(leftward_rgb_clip))
            add_depth_clips.append(numpy.array(leftward_depth_clip))

        augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
        augmented_depth_clips = augmented_depth_clips + add_depth_clips

    # noise augmentation. apply to rgb clips only
    add_rgb_clips = []
    add_depth_clips = []
    for i in range(0, len(augmented_rgb_clips)):
        rep_rgb_clip = []
        for frame in augmented_rgb_clips[i]:
            rep_rgb_clip.append((su.random_noise(frame) * 255).astype(numpy.uint8))
            # print(su.random_noise(frame))
        add_rgb_clips.append(numpy.array(rep_rgb_clip))
        add_depth_clips.append(augmented_depth_clips[i])
    augmented_rgb_clips = augmented_rgb_clips + add_rgb_clips
    augmented_depth_clips = augmented_depth_clips + add_depth_clips

    """    
    print('RGB')
    print(numpy.array(augmented_rgb_clips).shape)
    print('depth')
    print(numpy.array(augmented_depth_clips).shape)
    """

    return numpy.array(augmented_rgb_clips), numpy.array(augmented_depth_clips)


img_rows = 96
img_cols = 96
rgbDir = "train/users/"
outputDir = "train/augmented.all.96/"
frameLength = 16

existingFiles = os.listdir(outputDir)

# for debugging
for root, dirs, files in os.walk(rgbDir):
    for file in files:
        if "_rgb_" in file:
            gestureClip = os.path.join(root, file).replace("\\", "/")
            print(gestureClip)

            fileExists = False
            for file in existingFiles:
                if os.path.basename(gestureClip).replace("rgb", "rgbd") in file:
                    fileExists = True
                    break
            if fileExists:
                print(
                    outputDir
                    + os.path.basename(gestureClip).replace("rgb", "rgbd")
                    + "already processed."
                )
                continue


            # print(gestureClip)
            frames = []
            cap = cv2.VideoCapture(gestureClip)

            # Get all frames from video clip
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.resize(frame, (img_rows, img_cols))
                    frames.append(frame)
                else:
                    break
            cap.release()

            # Get all frames from corresponding depth video clip
            depthClip = gestureClip.replace("_rgb_", "_depth_")
            print(depthClip)
            depthFrames = []
            cap = cv2.VideoCapture(depthClip)

            # Get all frames from depth video clip
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.resize(frame, (img_rows, img_cols))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    depthFrames.append(frame)
                else:
                    break
            cap.release()

            tagClass = int(os.path.basename(gestureClip.split("_")[3]))
            # (tagClass==16)) #

            if tagClass == 16:
                partition_length = int(len(frames) / 24)
                for i in range(0, len(frames), partition_length):
                    if i > 0:
                        j = i - 15
                    else:
                        j = 0
                    rgb, depth = augment_video(
                        frames[j : i + partition_length],
                        depthFrames[j : i + partition_length],
                        img_rows,
                        img_cols,
                        frameLength,
                        True,
                    )
                    # numpy.savez(outputDir + os.path.basename(gestureClip).replace('rgb', 'rgbd') + '_' + str(i) + '_' + '.npz', rgb=rgb, depth=depth)

                    frame_length = 100
                    for k in range(0, len(rgb), frame_length):
                        if k + frame_length > len(rgb):
                            last_index = len(rgb)
                        else:
                            last_index = k + frame_length
                        numpy.savez(
                            outputDir
                            + os.path.basename(gestureClip).replace("rgb", "rgbd")
                            + "_"
                            + str(i)
                            + "_"
                            + str(k)
                            + ".npz",
                            rgb=rgb[k:last_index],
                            depth=depth[k:last_index],
                        )

            else:
                rgb, depth = augment_video(
                    frames, depthFrames, img_rows, img_cols, frameLength, False
                )

                frame_length = 40
                print(len(rgb))
                for k in range(0, len(rgb), frame_length):
                    if k + frame_length > len(rgb):
                        last_index = len(rgb)
                    else:
                        last_index = k + frame_length
                    print(k)
                    numpy.savez(
                        outputDir
                        + os.path.basename(gestureClip).replace("rgb", "rgbd")
                        + "_"
                        + str(k)
                        + ".npz",
                        rgb=rgb[k:last_index],
                        depth=depth[k:last_index],
                    )

                # numpy.savez(outputDir + os.path.basename(gestureClip).replace('rgb', 'rgbd') + '.npz', rgb=rgb, depth=depth)
            tf.reset_default_graph()


