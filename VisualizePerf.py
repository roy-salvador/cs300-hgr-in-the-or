from datetime import datetime
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os
import base64
import collections
import DemoDNN
import DemoDisplay
import cv2
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    action="store",
    dest="video_file",
    required=True,
    help="Video recording file of your gesture sequence",
)
parser.add_argument(
    "-a",
    action="store",
    dest="annotation_file",
    required=True,
    help="Annotation file of the video recording",
)
parser.add_argument(
    "-p",
    action="store",
    dest="plot_colors_file",
    required=False,
    default="plot_colors.config",
    help="Configuration file for plot colors",
)
parsed_args = parser.parse_args()


def convert_to_datetime(x):
    """
  Converts the frame number to datetime
  """
    return datetime.fromtimestamp(31536000 + x * 24 * 3600).strftime("%Y-%d-%m")


def convert_to_timetick(x, fps):
    """
  Converts the frame number to a time tick label string
  """
    seconds = int(x / fps) % 60
    hours = int(x / fps / 60)
    return str(hours).zfill(1) + ":" + str(seconds).zfill(2)


# Read and initialize plot colors for each gesture
colors = {}
not_used_in_system = [
    "Doing other things",
    "No gesture",
    "Shaking Hand",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Drumming Fingers",
]
df_plot_colors = pd.read_csv(parsed_args.plot_colors_file)
for index, row in df_plot_colors.iterrows():
    print(row["gesture"])
    print(row["rgb_color"])
    colors[row["gesture"]] = row["rgb_color"]
print(colors)

# Read and process annotation
df_annot = pd.read_csv(parsed_args.annotation_file)
plot_objects = []
for index, row in df_annot.iterrows():
    plot_objects.append(
        dict(
            Task="Ground Truth",
            Start=pd.to_datetime(
                convert_to_datetime(row["frame_start"]),
                errors="coerce",
                format="%Y-%d-%m",
            ),
            Finish=pd.to_datetime(
                convert_to_datetime(row["frame_end"]),
                errors="coerce",
                format="%Y-%d-%m",
            ),
            Gesture=row["gesture"],
        )
    )

plot_objects.append(
    dict(
        Task="Prediction<br>Confidence<br>(1.0)",
        Start=pd.to_datetime(
            convert_to_datetime(0), errors="coerce", format="%Y-%d-%m"
        ),
        Finish=pd.to_datetime(
            convert_to_datetime(0), errors="coerce", format="%Y-%d-%m"
        ),
        Gesture="Others",
    )
)
plot_objects.append(
    dict(
        Task="Prediction<br>Confidence<br>(0.0)",
        Start=pd.to_datetime(
            convert_to_datetime(0), errors="coerce", format="%Y-%d-%m"
        ),
        Finish=pd.to_datetime(
            convert_to_datetime(0), errors="coerce", format="%Y-%d-%m"
        ),
        Gesture="Others",
    )
)

# Get prediction of network for video
cap_rgb = cv2.VideoCapture(parsed_args.video_file)
fps = cap_rgb.get(cv2.CAP_PROP_FPS)
frame_length = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))


# Plot labels
frame_interval = (frame_length - 1) / 12.0
frame_labels = []
time_tick_labels = []
i = 0
while i <= frame_length:
    frame_labels.append(int(i))
    time_tick_labels.append(convert_to_timetick(int(i), fps))
    i = i + frame_interval


date_ticks = [
    pd.to_datetime(convert_to_datetime(x), format="%Y-%d-%m") for x in frame_labels
]
df_plot = pd.DataFrame(plot_objects)
fig = ff.create_gantt(
    df_plot, colors=colors, group_tasks=True, index_col="Gesture", show_colorbar=True
)
fig.layout.xaxis.update(
    {"tickvals": date_ticks, "ticktext": time_tick_labels, "showgrid": True}
)
fig.layout.yaxis.update({"showgrid": True, "range": [0, 4]})


demo_dnn = DemoDNN.DemoDNN()
current_frames = []
network_predictions = []
plot_frames = []
print("Getting predictions of video clips")
i = 0
while True:
    # Get frame and zoom in
    ret, frame_rgb = cap_rgb.read()

    if ret == True:
        # Collect image for graph display
        if i in frame_labels[1:]:
            plot_frames.append(frame_rgb)

        # Create DNN input
        b_channel, g_channel, r_channel = cv2.split(
            cv2.resize(frame_rgb, (demo_dnn.img_rows, demo_dnn.img_cols))
        )
        d_channel = np.zeros((demo_dnn.img_rows, demo_dnn.img_cols), dtype=np.uint8)
        d_channel[:] = 255
        img_RGBD = cv2.merge((b_channel, g_channel, r_channel, d_channel))
        img_RGBD = cv2.flip(img_RGBD, 1)
        current_frames.append(img_RGBD)

        # Get DNN result and its corresponding display
        if len(current_frames) > demo_dnn.depth:
            current_frames.pop(0)
        if len(current_frames) == demo_dnn.depth:
            result = demo_dnn.predict(current_frames)[0]
            network_predictions.append(result)
        i = i + 1
    else:
        break
network_predictions = np.array(network_predictions)


# create trace on the graph for each gesture
new_data = []
x_dim = []
for i in range(0, network_predictions.shape[0]):
    x_dim.append(
        pd.to_datetime(
            convert_to_datetime(i + demo_dnn.depth), errors="coerce", format="%Y-%d-%m"
        )
    )

for i, gesture in enumerate(demo_dnn.jester_action_classes):
    if gesture in not_used_in_system:
        continue
    if gesture in colors.keys():
        trace_color = colors[gesture]
    else:
        trace_color = colors["Others"]
    new_data.append(
        go.Scatter(
            x=x_dim,
            y=network_predictions[:, i],
            mode="lines",
            line=dict(color=trace_color, width=1),
            showlegend=False,
        )
    )

new_data.extend(list(fig.data))
new_fig = go.Figure(data=new_data, layout=fig.layout)


# measure the performance
demo_display = DemoDisplay.DemoDisplay()
GESTURE_CLASSES = demo_display.action_classes
GESTURE_COUNT = np.zeros([len(GESTURE_CLASSES), len(GESTURE_CLASSES)])

ground_truth = np.zeros(network_predictions.shape[0])
ground_truth[:] = demo_dnn.jester_action_classes.index("No gesture")
for index, row in df_annot.iterrows():
    start = int(row["frame_start"]) - demo_dnn.depth
    end = int(row["frame_end"]) - demo_dnn.depth
    ground_truth[start : end + 1] = demo_dnn.jester_action_classes.index(row["gesture"])

for i in range(0, network_predictions.shape[0]):

    network_prediction = np.argmax(
        demo_display.convert_to_system_action_prediction(
            network_predictions[i], demo_dnn.jester_action_classes
        )
    )
    one_hot_tag = np.zeros(len(demo_dnn.jester_action_classes))
    one_hot_tag[int(ground_truth[i])] = 1
    tag = np.argmax(
        demo_display.convert_to_system_action_prediction(
            one_hot_tag, demo_dnn.jester_action_classes
        )
    )
    GESTURE_COUNT[network_prediction][tag] = GESTURE_COUNT[network_prediction][tag] + 1


# System Action Accuracy
TOTAL_PER_CLASS = sum(GESTURE_COUNT)
TOTAL_CORRECT_SA = 0
i = 0
while i < len(GESTURE_CLASSES) - 1:
    print(
        GESTURE_CLASSES[i]
        + " = "
        + str(GESTURE_COUNT[i][i])
        + "/"
        + str(TOTAL_PER_CLASS[i])
        + "("
        + str(round(GESTURE_COUNT[i][i] * 100.0 / TOTAL_PER_CLASS[i], 2))
        + "%)"
    )
    TOTAL_CORRECT_SA = TOTAL_CORRECT_SA + GESTURE_COUNT[i][i]
    i = i + 1
print("")
print(
    "System Action Accuracy = "
    + str(TOTAL_CORRECT_SA)
    + "/"
    + str(sum(TOTAL_PER_CLASS[: len(GESTURE_CLASSES) - 1]))
    + "("
    + str(
        round(
            TOTAL_CORRECT_SA * 100.0 / sum(TOTAL_PER_CLASS[: len(GESTURE_CLASSES) - 1]),
            2,
        )
    )
    + "%)"
)

# Non System Action Performance/
TP = GESTURE_COUNT[len(GESTURE_CLASSES) - 1][len(GESTURE_CLASSES) - 1]
print(
    "Non System Action Precision = "
    + str(TP)
    + "/"
    + str(sum(GESTURE_COUNT[len(GESTURE_CLASSES) - 1]))
    + "("
    + str(round(TP * 100.0 / sum(GESTURE_COUNT[len(GESTURE_CLASSES) - 1]), 2))
    + "%)"
)
print(
    "Non System Action Recall = "
    + str(TP)
    + "/"
    + str(sum(GESTURE_COUNT[:, len(GESTURE_CLASSES) - 1]))
    + "("
    + str(round(TP * 100.0 / sum(GESTURE_COUNT[:, len(GESTURE_CLASSES) - 1]), 2))
    + "%)"
)

# embed video images
locx = 0.053
sizex = 0.079
for img in plot_frames:
    retval, buffer = cv2.imencode(".png", img)
    encoded_image = "data:image/png;base64," + base64.b64encode(buffer).decode()

    new_fig.add_layout_image(
        dict(
            source=encoded_image,
            xref="paper",
            yref="y",
            sizing="stretch",
            opacity=1.0,
            sizey=1,
            sizex=sizex,
            x=locx,
            y=3.5,
            layer="above",
        )
    )
    locx = locx + sizex

# adjust layout
new_fig.update_layout(
    template="plotly_white",
    legend=dict(
        orientation="h",
        traceorder="normal",
        yanchor="bottom",
        y=-0.25,
        xanchor="right",
        x=1,
    ),
)
new_fig.show()
