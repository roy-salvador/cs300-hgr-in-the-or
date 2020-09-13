import cv2
import numpy as np
import DemoDNN
import DemoDisplay
import time
import datetime
import os
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-v', action='store', dest='video_file', required=True, help='Video recording File of your gesture sequence')
parsed_args = parser.parse_args()

# Buffer frames
current_frames = []

# Initialize Network and Application Display
demo_dnn = DemoDNN.DemoDNN()
demo_display = DemoDisplay.DemoDisplay()

# read video from file
cap_rgb = cv2.VideoCapture(parsed_args.video_file)

# set zoom level
zoom_level = 1
max_offset = 5
offset_x = 0
offset_y = 0

while True :

    print("=============================================================================================")
    
    # Get frame and zoom in
    ret, frame_rgb = cap_rgb.read()
    
    if ret == True :
        #frame_rgb = frame_rgb[25:310,960:]
        #print(frame_rgb.shape)
        row_len = int(frame_rgb.shape[0] * zoom_level)
        col_len = int(frame_rgb.shape[1] * zoom_level)
        row_out = (frame_rgb.shape[0] - row_len) / 2
        col_out = (frame_rgb.shape[1] - col_len) / 2
        row_start = int(row_out + (row_out / max_offset * offset_y ))
        col_start = int(col_out + (col_out / max_offset * offset_x ))
        frame_rgb = frame_rgb[row_start: row_start + row_len,
                              col_start : col_start + col_len,
                              :]

        
        # Create DNN input
        b_channel, g_channel, r_channel = cv2.split(cv2.resize(frame_rgb, (demo_dnn.img_rows, demo_dnn.img_cols)))
        d_channel = np.zeros((demo_dnn.img_rows, demo_dnn.img_cols),  dtype=np.uint8)
        d_channel[:] = 255
        img_RGBD = cv2.merge((b_channel, g_channel, r_channel, d_channel))
        img_RGBD = cv2.flip(img_RGBD, 1)
        current_frames.append(img_RGBD)
        
        
        
        # Get DNN result and its corresponding display
        if len(current_frames) > demo_dnn.depth :
            current_frames.pop(0)
        if len(current_frames) == demo_dnn.depth :   
            
            pipeline_start = datetime.datetime.now()
            result = demo_dnn.predict(current_frames)[0]
            #result = np.zeros(demo_dnn.jester_num_classes)
            pipeline_end = datetime.datetime.now()
            pipeline_processing_time = pipeline_end - pipeline_start
            print('Network prediction took ' + str(int(pipeline_processing_time.total_seconds() * 1000)) + ' milliseconds.')
            
            pipeline_start = datetime.datetime.now()
            action_panel, center_panel, navigation_panel = demo_display.get_gui_components(result, demo_dnn.jester_action_classes)
        else :
            pipeline_start = datetime.datetime.now()
            action_panel, center_panel, navigation_panel = demo_display.get_gui_components(np.zeros(demo_dnn.jester_num_classes), demo_dnn.jester_action_classes)

        
        # Render the application
        frame_rgb = cv2.resize(frame_rgb,(512, 424-56) )
        #frame_rgb = cv2.flip(frame_rgb, 1)
        right_panel = np.vstack([frame_rgb, action_panel])
        cv2.imshow('Medical Image Navigator Demo', np.hstack([navigation_panel, center_panel, right_panel]))
        pipeline_end = datetime.datetime.now()
        pipeline_processing_time = pipeline_end - pipeline_start
        print('Rendering application took ' + str(int(pipeline_processing_time.total_seconds() * 1000)) + ' milliseconds.')


        # if the 'q' key is pressed, stop the loop
        k = cv2.waitKey(5) 
        if k == ord('q') or k == ord('Q'):
            print('Quitting Program')
            break
        elif k == ord('0') :
            zoom_level = 1.0
        elif k == ord('9') :
            zoom_level = 1.0 / 2
        elif k == ord('8') :
            zoom_level = 1.0 / 3
        elif k == ord('7') :
            zoom_level = 1.0 / 4
        elif k == ord('6') :
            zoom_level = 1.0 / 5
        elif k == ord('5') :
            zoom_level = 1.0 / 6
        elif k == ord('4') :
            zoom_level = 1.0 / 7
        elif k == ord('3') :
            zoom_level = 1.0 / 8
        elif k == ord('2') :
            zoom_level = 1.0 / 9
        elif k == ord('1') :
            zoom_level = 1.0 / 10
        elif k == ord('I') or k == ord('i'): 
            offset_y = max( max_offset * -1, offset_y - 1)
        elif  k == ord('J') or k == ord('j'): 
            offset_x = min( max_offset, offset_x + 1)
        elif  k == ord('K') or k == ord('k'): 
            offset_y = min( max_offset, offset_y + 1)
        elif  k == ord('L') or k == ord('l'): 
            offset_x = max( max_offset * -1, offset_x - 1)
    else :
        break    
    
    # Clean up on window closing
    if cv2.getWindowProperty('Medical Image Navigator Demo', 0) < 0:
        print('Application Closed')
        cv2.destroyAllWindows()
        cap_rgb.release()
        os._exit(1)


cap_rgb.release()

        
      
        
    