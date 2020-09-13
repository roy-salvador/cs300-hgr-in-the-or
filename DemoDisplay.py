import numpy as np
import cv2
import imutils
import os

class DemoDisplay:

    
    def __init__(self):
        self.action_classes=['Previous Series', 'Next Series', 'Previous Image', 'Next Image', 
                'Browse Up', 'Browse Down', 'Browse Left', 'Browse Right',
                'Play In Reverse', 'Play Forward', 'Stop', 
                'Increase Brightness', 'Decrease Brightness',
                'Rotate Counter-clockwise', 'Rotate Clockwise',
                'Zoom In', 'Zoom Out', 
                'Thumb Up (Unlock)', 'Thumb Down (Lock)', 
                'No System Action' ]
        self.action_labels=[]
        
        
        # default settings
        self.mode = 'view'
        self.current_series = 0
        self.current_image = 0
        self.current_action_counter = 0
        self.current_action = 16
        self.action_threshold = 10
        self.brightness_level = 5
        self.angle = 0
        self.zoom_level = 100
        self.offset_x = 0
        self.offset_y = 0
   
        self.status = 'LOCKED'
        self.locked_image = cv2.resize(cv2.imread('images/locked.png'),  (200, 200))
        self.unlocked_image = cv2.resize(cv2.imread('images/unlocked.png'), (200, 200))
        self.status_image = self.locked_image
        
        
        # Load medical images in series
        self.series = []
        print('Loading Medical Images')
        for series in sorted(os.listdir(os.path.join('images', 'series'))) :
            images = []
            for image_file in os.listdir(os.path.join('images', 'series', series)) :
                images.append(cv2.imread(os.path.join('images', 'series', series, image_file)))
            
            self.series.append({'name' : series, 'images' : images})
        
        self.action_panel_fg_color = (200,0,0)
        self.action_panel_bg_color = (255,255,255)
        
        self.app_panel_fg_color = (0,0,0)
        self.app_panel_bg_color = (169,169,169)
        
        self.action_label_height = 30
        
        # Initialize static display labels
        self.action_panel_title = np.ones((42,512,3),np.uint8)
        self.action_panel_title[:]= self.action_panel_bg_color
        cv2.putText(self.action_panel_title , "ACTION CLASSES",(40,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.action_panel_fg_color, 2)
        
        for action in self.action_classes :
            img = np.zeros((self.action_label_height,286,3),np.uint8)
            img[:]=self.action_panel_bg_color
            cv2.putText(img, action, (40,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.action_panel_fg_color, 2)
            self.action_labels.append(img)
    
    
    
    def convert_to_system_action_prediction(self, jester_prediction, jester_labels) :
        '''
        Converts the Jester network prediction vector to the needed system action probabilities
        Returns
        -------
        numpy array
            System action probabilities
        '''
        system_predictions = np.zeros(len(self.action_classes))

        # Series Navigation
        system_predictions[ self.action_classes.index('Previous Series') ] = jester_prediction[jester_labels.index('Swiping Up')]
        system_predictions[ self.action_classes.index('Next Series') ] = jester_prediction[jester_labels.index('Swiping Down')]
        
        # Series Image Navigation
        system_predictions[ self.action_classes.index('Previous Image') ] = jester_prediction[jester_labels.index('Swiping Left')] 
        system_predictions[ self.action_classes.index('Next Image') ] = jester_prediction[jester_labels.index('Swiping Right')] 

        # Browse Within Image
        system_predictions[ self.action_classes.index('Browse Up') ] = jester_prediction[jester_labels.index('Sliding Two Fingers Up')] 
        system_predictions[ self.action_classes.index('Browse Down') ] = jester_prediction[jester_labels.index('Sliding Two Fingers Down')] 
        system_predictions[ self.action_classes.index('Browse Left') ] = jester_prediction[jester_labels.index('Sliding Two Fingers Left')] 
        system_predictions[ self.action_classes.index('Browse Right') ] = jester_prediction[jester_labels.index('Sliding Two Fingers Right')] 
        
        # Play functionalities
        system_predictions[ self.action_classes.index('Play In Reverse') ] = jester_prediction[jester_labels.index('Rolling Hand Backward')] 
        system_predictions[ self.action_classes.index('Play Forward') ] = jester_prediction[jester_labels.index('Rolling Hand Forward')] 
        system_predictions[ self.action_classes.index('Stop') ] = jester_prediction[jester_labels.index('Stop Sign')]  \
            + jester_prediction[jester_labels.index('Pushing Hand Away')]
        
        # Brightness Manipulation
        system_predictions[ self.action_classes.index('Increase Brightness') ] = jester_prediction[jester_labels.index('Pushing Two Fingers Away')] 
        system_predictions[ self.action_classes.index('Decrease Brightness') ] = jester_prediction[jester_labels.index('Pulling Two Fingers In')] 

        # Rotational Manipulation
        system_predictions[ self.action_classes.index('Rotate Counter-clockwise') ] = jester_prediction[jester_labels.index('Turning Hand Counterclockwise')] 
        system_predictions[ self.action_classes.index('Rotate Clockwise') ] = jester_prediction[jester_labels.index('Turning Hand Clockwise')]

        # Scale Manipilation
        system_predictions[ self.action_classes.index('Zoom In') ] = jester_prediction[jester_labels.index('Zooming In With Full Hand')] \
            + jester_prediction[jester_labels.index('Zooming In With Two Fingers')]
        system_predictions[ self.action_classes.index('Zoom Out') ] = jester_prediction[jester_labels.index('Zooming Out With Full Hand')] \
            + jester_prediction[jester_labels.index('Zooming Out With Two Fingers')]
         
        # Locking Mechanism
        system_predictions[ self.action_classes.index('Thumb Up (Unlock)') ] = jester_prediction[jester_labels.index('Thumb Up')] 
        system_predictions[ self.action_classes.index('Thumb Down (Lock)') ] = jester_prediction[jester_labels.index('Thumb Down')]
        
        # No System Action
        system_predictions[ self.action_classes.index('No System Action') ] = jester_prediction[jester_labels.index('Doing other things')] \
            + jester_prediction[jester_labels.index('No gesture')] + jester_prediction[jester_labels.index('Drumming Fingers')] \
            + jester_prediction[jester_labels.index('Pulling Hand In')] + jester_prediction[jester_labels.index('Pushing Hand Away')] \
            + jester_prediction[jester_labels.index('Shaking Hand')] 
            
        
        return system_predictions
        
    
    
    def construct_action_panel(self, prediction) :
        '''
        Constructs the image containing the probabilistic output of each action class represented as gauges
        Returns
        -------
        numpy array
            Constructed image for the action panel
        '''
        ACTION_PROBABILITIES = []
        for prob in prediction  :
            img = np.zeros((self.action_label_height,226,3),np.uint8)
            img[:]=self.action_panel_bg_color
            pct = int(prob * 225)
            cv2.rectangle(img, (0,0), (pct,self.action_label_height), self.action_panel_fg_color, -1)
            ACTION_PROBABILITIES.append(img)
        
        img = np.hstack([np.vstack(self.action_labels), np.vstack(ACTION_PROBABILITIES)])
        return np.vstack([self.action_panel_title,img])
        
        
    def construct_center_panel(self) :
        '''
        Constructs the image containing the center view panel and application status
        Returns
        -------
        numpy array
            Constructed image for the center panel
        '''
        center_panel = np.zeros((1010,1250,3),np.uint8)
        
        # get current image
        if self.mode == 'play' :
            self.current_image = (self.current_image + 1) % len(self.series[self.current_series]['images'])
        elif self.mode == 'reverse' :
            self.current_image = (self.current_image - 1) % len(self.series[self.current_series]['images'])
        current_image =  self.series[self.current_series]['images'][self.current_image]
        
        # rotate
        current_image = imutils.rotate_bound(current_image, self.angle)
        
        # brightness manipulation        
        if self.brightness_level >= 5 :
            intensity_value = int(20 * (self.brightness_level - 5))
            mask = (255 - current_image) < intensity_value
            adjusted_image = np.where(mask, 255, current_image + intensity_value)
            
        else :
            intensity_value = int(20 * (5 - self.brightness_level))
            mask = current_image < intensity_value
            adjusted_image = np.where(mask, 0 ,current_image - intensity_value)        
        current_image = np.clip(adjusted_image, 0, 255)
        

        # zoom
        rescale_width = int(current_image.shape[1] * self.zoom_level / 100)
        rescale_height = int(current_image.shape[0] * self.zoom_level / 100)
        current_image = cv2.resize(current_image, (rescale_width,rescale_height))
        
        
        center_panel_x = int((center_panel.shape[0] - current_image.shape[0]) / 2 ) + self.offset_x
        center_panel_y = int((center_panel.shape[1] - current_image.shape[1]) / 2 ) + self.offset_y
        len_x = current_image.shape[0]
        len_y = current_image.shape[1]
        current_image_x = 0
        current_image_y = 0
        
        
        # current image is out of bounds
        if center_panel_x >= center_panel.shape[0] :
            None
        elif center_panel_y >= center_panel.shape[1] :
            None
        elif center_panel_x + len_x <= 0 :
            None
        elif center_panel_y + len_y <= 0 :
            None
        # current image within bounds
        else :
            if center_panel_x < 0 :
                len_x = len_x + center_panel_x 
                current_image_x = current_image_x - center_panel_x
                center_panel_x = 0
            if center_panel_y < 0 :
                len_y = len_y + center_panel_y 
                current_image_y = current_image_y - center_panel_y
                center_panel_y = 0
            if center_panel_x + len_x > center_panel.shape[0] :
                len_x = center_panel.shape[0] - center_panel_x
            if center_panel_y + len_y > center_panel.shape[1] :
                len_y = center_panel.shape[1] - center_panel_y
                
            
            # render manipulated current image 
            center_panel[center_panel_x:center_panel_x+len_x, center_panel_y:center_panel_y+len_y] = \
                 current_image[current_image_x:current_image_x+len_x, 
                               current_image_y:current_image_y+len_y]
                            
        
        # Annotate 
        annot_text_color = (255,255,255)
        cv2.putText(center_panel, self.series[self.current_series]['name'].upper(), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        cv2.putText(center_panel, "IMAGE " + str(self.current_image+1), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        
        cv2.putText(center_panel, "BRIGHTNESS: " + str(self.brightness_level), (10,960), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        cv2.putText(center_panel, "ANGLE: " + str(self.angle), (10,980), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        cv2.putText(center_panel, "ZOOM: " + str(self.zoom_level), (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        
        cv2.putText(center_panel, "ANONYMOUS JOE", (1110,960), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        cv2.putText(center_panel, "MALE 50 YO", (1147,980), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        cv2.putText(center_panel, "ABDOMEN", (1168,1000), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annot_text_color, 1)
        
        return center_panel
    
    
    def construct_navigation_panel(self) :
        '''
        Constructs the image containing the series navigation panel
        Returns
        -------
        numpy array
            Constructed image for the status and series navigation panel
        '''
        panel_width = 200
                
        # Series Navigator Label
        navigation_label = np.zeros((42,panel_width,3),np.uint8)
        navigation_label[:] = self.app_panel_bg_color
        cv2.putText(navigation_label, "SERIES NAVIGATOR ",  (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.app_panel_fg_color, 2)
        
        # Series Icons
        for i,series in enumerate(self.series) :
            series_item = np.zeros((192,panel_width,3),np.uint8)
            
            if i == self.current_series :
                series_item[:] = (153,102,52)
                text_color = (255,255,255)
            else :
                series_item[:] = (255,255,255)
                text_color = (0,0,0)
            series_icon = cv2.resize(series['images'][0], (96,96)) 
            series_item[48:48+96,52:52+96] = series_icon

            # Label the series item
            # get boundary of this text
            textsize = cv2.getTextSize(series['name'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = int( (series_item.shape[1] - textsize[0]) / 2 )   
            cv2.putText(series_item, series['name'], (text_x,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            images_cnt_text = str(len(series['images'])) + ' images'
            textsize = cv2.getTextSize(images_cnt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = int( (series_item.shape[1] - textsize[0]) / 2 ) 
            cv2.putText(series_item, images_cnt_text, (text_x,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            series_item[0:2] = self.app_panel_bg_color
            series_item[190:192] = self.app_panel_bg_color
                
            if i == 0 :
                series_list = series_item
            else :
                series_list = np.vstack([series_list,series_item])
            
        # Blank
        blank = np.zeros((192,panel_width,3),np.uint8)        
        blank[:] = self.app_panel_bg_color           
        navigation_panel =  np.vstack([self.status_image, navigation_label, series_list, blank])
      
        return navigation_panel
        
    
    def update_system_status(self, prediction) :
        '''
        Triggers system action based on prediction vector and updates the system status
        '''
        action = np.argmax(prediction)
        
        if action != self.current_action :
            self.current_action_counter = 1
            self.current_action = action
        else :
            self.current_action_counter += 1
        
        
        # Make sure to meet action threshold count
        if self.current_action_counter >= self.action_threshold :
        
            print(self.action_classes[action] + ' ' + str(self.current_action_counter))
            # Unlock Application
            if self.status == 'LOCKED' and action == self.action_classes.index('Thumb Up (Unlock)'):
                self.status = 'UNLOCKED'
                self.status_image = self.unlocked_image
                
            if self.status == 'UNLOCKED' :
            
                # Lock application
                if action == self.action_classes.index('Thumb Down (Lock)'):
                    self.status = "LOCKED"
                    self.status_image = self.locked_image
                    
                # Series Navigation 
                elif action == self.action_classes.index('Previous Series') and self.mode == 'view':
                    self.current_series = (self.current_series - 1) % len(self.series)
                    self.current_image = 0
                elif action == self.action_classes.index('Next Series') and self.mode == 'view':
                    self.current_series = (self.current_series + 1) % len(self.series)
                    self.current_image = 0
                    
                elif action == self.action_classes.index('Previous Image') and self.mode == 'view':
                    self.current_image = (self.current_image - 1) % len(self.series[self.current_series]['images'])
                elif action == self.action_classes.index('Next Image') and self.mode == 'view':
                    self.current_image = (self.current_image + 1) % len(self.series[self.current_series]['images'])  
                    
                # Play functionality
                elif action == self.action_classes.index('Play Forward') and self.mode != 'play':
                    self.mode = 'play'
                elif action == self.action_classes.index('Play In Reverse') and self.mode != 'reverse': 
                    self.mode = 'reverse'
                elif action == self.action_classes.index('Stop') and self.mode != 'view': 
                    self.mode = 'view'
                  
                    
                # Image Rotation
                elif action == self.action_classes.index('Rotate Counter-clockwise') :
                    self.angle = (self.angle - 90) % 360
                elif action == self.action_classes.index('Rotate Clockwise') :
                    self.angle = (self.angle + 90) % 360
                    
                # Brightness Control
                elif action == self.action_classes.index('Increase Brightness') :
                    if self.brightness_level < 10 :
                        self.brightness_level = (self.brightness_level + 1)
                elif action == self.action_classes.index('Decrease Brightness') :
                    if self.brightness_level > 1 :
                        self.brightness_level = (self.brightness_level - 1)
                        
                # Zoom Control
                elif action == self.action_classes.index('Zoom In') :
                    if self.zoom_level < 200 :
                        self.zoom_level = (self.zoom_level + 25)
                elif action == self.action_classes.index('Zoom Out') :
                    if self.zoom_level > 25 :
                        self.zoom_level = (self.zoom_level - 25)
                        
                # Browse Within Image Control
                else :
                    if action == self.action_classes.index('Browse Up'):
                        if self.offset_x > -500 :
                            self.offset_x = self.offset_x - 250
                    elif action == self.action_classes.index('Browse Down'):
                        if self.offset_x < 500 :
                            self.offset_x = self.offset_x + 250
                    elif action == self.action_classes.index('Browse Left'):
                        if self.offset_y > -500 :
                            self.offset_y = self.offset_y - 250
                    elif action == self.action_classes.index('Browse Right'):
                         if self.offset_y < 500 :
                            self.offset_y = self.offset_y + 250
                    
            
            self.current_action_counter = -10
    
    def get_gui_components(self, prediction, jester_labels) :
        '''
        Constructs the whole GUI window
        Returns
        -------
        numpy array
            Constructed image for whole application window
        '''
        
        system_prediction = self.convert_to_system_action_prediction(prediction, jester_labels)
        
        self.update_system_status(system_prediction)
        action_panel = self.construct_action_panel(system_prediction)
        center_panel = self.construct_center_panel()
        navigation_panel = self.construct_navigation_panel()
        
        return action_panel, center_panel, navigation_panel
        