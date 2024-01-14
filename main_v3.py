#initial code base copied from: ~/Python/detect_objects_new_cam_WORKS.py


import os
import cv2
import time
import utils
import numpy as np
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
from picamera2 import Picamera2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
 

""" USER INPUTS FOR CODE FLOW """
OPTION_SERVO_SWEEP = 'no' # set to 'yes' to sweep servo full range for testing
OPTION_TRIGGER_ON_OFF = "on"
OPTION_WINDUP_ON_OFF = "on"
OPTION_CROP_IMAGE = "off"


""" USER INPUTS FOR GPIO STUFF """
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
print('setting GPIO.setup to BCM pin 14 as output')
GPIO.setup(14,GPIO.OUT) # turret windup motor
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


""" USER INPUTS FOR SERVO i2c STUFF """
MAX_NUMBER_OF_WINDUP = 500 # this is to restrict many times the gun winds up
MAX_NUMBER_OF_SHOTS = 20 # this is to restrict how many times the gun is fired
tTriggerPull = .5 # seconds, hold trigger down for this many seconds
LEFT_MOST_ANGLE = 100 # degrees, gun turret left most angle
RIGHT_MOST_ANGLE = 170 # degrees, gun turret right most angle
TRIGGER_ANGLE_RANGE = 50 # degrees, desired full range of servo in degrees
GUN_WINDUP_TIMER = 5 # seconds, windup gun motor after this many seconds from initial detection
GUN_FIRE_TIMER = 10 # seconds, pull trigger after this many seconds from initial detection
min_pulse = 500 # 0 degrees
max_pulse = 2400 # 180 degrees


""" USER INPUTS FOR TENSORFLOW """
list_labels_to_detect = ["person"] # what do you want to detect: dog, cat, person, etc
model='efficientdet_lite0.tflite'
num_threads=4
dispW=640
dispH=360
base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
detection_options=processor.DetectionOptions(category_name_allowlist=list_labels_to_detect, max_results=1, score_threshold=.3)
options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)

""" USER INPUT FOR OPENCV STUFF """
CAMERA_INPUT = "picam" #picam or videofile
frequency = 500  # Set Frequency To 2500 Hertz
duration = 200  # Set Duration To 1000 ms == 1 second
imageSize_width = dispW # can delete if you replace in code
minDetectionSize = (5,5)
maxDetectionSize = (50,50)
pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(255,0,0)


# Crop dimensions
height_start = 250
height_end = 450
width_start = 180
width_end = 600


""" AUTO-CALCULATED STUFF FOR SERVO i2c """
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(min_pulse, max_pulse) #channel 0 is the pan servo
kit.servo[1].set_pulse_width_range(min_pulse, max_pulse) #channel 1 is the trigger servo
TURRET_ANGLE_RANGE = RIGHT_MOST_ANGLE - LEFT_MOST_ANGLE


""" CONSTANTS """
filename_startup_pic = "opencv_pic00_startup.png"
filename_continuous_pic = "opencv_pic10_continuous.png"
filename_firstspot_pic = "opencv_pic20_firstspot.png"
filename_windcup_pic =  "opencv_pic30_windup.png"
filename_fire_pic = "opencv_pic40_fire.png"


""" delete png image files from top-level directory only """
for item in os.listdir(os.getcwd()):
    if item.endswith(".png"):
        os.remove(item)


""" PICAMERA2 STUFF """
picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format='RGB888'
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

webCam=0
cam=cv2.VideoCapture(webCam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)

""" Flag to take pic at startup """
flag_initial_takepic = True

""" Throttle time between firstspot pictures """
COUNTER_FIRSTSPOT_PIC = 0
TIMER_COOLDOWN_FIRST_SPOT_PIC = 300 # number of seconds between pics, 300 = 5min
COOLDOWN_FIRSTSPOT_PIC = time.time() + TIMER_COOLDOWN_FIRST_SPOT_PIC

""" counter for number of shots """
COUNTER_NUMBER_OF_SHOTS = 0
COUNTER_NUMBER_OF_WINDUP = 0

def move_servo(desired_angle):
    # print(f'move_servo starting')
    step_angle = 1
    print(f'kit.servo[0].angle = {kit.servo[0].angle}')
    print(f'rounding = {round(kit.servo[0].angle)}')
    starting_angle = int(round(kit.servo[0].angle))
    current_angle = starting_angle
    desired_angle = int(round(desired_angle))
    if starting_angle < 0 or current_angle < 0:
        starting_angle = 0
        current_angle =0
    elif starting_angle > 180 or current_angle > 180:
        starting_angle = 180
        current_angle =180

    # angle_range = abs(current_angle - desired_angle)
    if desired_angle >= starting_angle:
        current_angle = starting_angle + step_angle
        while desired_angle >= current_angle:
            # if current_angle > TURRET_ANGLE_RANGE:
            #     current_angle = TURRET_ANGLE_RANGE
            kit.servo[0].angle = current_angle
            time.sleep(.1)
            current_angle = current_angle + step_angle
    elif desired_angle < starting_angle:
        current_angle = starting_angle - step_angle
        while desired_angle <= current_angle:
            if current_angle < 0:
                current_angle = 0
            kit.servo[0].angle = current_angle
            time.sleep(.1)
            current_angle = current_angle - step_angle

""" MOVE SERVOS TO STARTING POSITIONS """
print(f'moving servo to starting middle position {TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE}')
move_servo(TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE) #Move turret to center at start of program
kit.servo[1].angle = 0 
time.sleep(1)

""" SERVO AUTO-SWEEP FOR DEBUGGING SERVO ISSUES """
if OPTION_SERVO_SWEEP == 'yes':
    while True:
        print(f'setting angle to middle {TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE}')
        move_servo(TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE)
        time.sleep(5)
        print(f'setting angle to full left {LEFT_MOST_ANGLE}')
        move_servo(LEFT_MOST_ANGLE)
        time.sleep(5)
        print(f'setting angle to full right {RIGHT_MOST_ANGLE}')
        move_servo(RIGHT_MOST_ANGLE)
        time.sleep(5)


flag_quit_program = False #flag used to quit if user presses q
fps=0
tStart=time.time()

while True:
    print('DEBUG: first while loop is starting')
    init_time_plus_G_W_T = 0 # 5 second timer to turn on gun motor
    init_time_plus_t10s = 0 # 10 second timer to pull trigger
    tInitial = 0
    GPIO.output(14,GPIO.LOW) # this will windown gun motor
    print(f'timers are set to zero, windup motor is off, windup counter: {COUNTER_NUMBER_OF_WINDUP} shots fired counter: {COUNTER_NUMBER_OF_SHOTS}')
    flagSpotted = False
    flagWindingUp = False

    if GPIO.input(15) == 0: # run loop if switch is off
        print("Switch is off")

    if GPIO.input(15) == 1 and COUNTER_NUMBER_OF_SHOTS <= MAX_NUMBER_OF_SHOTS: # run loop while switch is on
        print("Switch is on")

        """ Camera stuff """
        ret, im = cam.read()
        im=picam2.capture_array()
        im=cv2.flip(im,-1)
        img_uncropped = im
        imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        my_detects=detector.detect(imTensor) #all of the detected objects, output: DetectionResult(detections=[Detection(bounding_box=BoundingBox(origin_x=259, origin_y=184, width=246, height=166), categories=[Category(index=0, score=0.39453125, display_name='', category_name='person')])])
        print(GPIO.input(15))

        time_minus_tInitial = time.time()-tInitial  
        print(f'time_minus_tInitial = {time_minus_tInitial:.2f}')  

        """ Take and save pic at startup to make sure things are working"""
        if flag_initial_takepic == True:
            img_name = f'opencv_pic00_startup.png'
            cv2.imwrite(img_name, im)
            flag_initial_takepic = False

        # print(my_detects)
        # print('end of my_detections')
        for my_detection in my_detects.detections:
            if my_detection: #executes code if something was detected
                print(my_detection)
                print()
                flagSpotted = True
                # print(my_detection.bounding_box.origin_x)
                UL=(my_detection.bounding_box.origin_x, my_detection.bounding_box.origin_y) #upper left of bounding box
                LR=(my_detection.bounding_box.origin_x + my_detection.bounding_box.width, my_detection.bounding_box.origin_y+my_detection.bounding_box.height) #lower right of bounding box
                target_x_pixel = int(UL[0] + my_detection.bounding_box.width/2)
                target_y_pixel = int(UL[1] + my_detection.bounding_box.height/2)
                center_mass = target_x_pixel
                print(f'center_mass = {center_mass}')
                # print(f'UL={UL}')
                # print(f'LR={LR}')
                # print(f'target_x_pixel = {target_x_pixel}')
                # im=cv2.rectangle(im,UL,LR,(255,0,255),2)
                UL_target=(target_x_pixel-10,target_y_pixel-10)
                LR_target=(target_x_pixel+10,target_y_pixel+10)
                im=cv2.rectangle(im,UL_target,LR_target,(255,0,0),2) #draw small square center of target
                shoot_at_angle = TURRET_ANGLE_RANGE*(center_mass / imageSize_width) + LEFT_MOST_ANGLE
                shoot_at_angle = (180 - shoot_at_angle) + 90

                """ saves overwritable pic continuously regardless of detection (used for debugging)"""
                if str(round(time_minus_tInitial,1))[-1] == "5": 
                    cv2.imwrite(filename_continuous_pic, img_uncropped)
            
            else:
                flagSpotted = False
                GPIO.output(14,GPIO.LOW) # this will windown gun motor
                """ saves overwritable pic continuously regardless of detection (used for debugging)"""
                if str(round(time_minus_tInitial,1))[-1] == "5": 
                    cv2.imwrite(filename_continuous_pic, img_uncropped)
                """ Take pic only every 5 minutes """
                if COUNTER_FIRSTSPOT_PIC == 0:
                    formatted_epoch_time = time.strftime('%Y%m%d_%H:%M:%S', time.localtime(time.time()))
                    cv2.imwrite(f'opencv_pics/continuous_{formatted_epoch_time}.png', img_uncropped)
                    COUNTER_FIRSTSPOT_PIC = 1
                elif COUNTER_FIRSTSPOT_PIC == 1 and time.time() > COOLDOWN_FIRSTSPOT_PIC:
                    COUNTER_FIRSTSPOT_PIC = 0
                    COOLDOWN_FIRSTSPOT_PIC = time.time() + TIMER_COOLDOWN_FIRST_SPOT_PIC # Timer set to 5 minutes
                break

        image=utils.visualize(im, my_detects)
        cv2.putText(im,str(int(fps))+' FPS',pos,font,height,myColor,weight)
        cv2.imshow('Camera',im)
        if cv2.waitKey(1)==ord('q'):
            print("User pressed 'q' to quit")
            flag_quit_program = True
            break
        tEnd=time.time()
        loopTime=tEnd-tStart
        fps= .9*fps +.1*1/loopTime
        tStart=time.time()
    if flag_quit_program == True:
        break
    if cv2.waitKey(1)==ord('q'):
        break

GPIO.cleanup()
cv2.destroyAllWindows()
print("exited program")