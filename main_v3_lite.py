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
from tabulate import tabulate #to print out variables in a nice table format (debugging purposes)

""" USER INPUTS FOR CODE FLOW """
OPTION_SERVO_SWEEP = 'no' # set to 'yes' to sweep servo full range for testing
OPTION_TRIGGER_ON_OFF = "on"
OPTION_WINDUP_ON_OFF = "on"


""" USER INPUTS FOR GPIO STUFF """
WINDUP_MOTOR_PIN = 14
SWITCH_PIN = 15
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
print(f'setting GPIO.setup for WINDUP_MOTOR_PIN to BCM pin {WINDUP_MOTOR_PIN} as output')
GPIO.setup(WINDUP_MOTOR_PIN,GPIO.OUT) # turret windup motor
print(f'setting GPIO.setup for WINDUP_MOTOR_PIN to BCM pin {WINDUP_MOTOR_PIN} as input')
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # switch to activate/deactivate turret


""" USER INPUTS FOR SERVO i2c STUFF """
MAX_NUMBER_OF_WINDUP = 500 # this is to restrict many times the gun winds up
MAX_NUMBER_OF_SHOTS = 20 # this is to restrict how many times the gun is fired
tTriggerPull = .5 # seconds, hold trigger down for this many seconds
LEFT_MOST_ANGLE = 100 # degrees, gun turret left most angle
RIGHT_MOST_ANGLE = 170 # degrees, gun turret right most angle
TRIGGER_ANGLE_RANGE = 50 # degrees, desired full range of servo in degrees
GUN_WINDUP_TIMER = 2 # seconds, windup gun motor after this many seconds from initial detection
GUN_FIRE_TIMER = 5 # seconds, pull trigger after this many seconds from initial detection
GUN_COOLDOWN_TIMER = 10 # seconds, amount of time to cool down after firing
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


""" AUTO-CALCULATED STUFF FOR SERVO i2c """
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(min_pulse, max_pulse) #channel 0 is the pan servo
kit.servo[1].set_pulse_width_range(min_pulse, max_pulse) #channel 1 is the trigger servo
TURRET_ANGLE_RANGE = RIGHT_MOST_ANGLE - LEFT_MOST_ANGLE


""" PICAMERA2 STUFF """
picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format='RGB888'
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


""" Flag to take pic at startup """
flag_initial_takepic = True

""" Throttle time between firstspot pictures """
COUNTER_FIRSTSPOT_PIC = 0
TIMER_COOLDOWN_FIRST_SPOT_PIC = 300 # number of seconds between pics, 300 = 5min
COOLDOWN_FIRSTSPOT_PIC = time.time() + TIMER_COOLDOWN_FIRST_SPOT_PIC

""" counter for number of shots """
COUNTER_NUMBER_OF_SHOTS = 0
COUNTER_NUMBER_OF_WINDUP = 0

def print_debug(phrase):
    formatted_epoch_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    print(f'{formatted_epoch_time} - DEBUG - {phrase}')

def center_turret():
    print_debug(f'centering turret slowly')
    TURRET_CENTER_ANGLE = TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE
    kit.servo[0].angle = TURRET_CENTER_ANGLE

def move_servo(desired_angle):
    # print_debug(f'move_servo starting')
    step_angle = 1
    # print_debug(f'kit.servo[0].angle = {kit.servo[0].angle}')
    # print_debug(f'rounding = {round(kit.servo[0].angle)}')
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
print_debug(f'moving servo to starting middle position {TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE}')
move_servo(TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE) #Move turret to center at start of program
kit.servo[1].angle = 0 
time.sleep(1)

""" SERVO AUTO-SWEEP FOR DEBUGGING SERVO ISSUES """
if OPTION_SERVO_SWEEP == 'yes':
    while True:
        print_debug(f'setting angle to middle {TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE}')
        move_servo(TURRET_ANGLE_RANGE / 2 + LEFT_MOST_ANGLE)
        time.sleep(5)
        print_debug(f'setting angle to full left {LEFT_MOST_ANGLE}')
        move_servo(LEFT_MOST_ANGLE)
        time.sleep(5)
        print_debug(f'setting angle to full right {RIGHT_MOST_ANGLE}')
        move_servo(RIGHT_MOST_ANGLE)
        time.sleep(5)

flag_quit_program = False #flag used to quit if user presses q
fps=0
tStart=time.time()
time_global_start_of_program = time.time() #global time
time_at_first_spot = 0
init_time_plus_G_F_T = 0
init_time_plus_G_W_T = 0
time_to_cool_down = 0

while True:
    print_debug('DEBUG: first while loop is starting')
    print_debug(f'timers are set to zero, windup motor is off, windup counter: {COUNTER_NUMBER_OF_WINDUP} shots fired counter: {COUNTER_NUMBER_OF_SHOTS}')
    init_time_plus_G_F_T = 0
    init_time_plus_G_W_T = 0 # 5 second timer to turn on gun motor
    time_at_first_spot = 0
    flagSpotted = False
    flagWindingUp = False
    GPIO.output(WINDUP_MOTOR_PIN,GPIO.LOW) # this will windown gun motor

    if GPIO.input(SWITCH_PIN) == 0: # run loop if switch is off
        print_debug("Switch is off")

    while GPIO.input(SWITCH_PIN) == 1 and COUNTER_NUMBER_OF_SHOTS <= MAX_NUMBER_OF_SHOTS: # run loop while switch is on
        print_debug("Switch is on")
        time_current = round(time.time() - time_global_start_of_program, 1)

        """ Camera stuff """
        im=picam2.capture_array()
        im=cv2.flip(im,-1)
        img_uncropped = im
        imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        my_detects=detector.detect(imTensor) #all of the detected objects, output: DetectionResult(detections=[Detection(bounding_box=BoundingBox(origin_x=259, origin_y=184, width=246, height=166), categories=[Category(index=0, score=0.39453125, display_name='', category_name='person')])])
        image=utils.visualize(im, my_detects)
        cv2.putText(im,str(int(fps))+' FPS',pos,font,height,myColor,weight)
        cv2.imshow('Camera',im)
        tEnd=time.time()
        loopTime=tEnd-tStart
        fps= .9*fps +.1*1/loopTime
        tStart=time.time()
        if cv2.waitKey(1)==ord('q'):
            print_debug("User pressed 'q' to quit")
            flag_quit_program = True
            break
        # print_debug(f'DEBUG: my_detects.detections={my_detects.detections}')

        FLAGStable = [['flagSpotted',flagSpotted],
                      ['flagWindingUp',flagWindingUp],
                      ['time_current',time_current],
                      ['time_at_first_spot',time_at_first_spot],
                      ['time_to_cool_down',time_to_cool_down],
                      ['init_time_plus_G_W_T',init_time_plus_G_W_T],
                      ['init_time_plus_G_F_T',init_time_plus_G_F_T],
                      ['COUNTER_NUMBER_OF_WINDUP',COUNTER_NUMBER_OF_WINDUP],
                      ['COUNTER_NUMBER_OF_SHOTS',COUNTER_NUMBER_OF_SHOTS],
                      ['GUN_WINDUP_TIMER',GUN_WINDUP_TIMER],
                      ['GUN_FIRE_TIMER',GUN_FIRE_TIMER],
                      ['GUN_COOLDOWN_TIMER',GUN_COOLDOWN_TIMER],
                      ['MAX_NUMBER_OF_WINDUP',MAX_NUMBER_OF_WINDUP],
                      ['MAX_NUMBER_OF_SHOTS',MAX_NUMBER_OF_SHOTS]]

        print(tabulate(FLAGStable))

        """ For each item that was detected in the frame """
        for my_detection in my_detects.detections:
            # print_debug(f'my_detection={my_detection}')
            if my_detection: #executes code if something was detected
                print_debug('Something detected')
                # print_debug(my_detection)
                # print()
                flagSpotted = True
                # print_debug(my_detection.bounding_box.origin_x)
                UL=(my_detection.bounding_box.origin_x, my_detection.bounding_box.origin_y) #upper left of bounding box
                LR=(my_detection.bounding_box.origin_x + my_detection.bounding_box.width, my_detection.bounding_box.origin_y+my_detection.bounding_box.height) #lower right of bounding box
                target_x_pixel = int(UL[0] + my_detection.bounding_box.width/2)
                target_y_pixel = int(UL[1] + my_detection.bounding_box.height/2)
                center_mass = target_x_pixel
                UL_target=(target_x_pixel-10,target_y_pixel-10)
                LR_target=(target_x_pixel+10,target_y_pixel+10)
                im=cv2.rectangle(im,UL_target,LR_target,(255,0,0),2) #draw small square center of target
                shoot_at_angle = TURRET_ANGLE_RANGE*(center_mass / imageSize_width) + LEFT_MOST_ANGLE
                shoot_at_angle = (180 - shoot_at_angle) + 90
        """ if nothing was detected in the frame """
        if not my_detects.detections: # 
            print_debug('Nothing detected, switching flagSpotted to False')
            break # breaks out of while loop: "while GPIO.input(SWITCH_PIN) == 1"


        """ FIRST TIME OBJECT IS DETECTED """
        if flagSpotted and time_at_first_spot == 0:
            print_debug(f'{str(time_current)} shoot_angle={shoot_at_angle:.2f}. Initial target acquired')
            time_at_first_spot = round(time_current, 1)
            init_time_plus_G_W_T = round(time_current + GUN_WINDUP_TIMER, 1)
            init_time_plus_G_F_T = round(time_current + GUN_FIRE_TIMER, 1)
            move_servo(shoot_at_angle)


        """ WIND UP TURRET IF OBJECT IS STILL BEING DETECTED"""
        if flagSpotted and time_at_first_spot !=0 and time_current > init_time_plus_G_W_T and time_current < init_time_plus_G_F_T and time_current > time_to_cool_down  and COUNTER_NUMBER_OF_WINDUP <= MAX_NUMBER_OF_WINDUP:
            if OPTION_WINDUP_ON_OFF == "on":
                print_debug(f'{str(time_at_first_spot)}, winding up gun, shoot_angle={shoot_at_angle:.2f}. ')
                move_servo(shoot_at_angle)
                GPIO.output(WINDUP_MOTOR_PIN ,GPIO.HIGH) #This will windup the gun motor
                flagWindingUp = True
                COUNTER_NUMBER_OF_WINDUP += 1
            """ This will switch assign True to flagWindingUp since we're obviously in winding up mode """
            if flagWindingUp == False:
                flagWindingUp = True # set flag to True since turret is in winding up mode


        """ FIRE THE TURRET """
        if flagSpotted and flagWindingUp and time_at_first_spot != 0 and time_current > init_time_plus_G_F_T and COUNTER_NUMBER_OF_SHOTS <= MAX_NUMBER_OF_SHOTS:
            print_debug(f'CONFIRMED, thats them alright! Move to servo angle = {shoot_at_angle:.2f}')
            move_servo(shoot_at_angle)
            time.sleep(1)
            print_debug(f'FIRE!')
            if OPTION_TRIGGER_ON_OFF == "on":
                kit.servo[1].angle = TRIGGER_ANGLE_RANGE #activates servo to pull trigger
                time.sleep(tTriggerPull) #trigger is held for this much time
                kit.servo[1].angle = 0 # trigger is released
                time.sleep(1)
            COUNTER_NUMBER_OF_SHOTS += 1
            GPIO.output(WINDUP_MOTOR_PIN ,GPIO.LOW) # this will windown gun motor
            time_to_cool_down = time_current + GUN_COOLDOWN_TIMER
            break # breaks out of while loop: "while GPIO.input(15) == 1"
        
        if not flagSpotted and time_at_first_spot !=0 and time_current > init_time_plus_G_W_T:
            print_debug('target lost')
            GPIO.output(WINDUP_MOTOR_PIN ,GPIO.LOW) # this will windown gun motor
            break # breaks out of while loop: "while GPIO.input(15) == 1"
        
        time.sleep(.2)
    if flag_quit_program == True:
        break
    if cv2.waitKey(1)==ord('q'):
        break

GPIO.cleanup()
cv2.destroyAllWindows()
print_debug("exited program")