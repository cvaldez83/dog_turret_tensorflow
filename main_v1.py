#initial code base copied from: ~/Python/detect_objects_new_cam_WORKS.py



import cv2
import time
from picamera2 import Picamera2
 
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
 
""" USER INPUTS FOR TENSORFLOW """
list_labels_to_detect = ["person"] # what do you want to detect: dog, cat, person, etc
model='efficientdet_lite0.tflite'
num_threads=4
dispW=640
dispH=480




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
 
pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(255,0,0)
 
fps=0
 
base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
detection_options=processor.DetectionOptions(category_name_allowlist=list_labels_to_detect, max_results=1, score_threshold=.3)
options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)
tStart=time.time()
while True:
    ret, im = cam.read()
    im=picam2.capture_array()
    im=cv2.flip(im,-1)
    imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    imTensor=vision.TensorImage.create_from_array(imRGB)
    my_detects=detector.detect(imTensor) #all of the detected objects, output: DetectionResult(detections=[Detection(bounding_box=BoundingBox(origin_x=259, origin_y=184, width=246, height=166), categories=[Category(index=0, score=0.39453125, display_name='', category_name='person')])])

    # print(my_detects)
    # print('end of my_detections')
    for my_detection in my_detects.detections:
        print(my_detection.bounding_box.origin_x)
        UL=(my_detection.bounding_box.origin_x, my_detection.bounding_box.origin_y) #upper left of bounding box
        LR=(my_detection.bounding_box.origin_x + my_detection.bounding_box.width, my_detection.bounding_box.origin_y+my_detection.bounding_box.height) #lower right of bounding box
        target_x_pixel = int(UL[0] + my_detection.bounding_box.width/2)
        target_y_pixel = int(UL[1] + my_detection.bounding_box.height/2)
        # print(f'UL={UL}')
        # print(f'LR={LR}')
        # print(f'target_x_pixel = {target_x_pixel}')
        # im=cv2.rectangle(im,UL,LR,(255,0,255),2)
        UL_target=(target_x_pixel-10,target_y_pixel-10)
        LR_target=(target_x_pixel+10,target_y_pixel+10)
        im=cv2.rectangle(im,UL_target,LR_target,(255,0,0),2) #draw small square center of target
    
        

    
    print()
    
    image=utils.visualize(im, my_detects)
    cv2.putText(im,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    cv2.imshow('Camera',im)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps= .9*fps +.1*1/loopTime
    tStart=time.time()
cv2.destroyAllWindows()