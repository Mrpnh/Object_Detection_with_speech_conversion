import cv2
import pyttsx3
import time
import math

engine = pyttsx3.init()
engine.setProperty('rate',150)

thres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
dimensions = (640, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def say(classIds):
    classIds_check=set(classIds.flatten())
    for i in range(0,len(classIds)):
         if len(classIds_check)==1:
             engine.say(f"{classNames[(classIds[i][0])-1]} found")
             time.sleep(0.5)
             
         else:
             engine.say(f"{classNames[(classIds[i][0])-1]} and")
             classIds_check.pop()
             time.sleep(0.5)
         
         
        
    engine.runAndWait()
while True:
    success, img = cap.read()
    img = cv2.resize(img, dimensions, cv2.INTER_AREA)
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    if len(classIds) != 0:
        
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("image",img)
    cv2.waitKey(1)
    if(math.floor(confidence*100))>65:
        say(classIds)
