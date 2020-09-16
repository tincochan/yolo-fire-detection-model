# Using Android IP Webcam video .jpg stream (tested) in Python2 OpenCV3
import argparse
import cv2 as cv
import numpy as np
import time

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image" , help="path to input image")
ap.add_argument("-v", "--video",  help="path to input video")    

args = vars(ap.parse_args())


#setup the model params
confidence_threshold = 0.25
mms_threshold = 0.40
inp_width = 416
inp_height = 416

#get the class from file
classes_file = 'fire.txt'
classes = None

with open(classes_file,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#mode weights and conf
model_conf = 'yt.cfg'
model_weights = 'yt6.weights'

#get model
net = cv.dnn.readNetFromDarknet(model_conf,model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


#gets the output names of the model 
def getOutputsNames(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#gets all the detectedt objects and their bouding boxes
def postProcess(img,outs):
    frame_height = img.shape[0]
    frame_width = img.shape[1]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]


            if confidence > confidence_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)


                width =  int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                left = int(center_x - width/2)
                top = int(center_y - height/2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left,top,width,height])
                
    
    indices = cv.dnn.NMSBoxes(boxes,confidences,confidence_threshold,mms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(class_ids[i],confidences[i],left,top,left + width,top + height,img) 


#draws the bounding boxes 
def drawPred(class_id,conf,left,top,right,bottom,img):
    cv.rectangle(img,(left,top),(right,bottom),(255,178,50),3)

    label = "%.2f" % conf

    if classes:
        assert(class_id <  len(classes))

        label = "%s:%s" % (classes[class_id],label)

    cv.putText(img,label,(left,top),cv.FONT_HERSHEY_PLAIN,1,(255,255,255),1) 


#passes the image through the model and displays result
def getPrediction(img):
    # Finally decode the array to OpenCV usable format ;) 
    blob = cv.dnn.blobFromImage(img,1/255,(inp_width,inp_height),[0,0,0],1,crop=False)
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postProcess(img,outs)

    cv.imshow('result',img)

if args["image"] is not None and args['image'] is not "":

    img = cv.imread(args["image"])

    getPrediction(img)
    
    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows()

elif args["video"] is not None and args['video'] is not "":

    vs = cv.VideoCapture(args["video"]) 

    # loop over frames from the video file stream
    while cv.waitKey(1) < 0:

        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break  

        getPrediction(frame)
       
    cv.destroyAllWindows()    
    





    

  
 

