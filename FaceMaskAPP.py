# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:24:10 2023

@author: damjan.janakievski
"""

import pandas as pd
import PySimpleGUI as sg
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import statistics
from PIL import Image, ImageEnhance
from playsound import playsound

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))


    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)
   
   	# initialize our list of faces, their corresponding locations,
   	# and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
   
   	# loop over the detections
    for i in range(0, detections.shape[2]):
   
   		confidence = detections[0, 0, i, 2]
   
   		# filter out weak detections by ensuring the confidence is
   		# greater than the minimum confidence
   		if confidence > 0.5:
   			# compute the (x, y)-coordinates of the bounding box for the object
   			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
   			(startX, startY, endX, endY) = box.astype("int")
   
   			# bounding boxes fall within the dimensions of the frame
   			(startX, startY) = (max(0, startX), max(0, startY))
   			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
   
   			# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224
   			face = frame[startY:endY, startX:endX]
   			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
   			face = cv2.resize(face, (224, 224))
   			face = img_to_array(face)
   			face = preprocess_input(face)
   
   			# add the face and bounding boxes to their respective
   			# lists
   			faces.append(face)
   			locs.append((startX, startY, endX, endY))
   
    if len(faces) > 0:
   		faces = np.array(faces, dtype="float32")
   		preds = maskNet.predict(faces, batch_size=32)
   
    return (locs, preds)

    


while True:
    EXTNS = ['Test On Video Regular', 'Test On Video CCTV','Test On Camera','Test On Camera ALARM']
    form = sg.FlexForm('Choose a option')
    radio_buttons = [sg.Radio(EXTNS[0], 1, key=0, default=True),
                     sg.Radio(EXTNS[1], 1, key=1),
                     sg.Radio(EXTNS[2], 1, key=2),
                     sg.Radio(EXTNS[3], 1, key=3),
                     ]
    layout_start = [
        [sg.Text('Select:', size=(30, 1),
                 font=("Helvetica", 25))],
        radio_buttons,
        [sg.Submit(),sg.Cancel()]
    ]
    window = sg.Window('App', layout_start, size=(900,300))
    button, values = window.read()
    if button == "Submit":
        for _ in values:
            if values[_]:
                value = _
                window.close()
                break
    elif button == "Cancel":
        window.close()
        break
    

    if(value == 0):
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        maskNet = load_model("mask_detector.model")
        
        print("Starting test video...")
        for i in range(1,10):
            print(f'Starting Video {i}...')
            cap = cv2.VideoCapture(f'C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\Videos\\Test_{i}.mp4')
    
            if not cap.isOpened():
                print('Error opening video file')
            while True:
                
                ret, frame = cap.read()
                if not ret:
                    break
                frame = imutils.resize(frame, width=1000)
                
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):
               	 	(startX, startY, endX, endY) = box
               	 	(mask, withoutMask) = pred
               
               	 	label = "Mask" if mask > withoutMask else "No Mask"
               	 	color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
               
               	 	label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
               
               	 	cv2.putText(frame, label, (startX, startY - 10),
               			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
               	 	cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
               
                cv2.imshow('Frame', frame)
               # Wait for a key press and check if the 'q' key was pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
           
            cap.release()
            cv2.destroyAllWindows()
        
    if(value == 1):
        
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        maskNet = load_model("mask_detector_CCTV.model")
        
        print("Starting test video...")
        for i in range(1,10):
            print(f'Starting Video {i}...')
            cap = cv2.VideoCapture(f'C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\Videos\\Test_{i}_grey.mp4')
    
            if not cap.isOpened():
                print('Error opening video file')
            while True:
                
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.GaussianBlur(frame, (5, 5), 1)
                frame = imutils.resize(frame, width=1000)
                
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):
               	 	# unpack the bounding box and predictions
               	 	(startX, startY, endX, endY) = box
               	 	(mask, withoutMask) = pred
               
               	 	label = "Mask" if mask > withoutMask else "No Mask"
               	 	color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
               
               	 	# include the probability in the label
               	 	label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
               
               	 	# display the label and bounding box rectangle on the output frame
               	 	cv2.putText(frame, label, (startX, startY - 10),
               			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
               	 	cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
               
                cv2.imshow('Frame', frame)
               # Wait for a key press and check if the 'q' key was pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
           
            cap.release()
            cv2.destroyAllWindows()
    
    if(value == 2):
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
        # load the face mask detector model
        maskNet = load_model("mask_detector.model")
    
        # video stream
        print("Starting video stream...")
        vs = VideoStream(src=0).start()
    
    
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
    
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                	break
        cv2.destroyAllWindows()
    
    if(value == 3):
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model
        maskNet = load_model("mask_detector.model")

        # video stream
        print("Starting video stream...")
        vs = VideoStream(src=0).start()

        counter_masked = 0
        counter_unmasked = 0
        br = 0
        lista = []
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                br = br+1
                # if (mask > withoutMask):
                #     if mask < 0.95:
                #         label = "Put Your Mask Proparly"
                #     else:
                #         label = "Mask"
                # else:
                #     label = "No Mask" 
                # if label == 'Mask':
                #     color = (0, 255, 0)
                # elif label == 'No Mask':
                #     color = (0, 0, 255)
                # else:
                #     color = (250,250,210)


                
                if (mask > withoutMask):
                    if mask < 0.98:
                        counter_unmasked = counter_unmasked + 1
                        lista.append(mask)
                    else:
                        counter_masked = counter_masked + 1
                        lista.append(mask)
                else:
                    counter_unmasked = counter_unmasked + 1
                    lista.append(withoutMask)
                    
                    
                if(br == 20):
                    probability = statistics.median(lista)
                    if ((counter_masked > counter_unmasked) and (probability >= 0.98)):
                    
                        label = 'Mask'
                        color = (0, 255, 0)
                        label = "{}: {:.2f}%".format(label, probability * 100)
                        
                        cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        
                        br = 0 
                        counter_masked = 0
                        counter_unmasked = 0
                        lista = []
                        
                    elif ((counter_masked <= counter_unmasked)):
                        
                        label = 'No Mask'
                        color = (0, 0, 255)
                        label = "{}: {:.2f}%".format(label, probability * 100)
                        
                        cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        
                        br = 0
                        counter_masked = 0
                        counter_unmasked = 0
                        lista = []
                        playsound('C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\mixkit-alarm-tone-996.wav')

                    elif((probability < 0.98)):
                        label = 'Not Decisive'
                        color = (255, 0, 0)
                        label = "{}:".format(label)
                        
                        cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        
                        br = 0
                        counter_masked = 0
                        counter_unmasked = 0
                        lista = []
                        
                else:
                    label = 'Calculating..'
                    color = (255,127,80)
                    label = "{}: {:.2f}%".format(label, br * 5)
                    
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        
                        
            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
       
        cap.release()
        cv2.destroyAllWindows()
