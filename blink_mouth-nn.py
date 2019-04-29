# USAGE
# python blink_mouth-nn.py --shape-predictor shape_predictor_68_face_landmarks.dat --video cam.mp4
# python blink_mouth-nn.py --shape-predictor shape_predictor_68_face_landmarks.dat
# Epoch 5000 : cost = 5030.34 W = 1.5330261 b = -9.608036
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib
import time
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
def eye_aspect_ratio(eye):	
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear
def mouth_ratio(mouth):
	#calculate the distance between the mouth coordinates 
	#only vertical distance as of now 
	mouthAR = dist.euclidean(mouth[3],mouth[18])
	#51 = dist.euclidean(mouth[3])
	#66	= dist.euclidean(mouth[18])
	return mouthAR


pred=0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

if(os.path.isfile("train.csv")):
	fo = open("train.csv","a")
else:
	fo = open("train.csv","w+")

ft = open("test.csv","w+")
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
start_time = time.time()
s=0
temp=0
cmew=np.array([1,2])
# tes=open("005_sleepyCombination_eye.txt","r")
# tes=open("005_slowBlinkWithNodding_eye.txt","r")
# tes=open("022_noglasses_mixing_drowsiness.txt","r")
# tes=open("004_noglasses_mixing_drowsiness.txt","r")

# l=[]
# while True:
'''    c=tes.read(1)
    if not c:
        break
    l.append(int(c))
# print(l)
FNO=0
cor=0
s=0
'''
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	# nframe=frame.copy()
	# nframe=cv2.flip( frame, -1 )
	# print(type(frame))
	if type(frame)!=type(cmew):
		break
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouthAR = mouth_ratio(mouth)

		#Increment the frame no.
		# FNO+=1

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			pre=0
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1 
				s+=1
				curt=(time.time() - start_time)
				# W = 2.655932 
				# b = -1.0237936 Surya
				# W = 1.9900047 
				# b = 1.2658211 
				#chinese bigger eyes
				W=3.84615
				b=1.112096
				y=W*TOTAL + b
				if(abs(y-curt)<=17):
					pre=0
				if(abs(y-curt)>17):
					# print("OUTSIDE")
					pre=1
					cv2.putText(frame,"OUTSIDE",(10,300),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
				# if(l[FNO]==pre):
					# cor+=1
				fo.write(str(curt)+"," + str(TOTAL)+"\n")

			# reset the eye frame counter
			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MouthAR: {}".format(mouthAR), (225, 320),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
cv2.destroyAllWindows()
# accuracy=pred/TOTAL*100
# print(accuracy)
fo.close()
fo = open("train.csv","a")
# W = 2.655932 
# b = -1.0237936 Surya
# W = 1.9900047 
# b = 1.2658211 Chinese small
#chinese bigger
W = 3.84615 
b = 1.112096
column_names=['y','x']
dataset=pd.read_csv('train.csv',names=column_names)
# print(dataset)
column_names=['y','x']
x=dataset.x.tolist()
y=dataset.y.tolist()
plt.scatter(x,y)
x=np.array(x)
y=np.array(y)
y1=np.array(W*(x)+b)
yu=np.array(W*(x)+b+17)
yl=np.array(W*(x)+b-17)
plt.plot(x,y1)
plt.plot(x,yu)
plt.plot(x,yl)
# accuracy=cor/s*100
# rsq=r2_score(y,yp)
# print("R-Squared:",rsq)
# print("ACCURACY:",accuracy)
plt.show()

# do a bit of cleanup

vs.stop()