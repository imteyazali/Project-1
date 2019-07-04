

import cv2
cap = cv2.VideoCapture(0)

#instantiate the Cascade Classifier with file_name
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
	ret, frame = cap.read() # Status ,Frame

	if not ret:
		continue


	#find all the face in the frame
	face = face_cascade.detectMultiScale(frame, 1.3,5) #frame ,scaling factor, Neighbors

	# print (faces)

	for face in face:
		x,y,w,h = face # Tuple Unpacking

		# Drawing boundary
		cv2.rectangle(frame , (x,y),(x+w , y+h), (0,255,0) , 2) #Frame , Start, End , Color, Thickness

		face_only = frame[y:y+h, x:x+w]
		cv2.imshow("Face Selection" , face_only)


	cv2.imshow("Feed" , frame)

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()


