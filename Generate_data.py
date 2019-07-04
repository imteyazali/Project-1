import cv2
import numpy as np 
import time


name  = input('Enter Name : ')
num = int(input('Number of photos : '))
face_data = []

cap = cv2.VideoCapture(0)

# instantiate the Cascade Classifier with file_name
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True and num:
	time.sleep(0.5) #Sleep for 0.5second
	ret , frame = cap.read() # status, frame

	if not ret:
		continue

	cv2.imshow('Feed' , frame)
	# find all faces in the frame
	faces = face_cascade.detectMultiScale(frame , 1.3, 5) #Frame , Scaling, Neighbor

	faces = sorted(faces , key = lambda x: x[2]*x[3], reverse = True)
	faces = faces[:1]

	# Print faces

	for face in faces:
		x,y,w,h = face # Tuople unpacking

		# Drawing boundary
		cv2.rectangle(frame , (x,y), (x+w , y+h) , (0,255,0), 2)

		face_only = frame[y:y+h, x:x+h]
		face_only = cv2.resize(face_only, (100,100))
		face_data.append(face_only)
		num -= 1

		cv2.imshow('Face Selection' , face_only)

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break




print(len(face_data))
face_data = np.array(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0] , -1))
print(face_data.shape)
np.save(('face_dataset/' + name), face_data)

cap.release()
cv2.destroyAllWindows()

