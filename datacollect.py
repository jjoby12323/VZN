import cv2
import os

video = cv2.VideoCapture(0)

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0

nameID = str(input("Enter your Name: ")).lower()

path = 'images/' + nameID

isExist = os.path.exists(path)

if isExist:
    print("Name already exists")
    nameID = str(input("Enter your Name again: ")).lower()
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()
    faces = face_detect.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in faces:
        count = count + 1
        name = './images/' + nameID + '/' + str(count) +'.jpg'
        print("Creating Image...." + name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("Face Detection Program", frame)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    if count == 10:
        break

video.release()
cv2.destroyAllWindows()