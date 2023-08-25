#Face Detection
import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/faces.jpg') # you can also try 00000004.jpg and lena.jpg
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(155,155,0),2)

cv2.imshow("Result",img)
cv2.waitKey(0)
