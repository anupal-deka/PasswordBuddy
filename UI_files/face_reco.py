import os
import cv2
import numpy as np
import face_recognition

path='C:/Python36/PROJECT/_TestApp/image_folder'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencoding(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnown=findencoding(images)
print('encodeing complete')

cap=cv2.VideoCapture(1)

while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrames=face_recognition.face_locations(imgs)
    enCurrframe=face_recognition.face_encodings(imgs,facesCurFrames)

    for encodeFace,faceloc in zip(enCurrframe,facesCurFrames):
        matches=face_recognition.compare_faces(encodelistKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodelistKnown,encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1=faceloc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            exec
    cv2.imshow('webcame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow()