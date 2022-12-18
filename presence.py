import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

location='Database'
dbimages= []
classNames=[]
myList=os.listdir(location)
for cl in myList:
    curImg=cv2.imread(f'{location}/{cl}')
    dbimages.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList=f.readlines()
        print(myDataList)
        nameList=[]
        for line in myDataList:
            entry =line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('  Present,   %D   %H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')



encodeListKnown=findEncodings(dbimages)

cap=cv2.VideoCapture(1)

while True:
    success, img =cap.read()
    imgS=cv2.resize(img, (0,0),None, 0.25, 0.25)
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrames=face_recognition.face_locations(imgS)
    encodeCurFrames=face_recognition.face_encodings(imgS, facesCurFrames)

    for encodeFace, faceLoc in zip(encodeCurFrames, facesCurFrames):
        matches=face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1, x2, y2, x1=faceLoc
            y1, x2, y2, x1 =y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
            cv2.rectangle(img, (x1,y2-20),(x2,y2),(0,0,0),cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Project',img)
    if cv2.waitKey(1) & 0xFF==ord('e'):
        break