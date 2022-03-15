from datetime import datetime, timedelta
from ast import literal_eval
from pathlib import Path
import face_recognition
from cv2 import cv2
import pandas as pd
import numpy as np
import pyttsx3
import pyaudio
import shutil
import os

speak = pyttsx3.init()

path = 'img'
images = []
class_names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

now = datetime.now()
date = now.strftime("%Y-%m-%d")
file = open(f'Attendance_{date}.csv','w+')
res = file.name
with open(res,'r+') as d:
    d.writelines('Name,Time,Date')

def markAttendance(name):
    with open(res, 'r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        date_list = []
        now = datetime.now()
        # time = now.strftime("%H:%M:%S")
        date = now.strftime("%Y-%m-%d")
        # new_date = datetime.today() + timedelta(days=1)
        # date_new = new_date.strftime("%Y-%m-%d")
        for line in myDataList:
            entry = line.split(',')
            print(entry)
            date_list.append(entry[0])
        if name not in date_list:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
        #     date = now.strftime("%Y-%m-%d")
            f.writelines(f"\n{name},{time},{date}")

encodeListKnown = find_encodings(images)
# print(len(encodeListKnown))
print('Encoding Complete')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img,(0,0),None,0.25,0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faceCur = face_recognition.face_locations(img_small)
    encodeCur = face_recognition.face_encodings(img_small, faceCur)

    for encodeFace,faceLoc in zip(encodeCur,faceCur):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)

        matchIndex = np.argmin(faceDis)
        value = faceDis[matchIndex]
        if value<0.3:
            if matches[matchIndex]:
                name = class_names[matchIndex]
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
                ans = f'Hi {name}'
                print(ans)
                speak.say(ans)
                speak.runAndWait()

        elif value>=0.3:
            if matches[matchIndex]:
                name = 'Unknown'
                # name = name.upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                ans = 'Hello There, I do not know you, What is your name ?'
                print(ans)
                speak.say(ans)
                speak.runAndWait()
                result = input("Enter your name : ")
                ans = f'Hi {result} can i take your picture so next time i can know you, Please, respond with Yes, or, No'
                print(ans)
                speak.say(ans)
                speak.runAndWait()
                result1 = input("Response:")
                if 'yes' in result1 or 'yeah' in result1 or 'sure' in result1:
                    ans = 'Success, your image shall now be taken'
                    print(ans)
                    speak.say(ans)
                    speak.runAndWait()

                    #Taking the Picture
                    face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

                    while(True):
                        # Capture frame-by-frame
                        ret, frame = cap.read()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                        for (x, y, w, h) in faces:
                            print(x, y, w, h)
                            roi_gray = gray[y:y + h, x:x + w]  # (ycord-height, ycord+height)
                            roi_color = frame[y:y + h, x:x + w]

                            # recognize the region of interest (using deep learning model to predict)

                            img_item = f'{result}.png'
                            cv2.imwrite(img_item, roi_gray)
                            color = (255, 0, 0)  # BGR 0-255
                            thickness = 2
                            end_cordx = x + w
                            end_cordy = y + h
                            cv2.rectangle(frame, (x, y), (end_cordx, end_cordy), color, thickness)
                            ans = f'Thank you for your cooperation, your image has been taken, and by the way, nice to meet you {result}'
                            print(ans)
                            speak.say(ans)
                            speak.runAndWait()

                            #Moving image to the img folder
                            target_folder = r'C:\Users\David Erivona\Documents\Face_Reg\img'
                            source_folder = f'C:\\Users\\David Erivona\\Documents\\Face_Reg\\{img_item}'
                            shutil.move(source_folder, target_folder)

                        # Display the resulting frame
                        cv2.imshow('frame', frame)
                        if cv2.waitKey(20) & 0xFF == ord('q'):
                            break
                else:
                    ans = 'Okay, alright then'
                    print(ans)
                    speak.say(ans)
                    speak.runAndWait()
                    break
    cv2.imshow('Face Recognition Project', img)
    cv2.waitKey(1)