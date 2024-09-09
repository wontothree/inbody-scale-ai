import os
import cv2
import face_recognition
import pickle

from firebase import FireBase

firebase = FireBase()

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList


_, storage = firebase.initializeFirebase()
bucket = storage.bucket()

# importing the mode images into a list
folderPath = '/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/faceRecognition/imgs'
PathList = os.listdir(folderPath)
imgList = []
personIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    personIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, personIds]
print("Encoding Complete")
print(encodeListKnownWithIds)

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")