# import cv2
# import numpy as np
# import face_recognition
# import pickle
# from firebase import FireBase

# class FaceIdentification:
#     def __init__(self, encode_file_path, camera_index=0, width=1280, height=720):
#         self.firebase = FireBase()
#         self.db, self.storage = self.firebase.initializeFirebase()
#         self.bucket = self.storage.bucket()

#         # 카메라 설정
#         self.cap = cv2.VideoCapture(camera_index)
#         self.cap.set(3, width)
#         self.cap.set(4, height)

#         # 인코딩 파일 로드
#         self.encode_file_path = encode_file_path
#         self.encodeListKnown, self.personIds = self.load_encode_file()

#         self.modeType = 0
#         self.counter = 0
#         self.id = -1
#         self.imgPerson = []

#     def load_encode_file(self):
#         print("Loading Encode File ...")
#         with open(self.encode_file_path, 'rb') as file:
#             encodeListWithKnownWithIds = pickle.load(file)
#         print("Encode File Loaded")
#         return encodeListWithKnownWithIds

#     def recognize_faces(self):
#         while True:
#             success, img = self.cap.read()

#             if not success:
#                 print("Failed to grab frame")
#                 break

#             # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # reduce size 1/4
#             imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             faceCurFrame = face_recognition.face_locations(imgS)
#             encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

#             for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#                 matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
#                 faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)

#                 matchIndex = np.argmin(faceDis)

#                 if matches[matchIndex]:
#                     y1, x2, y2, x1 = faceLoc
#                     y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1

#                     self.id = self.personIds[matchIndex]
#                     if self.counter == 0:
#                         self.counter = 1
#                         self.modeType = 1

#             if self.counter != 0:
#                 self.process_recognized_face(img)

#             cv2.imshow("Face Recognition", img)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

#     def process_recognized_face(self, img):
#         if self.counter == 1:
#             # 데이터베이스에서 정보 가져오기
#             personInfo = self.db.reference(f'Person/{self.id}').get()
#             print(personInfo)

#             # 스토리지에서 이미지 가져오기
#             blob = self.bucket.get_blob(f'imgs/{self.id}.png')

#             if blob is not None:
#                 array = np.frombuffer(blob.download_as_string(), np.uint8)
#                 self.imgPerson = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

#             # 사용 횟수 업데이트
#             personInfo['usage_count'] += 1
#             ref = self.db.reference(f'Person/{self.id}')
#             ref.child('usage_count').set(personInfo['usage_count'])
#             print(personInfo['usage_count'])

#         # 인식된 사람의 이름을 이미지에 표시
#         cv2.putText(img, str(personInfo['name']), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#         print(personInfo)
#         self.counter += 1

# if __name__ == "__main__":
#     encode_file_path = '/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/faceRecognition/src/EncodeFile.p'
#     face_identification = FaceIdentification(encode_file_path)
#     face_identification.recognize_faces()

import cv2
import numpy as np
import face_recognition
import pickle

from firebase import FireBase

class FaceIdentification:
    def __init__(self, encode_file_path, camera_index=0, resolution=(1280, 720)):
        self.firebase = FireBase()
        self.db, self.storage = self.firebase.initializeFirebase()
        self.bucket = self.storage.bucket()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(3, resolution[0])  # Width resolution
        self.cap.set(4, resolution[1])  # Height resolution

        # Load the encoding file
        self.encodeListKnown, self.personIds = self.load_encode_file(encode_file_path)

        self.modeType = 0
        self.counter = 0
        self.id = -1
        self.imgPerson = []

    def load_encode_file(self, path):
        print("Loading Encode File ...")
        with open(path, 'rb') as file:
            encodeListWithKnownWithIds = pickle.load(file)
        encodeListKnown, personIds = encodeListWithKnownWithIds
        print("Encode File Loaded")
        return encodeListKnown, personIds

    def recognize_faces(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break

            imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                self.process_face(img, encodeFace, faceLoc)

            self.display_image(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def process_face(self, img, encodeFace, faceLoc):
        matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1

            self.id = self.personIds[matchIndex]

            if self.counter == 0:
                self.counter = 1
                self.modeType = 1

            if self.counter != 0:
                self.retrieve_person_info(img)

    def retrieve_person_info(self, img):
        if self.counter == 1:
            personInfo = self.db.reference(f'Person/{self.id}').get()
            print(personInfo)

            blob = self.bucket.get_blob(f'imgs/{self.id}.png')

            if blob is not None:
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                self.imgPerson = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

            ref = self.db.reference(f'Person/{self.id}')
            personInfo['usage_count'] += 1
            ref.child('usage_count').set(personInfo['usage_count'])

        cv2.putText(img, str(personInfo['name']), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        print(personInfo)
        self.counter += 1

    def display_image(self, img):
        cv2.imshow("Face Recognition", img)

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Usage
face_identifier = FaceIdentification('/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/faceRecognition/src/EncodeFile.p')
face_identifier.recognize_faces()
