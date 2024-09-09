# Face Recognition

    faceRecognition
    ├── imgs/                                       # images folder
    ├── src/            
    |     ├── faceIdentification.py                 # opencv and face_recognition codes
    |     ├── firebase.py                           # add data to Firebase
    |     ├── encodeGenerator.py                    # firebase storage에서 저장된 사진으로 encoding을 추출하여 .p 파일로 저장한다.
    |     ├── serviceAccountKey.json                # authority for Firebase
    |     ├── Encode.p                              # 

# Database

Firebase

- Realtime Database : meta data
- Storage : image data

# Pipeline

## Face Registration

등록하는 과정에서 이미 등록된 사용자는 등록되어 있음을 알린다.

1. Name, sex, age를 입력한다.
2. 이미 등록된 사용자에 대해 이미 등록되어 있음을 알린다.
3. 사진을 찍는다. 이때 사용자가 제대로 된 사진을 찍을 수 있도록 유도한다.
4. 사진에서 얼굴의 encoding vector를 추출한다.
5. 각 사람에 대해 id를 부여한다.
6. Id, name, sex, age, image, encoding_vector, usage_count, last_usage_time을 database에 저장한다.
7. 등록 완료 메시지를 전달한다.

database

|Id|Name|Sex|Age|EncodingVector|UsageCount|LastUsageTime|
|---|---|---|---|---|---|---|

## Face Identification

1. 사진을 촬영한다.
2. 얼굴 인식 및 encoding
3. database에 있는 encodeing과 cosine similarity를 비교한다.
4. 해당하는 encoding을 찾고 사용자 정보를 반환한다.

# Dependencies

```bash
pip install -r requirements.txt
```

```bash
pip install numpy
```

```bash
pip install opencv-python
```

```bash
pip install face_recognition
```

```bash
pip install pickle
```

```bash
pip install firebase_admin
```
