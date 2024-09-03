import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage

def initializeFirebase():
    # Initialize Firebase
    cred = credentials.Certificate("/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/faceRecognition/src/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://inbody-scale-ai-default-rtdb.firebaseio.com/",
        'storageBucket': "inbody-scale-ai.appspot.com",
    })

    return db, storage
