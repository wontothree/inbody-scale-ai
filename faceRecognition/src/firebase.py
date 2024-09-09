import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage

data = {
    "111111" :
        {
             "name" : "Sewon Kim",
             "age" : 23,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 10,
        },
    "222222" :
        {
             "name" : "Elon Musk",
             "age" : 53,
             "last_using_time" : "2024-08-31 00:00:01",
             "usage_count" : 12,
        },
    "555555" :
        {
             "name" : "Sewon Kim",
             "age" : 21,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        }
}

class FireBase():
    def initializeFirebase(self):
        # Initialize Firebase
        cred = credentials.Certificate("/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/faceRecognition/src/serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://inbody-scale-ai-default-rtdb.firebaseio.com/",
            'storageBucket': "inbody-scale-ai.appspot.com",
        })

        return db, storage

    def upload_data_to_firebase(self, data):
        db, _ = self.initializeFirebase()
        ref = db.reference('Person')
        
        for key, value in data.items():
            ref.child(key).set(value)

if __name__ == "__main__":
    firebase = FireBase()
    firebase.upload_data_to_firebase(data)
