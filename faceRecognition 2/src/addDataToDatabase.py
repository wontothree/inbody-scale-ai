from firebaseInit import initializeFirebase

db, _ = initializeFirebase()
ref = db.reference('Person')

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

for key, value in data.items():
    ref.child(key).set(value)