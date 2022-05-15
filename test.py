import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
cred = credentials.Certificate('configtest.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://time4break-b3569-default-rtdb.firebaseio.com/'
})


ref = db.reference('/startState/')

# Read the data at the posts reference (this is a blocking operation)
print(ref.get()['startState'])

