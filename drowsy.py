# dependencies
from turtle import update
import cv2
from matplotlib.pyplot import xlabel
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dis
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# mediapipe tools
mediapipe_face_mesh = mp.solutions.face_mesh
mediapipe_draw_utils = mp.solutions.drawing_utils 

# generate face mesh overlay 
face_mesh_overlay = mediapipe_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mesh structure
mesh_drawing = mediapipe_draw_utils.DrawingSpec(thickness = 1, circle_radius = 2)

# Firebase tools 
cred = credentials.Certificate('config.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://time4break-b3569-default-rtdb.firebaseio.com/'
})

# Firebase reference
ref = db.reference('/')

# Eye tracking functions (euclidian distance and eye aspect ratio)
def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
            
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
            
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    
    top_bottom_dis = euclidean_distance(image, top, bottom)
    
    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis/ top_bottom_dis
    
    return aspect_ratio

# Eye positions 
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

# start video
video = cv2.VideoCapture(0)

# total frames
drowsy_framecount = 0
update_counter = 0
display_update_counter = 0

while video.isOpened():
    # frame reader
    succes, image = video.read()
    
    # start time (for FPS display)
    time_start = time.time()

    # if a frame was empty, skip it
    if not succes:
        print("Frame was skipped due to being empty")
        continue

    # formatting image (convert from BGR to RGB for mediapipe processing)
    flipped_image = cv2.flip(image, 1)
    image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)

    # disable write for faster processing
    image.flags.writeable = False
    results = face_mesh_overlay.process(image)
    image.flags.writeable = True

    # convert image back to BGR 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # image parameters
    height, width, channels = image.shape

    face_coords_2d = []
    face_coords_3d = []

    if not results.multi_face_landmarks:
        continue

    # Drowsiness state
    drowsyness_state = False
    
    # ratio height and width of your eye (width/height -> float)
    ratio_left =  get_aspect_ratio(image, results, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right =  get_aspect_ratio(image, results, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    ratio = (ratio_left + ratio_right)/2.0

    # Tracked points in the mask point cloud
    important_points = set([33, 263, 1, 61, 291, 199])
    for face_landmarks in results.multi_face_landmarks:
        for index, landmark in enumerate(face_landmarks.landmark):
            if index in important_points:
                if index == 1:
                    nose_coords_2d = (landmark.x * width, landmark.y * height)
                    nose_coords_3d = (landmark.x * width, landmark.y * height, landmark.z * 3000)
                x_val, y_val = int(landmark.x * width), int(landmark.y * height)
                
                # append values to 2D facial coordinates
                face_coords_2d.append([x_val, y_val])
                # append values to 3D facial coordinates 
                face_coords_3d.append([x_val, y_val, landmark.z])

        # convert both arrays to numpy arrays
        face_coords_2d = np.array(face_coords_2d, dtype=np.float64)
        face_coords_3d = np.array(face_coords_3d, dtype=np.float64)

        # Camera Focus 
        focal_len = width
        camera_details = [[focal_len, 0, height / 2],
                        [0, focal_len, width / 2],
                        [0,0,1]]

        # Convert to numpy array
        np_arr_camera_details = np.array(camera_details)

        # Initialize a distortion matrix (set all entries to 0)
        distortion = np.zeros((4,1), dtype=np.float)

        # Call solve PnP method on our positional parameters
        success, rotation_vector, translation_vector = cv2.solvePnP(face_coords_3d, face_coords_2d, np_arr_camera_details, distortion)

        '''
        Below we get the rotational data to interpret the position of the nose given the face mesh.
        This data will be used to interpret a users awake state (detect drowsiness).

        Drowsiness Algorithm: 
        
        - Detects Drowsiness if the user's nose is below a certain point threshold for an extended period of time
        - Relays audio queue based on the above conditions (suggests the user takes a break)
        '''

        # Get rotation matrix from the positional data from PnP
        rotation_matrix, jacobian_matrix = cv2.Rodrigues(rotation_vector)
        
        # Angular Data converted to a matrix format
        angles, matrixQ, matrixR, matrixR_x_val, matrixR_y_val, matrixR_z_val = cv2.RQDecomp3x3(rotation_matrix)

        # lateral rotation degree
        full_rotation = 360
        x_lat, y_lat, z_lat = angles[0] * full_rotation, \
                            angles[1] * full_rotation, \
                            angles[2] * full_rotation

        # Head tilt
        upstate = False
        downstate = False
        if y_lat < -5:
            textout = "Left"
            upstate = False
            downstate = False
        elif y_lat > 5:
            textout = "Right"
            upstate = False
            downstate = False
        elif x_lat < -10:
            textout = "Down"
            upstate = False
            downstate = True
        elif x_lat > 10:
            textout = "Up"
            upstate = True
            downstate = False 
        else:
            textout = "Front"
            upstate = False
            downstate = False
        
        if ratio > 3.5:
            if (not upstate and not downstate) and (drowsy_framecount >= 5000):
                drowsyness_state = True
            elif upstate and drowsy_framecount >= 15000:
                drowsyness_state = True
        elif downstate:
            if drowsy_framecount >= 15000:
                drowsyness_state = True
        else:
            drowsyness_state = False
            drowsy_framecount = 0

        # Nose direction output display
        project_nose, jacobian = cv2.projectPoints(nose_coords_3d,
                                                rotation_vector,
                                                translation_vector,
                                                np_arr_camera_details,
                                                distortion)


        '''
        The line going outward from your nose is drawn out by these points. 
        These points represent the start and end points of the line

        Start: nose_tip (starting from the tip of your nose)
        End: scale_nose_direction (scaled line with direction given the starting point of the nose tip)
        '''

        # Nose line
        nose_tip = (int(nose_coords_2d[0]), int(nose_coords_2d[1]))
        scale_nose_direction = (int(nose_coords_2d[0] + y_lat * 10), int(nose_coords_2d[1] - x_lat * 10))
        cv2.line(image, nose_tip, scale_nose_direction, (0,0,255), thickness = 3)

        # Calculate frames per second
        time_end = time.time()
        frame_rate = int(1/ abs(time_end - time_start))
        drowsy_framecount += frame_rate
        update_counter += 1
        if update_counter % 250 == 0:
            display_update_counter += 1 
            if drowsyness_state:
                ref.update({
                    'state': 'Drowsy',
                })
            else:
                ref.update({
                    'state': 'Awake',
                })


        # Annotate video stream with data
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"FPS: {frame_rate}", (50, 150), font, 1, (255,0,0), 2)
        cv2.putText(image, f"Direction: {textout}", (50, 100), font, 1, (255,0,0), 2)
        if drowsyness_state:
            cv2.putText(image, f"State: Drowsy", (50, 50), font, 1, (255,0,0), 2)
        else:
            cv2.putText(image, f"State: Awake", (50, 50), font, 1, (255,0,0), 2)
        cv2.putText(image, f"updates: {display_update_counter}", (400, 50), font, 1, (255,0,0), 2)

        mediapipe_draw_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mediapipe_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mesh_drawing,
            connection_drawing_spec=mesh_drawing)

    cv2.imshow("Pose Estimation", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
