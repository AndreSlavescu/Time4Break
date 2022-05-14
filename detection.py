
import mediapipe as mp
import cv2
import time

from sqlalchemy import false


#cv2.destroyAllWindows()
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  while cap.isOpened():
    prev_frame_time = 0
    new_frame_time = 0
    success, frame = cap.read()
    new_frame_time = time.time()
 
    # if video finished or no Video Input
    if not success:
        continue

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame


    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None)
            #connection_drawing_spec=mp_drawing_styles
            #.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None)
            #connection_drawing_spec=mp_drawing_styles
            #.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None)
            #connection_drawing_spec=mp_drawing_styles
            #.get_default_face_mesh_iris_connections_style())

    prev_frame_time = new_frame_time
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    fps = int(fps)
    fps = str(fps)
    fpsImage = cv2.putText(image, f"fps = {fps}", (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    print(fps)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', fpsImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()