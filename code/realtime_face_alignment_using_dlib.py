import cv2
import dlib

# load the pretrained HOG SVN model
face_detection_classifier = dlib.get_frontal_face_detector()

# load shape predictor to predict face landmark points of individual facial structures
face_shape_predictor = dlib.shape_predictor('C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/models/shape_predictor_5_face_landmarks.dat')

# open the webcam (0 represents the default camera, you may need to change it based on your system)
# webcam_video_stream = cv2.VideoCapture(0)
#For Prerecorded video

webcam_video_stream = cv2.VideoCapture("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/video.mp4")

while True:
    # capture video frame-by-frame
    ret, frame = webcam_video_stream.read()

    # detect all face locations using the HOG SVN classifier
    all_face_locations = face_detection_classifier(frame, 1)

    # object to hold the 5 face landmark points for every face
    face_landmarks = dlib.full_object_detections()

    # looping through the face locations
    for current_face_location in all_face_locations:
        # looping through all face detections and append shape predictions
        face_landmarks.append(face_shape_predictor(frame, current_face_location))

    # check if there are faces before attempting to get face chips
    if face_landmarks:
        # get all face chips using dlib.get_face_chips()
        all_face_chips = dlib.get_face_chips(frame, face_landmarks)

        # loop through the face chips and show them
        for index, current_face_chip in enumerate(all_face_chips):
            # show the face chip
            cv2.imshow("Face no " + str(index + 1), current_face_chip)

    # display the original frame with face rectangles
    for face_location in all_face_locations:
        x, y, w, h = face_location.left(), face_location.top(), face_location.width(), face_location.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the original frame
    cv2.imshow('Original Frame', frame)

    # check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# release the video capture object and close all windows
webcam_video_stream.release()
cv2.destroyAllWindows()



# #using shape_predictor_68_face_landmarks.dat
# import cv2
# import dlib
# import numpy as np

# # Load the pretrained HOG SVN model
# face_detection_classifier = dlib.get_frontal_face_detector()

# # Load shape predictor to predict face landmark points of individual facial structures
# face_shape_predictor = dlib.shape_predictor('C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/models/shape_predictor_68_face_landmarks.dat')

# # Open the webcam (0 represents the default camera, you may need to change it based on your system)
# webcam_video_stream = cv2.VideoCapture(0)
# # For Prerecorded video

# # webcam_video_stream = cv2.VideoCapture("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/video.mp4")

# while True:
#     # Capture video frame-by-frame
#     ret, frame = webcam_video_stream.read()

#     # Detect all face locations using the HOG SVN classifier
#     all_face_locations = face_detection_classifier(frame, 1)

#     # Object to hold the 68 face landmark points for every face
#     face_landmarks = dlib.full_object_detections()

#     # Looping through the face locations
#     for current_face_location in all_face_locations:
#         # Looping through all face detections and append shape predictions
#         face_landmarks.append(face_shape_predictor(frame, current_face_location))

#     # Check if there are faces before attempting to get face chips
#     if face_landmarks:
#         # Draw face landmarks on the frame
#         for landmarks in face_landmarks:
#             # Convert the landmarks to a NumPy array
#             landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

#             # Loop through the 68 face landmarks and draw them on the frame
#             for (x, y) in landmarks_np:
#                 cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

#     # Display the original frame with face rectangles and landmarks
#     for face_location in all_face_locations:
#         x, y, w, h = face_location.left(), face_location.top(), face_location.width(), face_location.height()
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Show the original frame
#     cv2.imshow('Original Frame', frame)

#     # Check for the 'q' key to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         break

# # Release the video capture object and close all windows
# webcam_video_stream.release()
# cv2.destroyAllWindows()