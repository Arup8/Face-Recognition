# from PIL import Image
# import face_recognition

# # Load the jpg file into a numpy array
# image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/testing/trump-modi.jpg")

# # Find all the faces in the image using a pre-trained convolutional neural network.
# # This method is more accurate than the default HOG model, but it's slower
# # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
# # this will use GPU acceleration and perform well.
# # See also: find_faces_in_picture.py
# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

# print("I found {} face(s) in this photograph.".format(len(face_locations)))

# for face_location in face_locations:

#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()
from PIL import Image
import face_recognition
import cv2
import numpy as np

# Set up video capture (adjust resolution if needed)
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Set width
video_capture.set(4, 480)  # Set height

# Frame skipping for better performance
frame_skip = 5  # Process every 5th frame
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if count % frame_skip == 0:
        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="hog")

        for face_location in face_locations:
            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

    count += 1

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()

