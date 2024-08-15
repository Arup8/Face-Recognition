import face_recognition
import cv2
from PIL import Image,ImageDraw
import numpy as np

#1. Get the default WebCam Video
#Get the Webcam "0"(means only single camera), (the default one, 1, 2 etc means for additional attached camera)
webcam_video_stream = cv2.VideoCapture(0)

#For Prerecorded video

# webcam_video_stream = cv2.VideoCapture("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/video.mp4")

#2. Initialize empty array for store all face locations in the frame bcz it's not an static image it's a video
all_face = []
#3. Create an outer while loop to loop through ecah frame of video 
while True:
    #4. Get the current frame the video stream as an image
    # Here the 1st variable "ret" is boolean type which assigned True if the ".read()" was successfull or if we get a valid frame of image
    # "curr_frame" which will be an image at exact frame at the point the statement is executing    
    ret,curr_frame = webcam_video_stream.read()
    # 5. Find all face landmarks & Print the list to check
    # 5.1 Find all facial landmarks in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(curr_frame)
    # 5.2 print all face landmarks & no. of faces
    print(face_landmarks_list)
    # 5.3 no. of faces
    print(len(face_landmarks_list))
    # 6. convert the numoy array image into the pil image object & create a draw object
    # convert the numoy array image into the pil image object
    pil_image = Image.fromarray(curr_frame)
    # Convert the PIL image to draw object
    d = ImageDraw.Draw(pil_image)
    i = 0
    # 7. Now after getting face landmarks points we will join all the landmarks to draw
    # 8. Iterate loop to iterate through each face which contains all face landmarks 
    # For each face
    while i < len(face_landmarks_list):
        for face_landmarks in face_landmarks_list:
            # 9. Draw or join white line joining each face landmarks points
            d.line(face_landmarks["chin"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["left_eyebrow"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["right_eyebrow"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["nose_bridge"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["nose_tip"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["left_eye"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["right_eye"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["top_lip"] , fill = (255,255,255) , width=2)
            d.line(face_landmarks["bottom_lip"] , fill = (255,0,0) , width=2)
        i += 1
    # 10. Now for show the image in opencv we need to convert the PIL image to opencv image(cuase now the image is in form of PIL not opencv)
    # convert PIL image to RGB to show in opencv window
    rgb_image = pil_image.convert('RGB')
    # Convert the image into numpy array(cause opencv is able to load numpy based array)
    rgb_opencv_image = np.array(pil_image)
    # But opencv use BGR format so we need to convert
    #convert RGB to BGR
    bgr_opencv_image = cv2.cvtColor(rgb_opencv_image , cv2.COLOR_RGB2BGR)
    bgr_opencv_image = bgr_opencv_image[ :, :, ::-1].copy()
    #Showing the current face with rectangle drawn
    cv2.imshow("Webcam Video : ",bgr_opencv_image)
    # 11. Wait for a key press to break the while loop 
    # Press "a" on the keyboard to break the while loop
    # "cv2.waitkey(1)" return a 32 bit value of a key pressed whenever the key "a" is pressed in the keyboard it had break the loop
    # "0xFF" is a 32 bit hexa 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# 12. Once loop breaks, release the camera resources & close all open windows 
# Release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
     