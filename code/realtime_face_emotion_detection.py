import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
#1. Get the default WebCam Video
#Get the Webcam "0"(means only single camera), (the default one, 1, 2 etc means for additional attached camera)
webcam_video_stream = cv2.VideoCapture(0)

#For Prerecorded video

# webcam_video_stream = cv2.VideoCapture("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/video.mp4")

#2. Initialize the models & load weights
#face expression model initialization
face_exp_model = model_from_json(open("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/facial_expression_model_structure.json","r").read())
#load weights into the model
face_exp_model.load_weights("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/facial_expression_model_weights.h5")
#list(array) of emotion labels
# "0" th index means "angry" "1" th index means "disgust" like this way
emotions_label = ("angry" , "disgust" , "fear" , "happy" , "sad" , "surprise" , "neutral")
#3. Initialize empty array for store all face locations in the frame bcz it's not an static image it's a video
all_face = []
#4. Create an outer while loop to loop through ecah frame of video 
while True:
    #5. Get the current frame the video stream as an image
    # Here the 1st variable "ret" is boolean type which assigned True if the ".read()" was successfull or if we get a valid frame of image
    # "curr_frame" which will be an image at exact frame at the point the statement is executing    
    ret,curr_frame = webcam_video_stream.read()
    #6. Resize the frame to a Quarter(1/4th) of size so that the PC can process it faster 
    # 0..25 means 25% so its 1/4
    curr_frame_small = cv2.resize(curr_frame,(0,0), fx = 0.25, fy = 0.25)
    #7. Find the total no of faces
    # Detect all faces in the image
    # find all face locations using "face_locations()" function 
    # model can be "cnn" or "hog("hog" is faster than "cnn")
    # arguments are "image","number_of_times_to_upsample","model"
    #now here "curr_frame_small" is main resized image
    # Here we did "number_of_times_to_upsample = 2" cause it detects the face far from camera distance  
    all_face = face_recognition.face_locations(curr_frame_small , number_of_times_to_upsample = 2 , model = 'hog')
    # 8. Now after detecting the face locations in main images just find positions of all faces
    #Just run a for loop with index(i) & each face position with condition of the size of tuple(all_face)
    
    for i, curr_face in enumerate(all_face):
        # 9. Splitting the tupple to get the 4 position values of current faces(curr_face)
        top_pos, right_pos, bottom_pos, left_pos = curr_face
        #10. change the position magnitude to fit the actual size video frame
        # Correct the Co-ordinate location to the change the while resizing to 1/4 size inside the loop
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        # 11. Now printing the current face locations of the main images
        print("Found face {} at top : {}, right = {}, bottom = {}, left = {}".format(i+1, top_pos, right_pos, bottom_pos, left_pos))

        # 12. Slice frame image array by positions in "curr_face_img" 
        # Now after finding the locations of the faces just store that locations in array
        # Means slicing the faces from main images
    
        curr_face_img = curr_frame[top_pos: bottom_pos, left_pos: right_pos]

        # 13. Draw rectangle around each face location in the main video frame inside the loop 
        # Draw rectangle around the face detected 
        # Now basically our image in "curr_frame" cause we resized our image by step 9 
        # "(0,0,255)" means (B[Blue],G[Green],R[Red]) and last value 2 means border thickness
        cv2.rectangle(curr_frame , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)

        # 14. Now Preprocess the input image(video frame) cause in dataset we have gray image which had 48*48 pixels each image
        #preprocess input, convert it into an image like as the data in dataset
        # 14.1 Convert it into gray scale 
        curr_face_img = cv2.cvtColor(curr_face_img , cv2.COLOR_BGR2GRAY)
        #14.2 resize into 48*48 px size
        curr_face_img = cv2.resize(curr_face_img , (48,48)) 
        # 14.3 convert the PIL(pillow library) image into a 3D numoy array 
        img_pixels = image.img_to_array(curr_face_img)
        # 14.4 Expand the shape of an array into single row multiple columns(means 1D single dimensional array)
        # Means Row & Columns are convereted into a single row with multiple no of columns
        img_pixels = np.expand_dims(img_pixels , axis = 0)
        # 14.5 pixels are in range of [0,255]("0" -> completely black & "255" -> completely White). Normalize all pixels in scale of [0,1]
        # Actually we did this step cause in our dataset all pixels are gray scale pixels so each and every px's having value in between "0" to "255"
        # So for that we normalized that array 
        # For example Normalize the range 0 to 10 means divide all values by 10[max value], & we will get range as 0 to 1
        # Ex :- 0,1,2,3,...,10 will be 0.0 , 0.1 , 0.2 , 0.3 , ... , 1.0  
        img_pixels /= 255

        # 15. Do predictions & display the results
        # 15.1 do prediction using model, get the prediction values for all 7 expressions 
        exp_predictions = face_exp_model(img_pixels)
        # 15.2 Find max indexed prediction value (0 till 7)
        max_index = np.argmax(exp_predictions[0])
        # 15.3 Get the corresponding label from emotions_label 
        emotion_label = emotions_label[max_index]

        # 15.4 Display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX  #font style
        # By using cv2.puttext() funtion we can give any name or anything
        cv2.putText(curr_frame , emotion_label , (left_pos , bottom_pos) , font , 0.5 , (255 , 255 , 255) , 1)

                
    #16. Displaying the current frame outside for loop inside the while loop
    #Showing the current face with rectangle drawn
    cv2.imshow("Webcam Video : ",curr_frame)
    # 17. Wait for a key press to break the while loop 
    # Press "a" on the keyboard to break the while loop
    # "cv2.waitkey(1)" return a 32 bit value of a key pressed whenever the key "a" is pressed in the keyboard it had break the loop
    # "0xFF" is a 32 bit hexa 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# 18. Once loop breaks, release the camera resources & close all open windows 
# Release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows() 

