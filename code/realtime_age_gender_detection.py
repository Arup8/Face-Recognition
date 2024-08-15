import cv2
import face_recognition
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
    #5. Resize the frame to a Quarter(1/4th) of size so that the PC can process it faster 
    # 0..25 means 25% so its 1/4
    curr_frame_small = cv2.resize(curr_frame,(0,0), fx = 0.25, fy = 0.25)
    #6. Find the total no of faces
    # Detect all faces in the image
    # find all face locations using "face_locations()" function 
    # model can be "cnn" or "hog("hog" is faster than "cnn")
    # arguments are "image","number_of_times_to_upsample","model"
    #now here "curr_frame_small" is main resized image
    # Here we did "number_of_times_to_upsample = 2" cause it detects the face far from camera distance  
    all_face = face_recognition.face_locations(curr_frame_small , number_of_times_to_upsample = 2 , model = 'hog')
    # 7. Now after detecting the face locations in main images just find positions of all faces
    #Just run a for loop with index(i) & each face position with condition of the size of tuple(all_face)
    
    for i, curr_face in enumerate(all_face):
        # 8. Splitting the tupple to get the 4 position values of current faces(curr_face)
        top_pos, right_pos, bottom_pos, left_pos = curr_face
        #9. change the position magnitude to fit the actual size video frame
        # Correct the Co-ordinate location to the change the while resizing to 1/4 size inside the loop
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        # 10. Now printing the current face locations of the main images
        print("Found face {} at top : {}, right = {}, bottom = {}, left = {}".format(i+1, top_pos, right_pos, bottom_pos, left_pos))

        # 11. Slice frame image array by positions in "curr_face_img" 
        # Now after finding the locations of the faces just store that locations in array
        # Means slicing the faces from main images
    
        curr_face_img = curr_frame[top_pos: bottom_pos, left_pos: right_pos]
        
        # 12. Declare the mean value of age and gender
        # 12.1 The "AGE_GENDER_MODEL_MEAN_VALUES" calculated by using "numpy.mean()" 
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  
        # 13. Create "blob"(Binary large object) of face slices
        # 13.1 create blob of current face slice
        # "1" = 100% and "swapRB" means convert the image from BGR to RGB cause in opencv the actual image colour is BGR but universal image colour is RGB 
        curr_face_img_blob = cv2.dnn.blobFromImage(curr_face_img , 1 , (227 , 227) , AGE_GENDER_MODEL_MEAN_VALUES , swapRB = False) 

        # 14. Declare Gender labels, protext & caffemodel file paths
        gender_label_list = ["Male" , "Female"]
        # Creating the file paths
        gender_protext = "C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/gender_deploy.prototxt"
        gender_caffemodel = "C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/gender_net.caffemodel"
        # 15. Create model from files & provide blob as input
        # creating the model
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel , gender_protext)
        # giving to the input model
        gender_cov_net.setInput(curr_face_img_blob)
        #16. Get gender predictions & get label of maximum value returned item
        # Get the predictions from the model(Actually by using ".forward()" it works the further process)
        gender_predictions = gender_cov_net.forward()
        #16.1 Find the max value of predictions index
        #16.2 pass index to label array and get the label
        gender = gender_label_list[gender_predictions[0].argmax()]
        
        # 17. Repeat same steps for Age Prediction
        # 18. Declare Age labels, protext & caffemodel file paths
        age_label_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
        # Creating the file paths
        age_protext = "C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/age_deploy.prototxt"
        age_caffemodel = "C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/age_net.caffemodel"
        # 19. Create model from files & provide blob as input
        # creating the model
        age_cov_net = cv2.dnn.readNet(age_caffemodel , age_protext)
        # giving to the input model
        age_cov_net.setInput(curr_face_img_blob)
        # 20. Get age predictions & get label of maximum value returned item
        # Get the predictions from the model(Actually by using ".forward()" it works the further process)
        age_predictions = age_cov_net.forward()
        #20.1 Find the max value of predictions index
        #20.2 pass index to label array and get the label
        age = age_label_list[age_predictions[0].argmax()]
        # 21. Draw rectangle around each face location in the main video frame inside the loop 
        # Draw rectangle around the face detected 
        # Now basically our image in "curr_frame" cause we resized our image by step 9 
        # "(0,0,255)" means (B[Blue],G[Green],R[Red]) and last value 2 means border thickness
        cv2.rectangle(curr_frame , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)
        # 22. Print the gender & age under the rectangle box
        # Display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX  #font style
        # By using cv2.puttext() funtion we can give any name or anything
        cv2.putText(curr_frame , gender+" "+age+"yrs" , (left_pos , bottom_pos + 20) , font , 0.5 , (0 , 255 , 0) , 1)
    #23. Displaying the current frame outside for loop inside the while loop
    #Showing the current face with rectangle drawn
    cv2.imshow("Webcam Video : ",curr_frame)
    # 24. Wait for a key press to break the while loop 
    # Press "a" on the keyboard to break the while loop
    # "cv2.waitkey(1)" return a 32 bit value of a key pressed whenever the key "a" is pressed in the keyboard it had break the loop
    # "0xFF" is a 32 bit hexa 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# 25. Once loop breaks, release the camera resources & close all open windows 
# Release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows() 

