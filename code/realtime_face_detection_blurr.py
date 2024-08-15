import cv2
import face_recognition
# There are 68 landmark points
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
        
        #12. Blurr the current face image

        curr_face_img = cv2.GaussianBlur(curr_face_img , (99,99) , 30)

        #13. Put the blurred face region back into the frame image 
        #Paste the blurred face into the actual frame

        curr_frame[top_pos: bottom_pos, left_pos: right_pos] = curr_face_img

        # 14. Draw rectangle around each face location in the main video frame inside the loop 
        # Draw rectangle around the face detected 
        # Now basically our image in "curr_frame" cause we resized our image by step 9 
        # "(0,0,255)" means (B[Blue],G[Green],R[Red]) and last value 2 means border thickness
        cv2.rectangle(curr_frame , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)
    #15. Displaying the current frame outside for loop inside the while loop
    #Showing the current face with rectangle drawn
    cv2.imshow("Webcam Video : ",curr_frame)
    # 16. Wait for a key press to break the while loop 
    # Press "a" on the keyboard to break the while loop
    # "cv2.waitkey(1)" return a 32 bit value of a key pressed whenever the key "a" is pressed in the keyboard it had break the loop
    # "0xFF" is a 32 bit hexa 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# 17. Once loop breaks, release the camera resources & close all open windows 
# Release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows() 

