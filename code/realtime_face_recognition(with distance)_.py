import cv2
import face_recognition
import numpy as np
#1. Get the default WebCam Video
#Get the Webcam "0"(means only single camera), (the default one, 1, 2 etc means for additional attached camera)
webcam_video_stream = cv2.VideoCapture(0)

#For Prerecorded video

# webcam_video_stream = cv2.VideoCapture("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/video.mp4")

# Set a threshold for face distance
face_distance_threshold = 0.58

# 2. Load Sample Images & Extract the Face Encoding
# Load a sample picture & extract the face encodings
# Returns a list of 128-dimensional face encodings
# "(One for each face in the image)"
# We are getting the first face (assuming only one face)
Arup_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Arup.jpeg")
Arup_face_encodings = face_recognition.face_encodings(Arup_image)[0]
# load next Sample Image & Extract the Face Encoding
Arup1_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Arup2.jpeg")
Arup1_face_encodings = face_recognition.face_encodings(Arup1_image)[0]
# load next Sample Image & Extract the Face Encoding
elon_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/E.jpg")
elon_face_encodings = face_recognition.face_encodings(elon_image)[0]
# load next Sample Image & Extract the Face Encoding
mark_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Mark Zuckerberg.webp")
mark_face_encodings = face_recognition.face_encodings(mark_image)[0]
# load next Sample Image & Extract the Face Encoding
modi_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/modi.jpg")
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]
# load next Sample Image & Extract the Face Encoding
trump_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/trump.jpg")
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]
# load next Sample Image & Extract the Face Encoding
shahrukh_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/shah-rukh-khan.jpg")
shahrukh_face_encodings = face_recognition.face_encodings(shahrukh_image)[0]
# load next Sample Image & Extract the Face Encoding
shahrukh1_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/testing/Shah-Rukh-Khan.jpg")
shahrukh1_face_encodings = face_recognition.face_encodings(shahrukh1_image)[0]
# load next Sample Image & Extract the Face Encoding
rdj_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/RDJ.jpg")
rdj_face_encodings = face_recognition.face_encodings(rdj_image)[0]
# load next Sample Image & Extract the Face Encoding
rdj1_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/RDJ1.jpg")
rdj1_face_encodings = face_recognition.face_encodings(rdj1_image)[0]
# load next Sample Image & Extract the Face Encoding
rdj2_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/RDJ2.jpg")
rdj2_face_encodings = face_recognition.face_encodings(rdj2_image)[0]
# load next Sample Image & Extract the Face Encoding
kohli_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Virat_Kohli.jpg")
kohli_face_encodings = face_recognition.face_encodings(kohli_image)[0]
# load next Sample Image & Extract the Face Encoding
msd_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/MSD.jpg")
msd_face_encodings = face_recognition.face_encodings(msd_image)[0]
# load next Sample Image & Extract the Face Encoding
tom_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Tom_Holland.jpg")
tom_face_encodings = face_recognition.face_encodings(tom_image)[0]
# load next Sample Image & Extract the Face Encoding 
tobey_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Tobey_Maguire.jpg")
tobey_face_encodings = face_recognition.face_encodings(tobey_image)[0]
# load next Sample Image & Extract the Face Encoding
andrew_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Andrew Garfield.jpg")
andrew_face_encodings = face_recognition.face_encodings(andrew_image)[0]
# load next Sample Image & Extract the Face Encoding
sachin_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Sachin.jpg")
sachin_face_encodings = face_recognition.face_encodings(sachin_image)[0]
# load next Sample Image & Extract the Face Encoding
henry_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Henry_Cavill.jpg")
henry_face_encodings = face_recognition.face_encodings(henry_image)[0]
# 3. Create an array to save the encodings
known_face_encodings = [Arup_face_encodings , Arup1_face_encodings , elon_face_encodings , mark_face_encodings , modi_face_encodings , trump_face_encodings , shahrukh_face_encodings , shahrukh1_face_encodings , rdj_face_encodings , rdj1_face_encodings , rdj2_face_encodings , kohli_face_encodings , msd_face_encodings , tom_face_encodings , tobey_face_encodings , andrew_face_encodings , sachin_face_encodings , henry_face_encodings]
# 4. Create another array to hold labels
# Save the encodings & corresponding labels in separate arrays in the same order
known_face_names = ["Arup" , "Arup" , "Elon Musk" , "Mark Zuckerberg" , "Narendra Modi" , "Donald Trump" , "Shah Rukh Khan" , "Shah Rukh Khan" , "Robert Downey Jr." , "Robert Downey Jr." , "Robert Downey Jr." , "Virat Kohli" , "MS Dhoni" , "Tom Holland" , "Tobey Maguire" , "Andrer Garfield" , "Sachin Tendulkar" , "Henry Cavill"]


#5. Initialize empty array for store all face locations in the frame bcz it's not an static image it's a video
all_face_locations = []
#6. Initialize the array to hold all face encodings & labels in frame
all_face_encodings = []
all_face_names = []
#7. Create an outer while loop to loop through ecah frame of video 
while True:
    #8. Get the current frame the video stream as an image
    # Here the 1st variable "ret" is boolean type which assigned True if the ".read()" was successfull or if we get a valid frame of image
    # "curr_frame" which will be an image at exact frame at the point the statement is executing    
    ret,curr_frame = webcam_video_stream.read()
    #9. Resize the frame to a Quarter(1/4th) of size so that the PC can process it faster 
    # 0..25 means 25% so its 1/4
    curr_frame_small = cv2.resize(curr_frame,(0,0), fx = 0.25, fy = 0.25)
    #10. Find the total no of faces
    # Detect all faces in the image
    # find all face locations using "face_locations()" function 
    # model can be "cnn" or "hog("hog" is faster than "cnn")
    # arguments are "image","number_of_times_to_upsample","model"
    #now here "curr_frame_small" is main resized image
    # Here we did "number_of_times_to_upsample = 2" cause it detects the face far from camera distance  
    all_face_locations = face_recognition.face_locations(curr_frame_small , number_of_times_to_upsample = 1 , model = 'hog')
    # 11. Now after detecting the face locations in main images just find positions of all faces
    #Just run a for loop with index(i) & each face position with condition of the size of tuple(all_face)
    
    all_face_encodings = face_recognition.face_encodings(curr_frame_small , all_face_locations)

    # 12. Now just print the no of faces as output so here "{}" this is replaced by le(all_face)
    # Basically here "{}" this replaced by the value ".format(.......)"

    print("There are {} no of face(s) in this image".format(len(all_face_locations)))

    # 13. Loop through face location & encodings
    # Loop through each face location & face encodings found in the unknown image

    for curr_face_location, curr_face_encoding in zip(all_face_locations , all_face_encodings):

        # 14. Splitting the tupple to get the 4 position values of current faces(curr_face_location)
        # "top,right,bottom,left" this order cause it is in clock wise direction 

        top_pos, right_pos, bottom_pos, left_pos = curr_face_location
        #15. change the position magnitude to fit the actual size video frame
        # Correct the Co-ordinate location to the change the while resizing to 1/4 size inside the loop
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        # 16. Compare faces & get the matches list(inside the loop)
        # See if the face is any mattch(es) for the known face(s)
        # by using ".compre_faces()" we can compare faces ".compare_faces("known_face_encodings","face_encoding_to_check")"
        all_matches = face_recognition.compare_faces(known_face_encodings, curr_face_encoding)
        # 17. Initialize the name string(inside the for loop)
        # Initialize a name string as "Unknown Face"
        name_of_person = "Unknowwn Face"
        # 18. Use first match & get name from the respective index(inside the loop)
        # If a match was found in  "known_face_encodings", use the first one
        face_distances = face_recognition.face_distance(known_face_encodings, curr_face_encoding)
        matches = face_distances < face_distance_threshold

        if np.any(matches):
            first_match_index = np.argmin(face_distances)
            name_of_person = known_face_names[first_match_index]
        # else:
        #     name_of_person = "Unknown Face"

        # 19. Draw rectangle around the face detected(unknown face[which was stored in "image_to_recognize"])
        # Now basically our image in "curr_frame" cause we resized our image by step 9 
        # "(0,0,255)" means (B[Blue],G[Green],R[Red]) and last value 2 means border thickness
        cv2.rectangle(curr_frame , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)
        # 20. Write the name below the face

        # Draw a label with a name below the face
        cv2.rectangle(curr_frame, (left_pos, bottom_pos - 35), (right_pos, bottom_pos), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(curr_frame, name_of_person, (left_pos + 6, bottom_pos - 6), font, 1.0, (255, 255, 255), 1)
        # Display the name as text in the image

        # font = cv2.FONT_HERSHEY_DUPLEX  #font style

        # # By using cv2.puttext() funtion we can give any name or anything

        # cv2.putText(curr_frame , name_of_person , (left_pos , bottom_pos + 20) , font , 0.5 , (255 , 255 , 0) , 1)

    #Showing the current face with rectangle drawn
    cv2.imshow("Identified Face ", curr_frame)





    # 13. Wait for a key press to break the while loop 
    # Press "a" on the keyboard to break the while loop
    # "cv2.waitkey(1)" return a 32 bit value of a key pressed whenever the key "a" is pressed in the keyboard it had break the loop
    # "0xFF" is a 32 bit hexa 
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# 14. Once loop breaks, release the camera resources & close all open windows 
# Release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows() 

