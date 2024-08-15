# Reuse the image face detection code
import cv2
import face_recognition

# 1. load image
# original_image = cv2.imread("code/images/testing/E&M.jpg")
original_image = cv2.imread("code/dataset/testing/Arup4.jpg")

# 2. Load Sample Images & Extract the Face Encoding
# Load a sample picture & extract the face encodings
# Returns a list of 128-dimensional face encodings
# "(One for each face in the image)"
# We are getting the first face (assuming only one face)
Arup_image = face_recognition.load_image_file("code/dataset/testing/Arup4.jpg")
Arup_face_encodings = face_recognition.face_encodings(Arup_image)[0]
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

# 3. Create an array to save the encodings
known_face_encodings = [Arup_face_encodings , elon_face_encodings , mark_face_encodings , modi_face_encodings , trump_face_encodings]
# 4. Create another array to hold labels
# Save the encodings & corresponding labels in separate arrays in the same order
known_face_names = ["Arup" , "Elon Musk" , "Mark Zuckerberg" , "Narendra Modi" , "Donald Trump"]
# 5. Load an unknown image to recognize faces in it
# image_to_recognize = face_recognition.load_image_file("code/images/testing/E&M.jpg")
image_to_recognize = face_recognition.load_image_file("code/dataset/testing/Arup4.jpg")
# 6. Find all the faces(locations) & face encodings in the unknown image
''' Now after the loaded image just detect how many faces on that image by using "hog" or "cnn" model.
Now "hog" model is faster and "cnn" model is slower so it takes time to give output so here we used "hog" model'''
#
# Detect all faces in the image
# find all face locations using "face_locations()" function which was included in "face_recognition" module function
# model can be "cnn" or "hog"
# arguments are image,number_of_times_to_upsample,model
all_face_locations = face_recognition.face_locations(image_to_recognize , model = 'hog')
# detect face encodings for all the faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize , all_face_locations)

# 7. Now just print the no of faces as output so here "{}" this is replaced by le(all_face)
# Basically here "{}" this replaced by the value ".format(.......)"

print("There are {} no of face(s) in this image".format(len(all_face_locations)))

# 8. Loop through face location & encodings
# Loop through each face location & face encodings found in the unknown image

for curr_face_location, curr_face_encoding in zip(all_face_locations , all_face_encodings):

    # 9. Splitting the tupple to get the 4 position values of current faces(curr_face_location)
    # "top,right,bottom,left" this order cause it is in clock wise direction 

    top_pos, right_pos, bottom_pos, left_pos = curr_face_location

    # 10. Compare faces & get the matches list(inside the loop)
    # See if the face is any mattch(es) for the known face(s)
    # by using ".compre_faces()" we can compare faces ".compare_faces("known_face_encodings","face_encoding_to_check")"
    all_matches = face_recognition.compare_faces(known_face_encodings, curr_face_encoding)
    # 11. Initialize the name string(inside the for loop)
    # Initialize a name string as "Unknown Face"
    name_of_person = "Unknowwn Face"
    # 12. Use first match & get name from the respective index(inside the loop)
    # If a match was found in  "known_face_encodings", use the first one
    if True in all_matches:
       first_match_index = all_matches.index(True)
       name_of_person = known_face_names[first_match_index]
    # 13. Draw rectangle around the face detected(unknown face[which was stored in "image_to_recognize"])
    # Now basically our image in "curr_frame" cause we resized our image by step 9 
    # "(0,0,255)" means (B[Blue],G[Green],R[Red]) and last value 2 means border thickness
    cv2.rectangle(original_image , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)
    # 14. Write the name below the face
    # Display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX  #font style
    # By using cv2.puttext() funtion we can give any name or anything
    cv2.putText(original_image , name_of_person , (left_pos , top_pos + 20) , font , 2 , (0 , 255 , 0) , 2)
    #Showing the current face with rectangle drawn
    # cv2.imshow("Identified Face ", original_image)
    cv2.imshow("Identified Face ", cv2.resize(original_image, (500, 500)))


#it was used for keeps the output image or webcam as stable for certain time until the user press anything meabs close
cv2.waitKey(0)

#It bassically used when user close the image window which was shown in output after that it just close all opencv output 
cv2.destroyAllWindows()
