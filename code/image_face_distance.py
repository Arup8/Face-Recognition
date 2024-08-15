# Reuse the image face detection code
import cv2
import face_recognition

# 1. load image
image_to_recognize_path = "C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/testing/Arup.jpeg"

original_image = cv2.imread(image_to_recognize_path)

# 2. Load Sample Images & Extract the Face Encoding
# Load a sample picture & extract the face encodings
# Returns a list of 128-dimensional face encodings
# "(One for each face in the image)"
# We are getting the first face (assuming only one face)
Arup_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/Arup2.jpeg")
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
# 5. Load an unknown image with preferably a single face in it
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]
# 6. Face Distance Implementation
# 6.1 Find the face distances using ".face_distance()" method
# find the distance of current encoding with all known encodings
face_distances = face_recognition.face_distance(known_face_encodings , image_to_recognize_encodings)
# 6.2 Loop through every face distance obtained against the face samples & print the value
# print the face distance for each known sample to the unknown image
for i,face_distance in enumerate(face_distances):
    # Remember One thing after getting the face distance value of each image,find which is low face distance then that low face distance is perfect accurate recognized image
    # If the distance increases then matching decreases
    print("The calculated face distance is {:.2} against the sample {}".format(face_distance,known_face_names[i])) #"{:.2}" means [2 decimal value Ex:- 0.37]
    # For getting matching percentage the calculation will be "round-off(((1 - calculated face distance) * 100),2)" hete 2 means [2 decimal value]
    # If the percentage increases then matching increases 
    print("The matching percentage is {} against the sample {}".format(round(((1 - face_distance) * 100),2),known_face_names[i]))
#it was used for keeps the output image or webcam as stable for certain time until the user press anything meabs close
cv2.waitKey(0)

#It bassically used when user close the image window which was shown in output after that it just close all opencv output 
cv2.destroyAllWindows()
