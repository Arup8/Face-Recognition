import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
# 1. load image
image_to_detect = cv2.imread('C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/E&M.jpg')

#2. Initialize the models & load weights
#face expression model initialization
face_exp_model = model_from_json(open("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/facial_expression_model_structure.json","r").read())
#load weights into the model
face_exp_model.load_weights("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/dataset/facial_expression_model_weights.h5")
#list(array) of emotion labels
# "0" th index means "angry" "1" th index means "disgust" like this way
emotions_label = ("angry" , "disgust" , "fear" , "happy" , "sad" , "surprise" , "neutral")

# 2. show the image as output terminal

cv2.imshow("test", image_to_detect)

#
'''3. Now after the loaded image just detect how many faces on that image by using "hog" or "cnn" model.
Now "hog" model is faster and "cnn" model is slower so it takes time to give output so here we used "hog" model'''
#
# Detect all faces in the image
# find all face locations using "face_locations()" function which was included in "face_recognition" module function
# model can be "cnn" or "hog"
# arguments are image,number_of_times_to_upsample,model
 
all_face = face_recognition.face_locations(image_to_detect , model = 'hog')

# 4. Now just print the no of faces as output so here "{}" this is replaced by le(all_face)
# Basically here "{}" this replaced by the value ".format(.......)"

print("There are {} no of face(s) in this image".format(len(all_face)))

# 5. Now after detecting the no of faces in main images just find positions of all faces
#Just run a for loop with index(i) & each face position with condition of the size of tuple(all_face)

for i, curr_face in enumerate(all_face):

    # 6. Splitting the tupple to get the 4 position values of current faces(curr_face)
    # "top,right,bottom,left" this order cause it is in clock wise direction 

    top_pos, right_pos, bottom_pos, left_pos = curr_face

    # 7. Now printing the current face locattions of the main images

    print("Found face {} at top : {}, right = {}, bottom = {}, left = {}".format(i+1, top_pos, right_pos, bottom_pos, left_pos))

    # 8. Now after finding the locations of the faces just store that locations in array
    # Means slicing the faces from main images
    
    curr_face_img = image_to_detect[top_pos: bottom_pos, left_pos: right_pos]

    # 13. Draw rectangle around each face location in the main video frame inside the loop 
    # Draw rectangle around the face detected 
    # Now basically our image in "curr_frame" cause we resized our image by step 9 
    # "(0,0,255)" means (B[Blue],G[Green],R[Red]) and last value 2 means border thickness
    cv2.rectangle(image_to_detect , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)

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
    cv2.putText(image_to_detect , emotion_label , (left_pos , bottom_pos) , font , 0.5 , (255 , 255 , 255) , 1)

            
#16. Displaying the current frame outside for loop inside the while loop
#Showing the current face with rectangle drawn
cv2.imshow("Image Face Emotion Detection : ",image_to_detect)

#it was used for keeps the output image or webcam as stable for certain time until the user press anything meabs close
cv2.waitKey(0)

#It bassically used when user close the image window which was shown in output after that it just close all opencv output 
cv2.destroyAllWindows()
