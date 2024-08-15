import cv2
import face_recognition
# 1. load image
image_to_detect = cv2.imread('C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/trump-modi.jpg')

# 2. show the image as output terminal

# cv2.imshow("test", image_to_detect)

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
    cv2.rectangle(image_to_detect , (left_pos , top_pos) , (right_pos , bottom_pos) , (0 , 85 , 255) , 2)
    # 22. Print the gender & age under the rectangle box
    # Display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX  #font style
    # By using cv2.puttext() funtion we can give any name or anything
    cv2.putText(image_to_detect , gender+" "+age+"yrs" , (left_pos , bottom_pos + 20) , font , 0.5 , (0 , 255 , 0) , 1)
#23. Displaying the current frame outside for loop inside the while loop
#Showing the current face with rectangle drawn
cv2.imshow("Image Age & Gender prediction : ",image_to_detect)

#it was used for keeps the output image or webcam as stable for certain time until the user press anything meabs close
cv2.waitKey(0)

#It bassically used when user close the image window which was shown in output after that it just close all opencv output 
cv2.destroyAllWindows()
