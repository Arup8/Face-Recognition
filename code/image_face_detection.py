import cv2
import face_recognition
# 1. load image
image_to_detect = cv2.imread('C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/E&M.jpg')

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

    #9. Showing the current face with dynamic title(different name) which was givn as concatenation of "str(i+1)"
    
    cv2.imshow("Face no " + str(i+1), curr_face_img)

#it was used for keeps the output image or webcam as stable for certain time until the user press anything meabs close
cv2.waitKey(0)

#It bassically used when user close the image window which was shown in output after that it just close all opencv output 
cv2.destroyAllWindows()
