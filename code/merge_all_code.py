import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# Load face recognition data
Arup_image = face_recognition.load_image_file("code/images/samples/Arup.jpg")
Arup_face_encodings = face_recognition.face_encodings(Arup_image)[0]
# load next Sample Image & Extract the Face Encoding
Arup1_image = face_recognition.load_image_file("code/images/samples/Arup2.jpg")
Arup1_face_encodings = face_recognition.face_encodings(Arup1_image)[0]
# load next Sample Image & Extract the Face Encoding
elon_image = face_recognition.load_image_file("code/images/samples/E.jpg")
elon_face_encodings = face_recognition.face_encodings(elon_image)[0]
# load next Sample Image & Extract the Face Encoding
mark_image = face_recognition.load_image_file("code/images/samples/Mark Zuckerberg.webp")
mark_face_encodings = face_recognition.face_encodings(mark_image)[0]
# load next Sample Image & Extract the Face Encoding 
modi_image = face_recognition.load_image_file("code/images/samples/modi.jpg")
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]
# load next Sample Image & Extract the Face Encoding
trump_image = face_recognition.load_image_file("code/images/samples/trump.jpg")
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]
# load next Sample Image & Extract the Face Encoding
shahrukh_image = face_recognition.load_image_file("code/images/samples/shah-rukh-khan.jpg")
shahrukh_face_encodings = face_recognition.face_encodings(shahrukh_image)[0]
# load next Sample Image & Extract the Face Encoding
shahrukh1_image = face_recognition.load_image_file("code/images/testing/Shah-Rukh-Khan.jpg")
shahrukh1_face_encodings = face_recognition.face_encodings(shahrukh1_image)[0]
# load next Sample Image & Extract the Face Encoding
rdj_image = face_recognition.load_image_file("code/images/samples/RDJ.jpg")
rdj_face_encodings = face_recognition.face_encodings(rdj_image)[0]
# load next Sample Image & Extract the Face Encoding
rdj1_image = face_recognition.load_image_file("code/images/samples/RDJ1.jpg")
rdj1_face_encodings = face_recognition.face_encodings(rdj1_image)[0]
# load next Sample Image & Extract the Face Encoding
rdj2_image = face_recognition.load_image_file("code/images/samples/RDJ2.jpg")
rdj2_face_encodings = face_recognition.face_encodings(rdj2_image)[0]
# load next Sample Image & Extract the Face Encoding
kohli_image = face_recognition.load_image_file("code/images/samples/Virat_Kohli.jpg")
kohli_face_encodings = face_recognition.face_encodings(kohli_image)[0]
# load next Sample Image & Extract the Face Encoding
msd_image = face_recognition.load_image_file("code/images/samples/MSD.jpg")
msd_face_encodings = face_recognition.face_encodings(msd_image)[0]
# load next Sample Image & Extract the Face Encoding
tom_image = face_recognition.load_image_file("code/images/samples/Tom_Holland.jpg")
tom_face_encodings = face_recognition.face_encodings(tom_image)[0]
# load next Sample Image & Extract the Face Encoding 
tobey_image = face_recognition.load_image_file("code/images/samples/Tobey_Maguire.jpg")
tobey_face_encodings = face_recognition.face_encodings(tobey_image)[0]
# load next Sample Image & Extract the Face Encoding
andrew_image = face_recognition.load_image_file("code/images/samples/Andrew Garfield.jpg")
andrew_face_encodings = face_recognition.face_encodings(andrew_image)[0]
# load next Sample Image & Extract the Face Encoding
sachin_image = face_recognition.load_image_file("code/images/samples/Sachin.jpg")
sachin_face_encodings = face_recognition.face_encodings(sachin_image)[0]
# load next Sample Image & Extract the Face Encoding
henry_image = face_recognition.load_image_file("code/images/samples/Henry_Cavill.jpg")
henry_face_encodings = face_recognition.face_encodings(henry_image)[0]

# 3. Create an array to save the encodings
known_face_encodings = [Arup_face_encodings , Arup1_face_encodings , elon_face_encodings , mark_face_encodings , modi_face_encodings , trump_face_encodings , shahrukh_face_encodings , shahrukh1_face_encodings , rdj_face_encodings , rdj1_face_encodings , rdj2_face_encodings , kohli_face_encodings , msd_face_encodings , tom_face_encodings , tobey_face_encodings , andrew_face_encodings , sachin_face_encodings , henry_face_encodings]
# 4. Create another array to hold labels
# Save the encodings & corresponding labels in separate arrays in the same order
known_face_names = ["Arup" , "Arup" , "Elon Musk" , "Mark Zuckerberg" , "Narendra Modi" , "Donald Trump" , "Shah Rukh Khan" , "Shah Rukh Khan" , "Robert Downey Jr." , "Robert Downey Jr." , "Robert Downey Jr." , "Virat Kohli" , "MS Dhoni" , "Tom Holland" , "Tobey Maguire" , "Andrer Garfield" , "Sachin Tendulkar" , "Henry Cavill"]



# Load gender and age detection models
gender_label_list = ["Male", "Female"]
gender_protext = "code/dataset/gender_deploy.prototxt"
gender_caffemodel = "code/dataset/gender_net.caffemodel"

age_label_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
age_protext = "code/dataset/age_deploy.prototxt"
age_caffemodel = "code/dataset/age_net.caffemodel"

face_exp_model = model_from_json(open("code/dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights("code/dataset/facial_expression_model_weights.h5")
emotions_label = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Get the default WebCam Video
# webcam_video_stream = cv2.VideoCapture(0)
webcam_video_stream = cv2.VideoCapture("code/images/Robert Downey Jr.mp4")

while True:
    ret, curr_frame = webcam_video_stream.read()
    curr_frame_small = cv2.resize(curr_frame, (0, 0), fx=0.25, fy=0.25)

    all_face_locations = face_recognition.face_locations(curr_frame_small, number_of_times_to_upsample=2, model='hog')
    all_face_encodings = face_recognition.face_encodings(curr_frame_small, all_face_locations)

    for curr_face_location, curr_face_encoding in zip(all_face_locations, all_face_encodings):
        top_pos, right_pos, bottom_pos, left_pos = curr_face_location
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4

        all_matches = face_recognition.compare_faces(known_face_encodings, curr_face_encoding)
        name_of_person = "Unknown Face"

        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

        cv2.rectangle(curr_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 85, 255), 2)

        curr_face_img = curr_frame[top_pos: bottom_pos, left_pos: right_pos]

        curr_face_img_gray = cv2.cvtColor(curr_face_img, cv2.COLOR_BGR2GRAY)
        curr_face_img_gray = cv2.resize(curr_face_img_gray, (48, 48))
        img_pixels = image.img_to_array(curr_face_img_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        emotion_predictions = face_exp_model.predict(img_pixels)
        emotion_label = emotions_label[np.argmax(emotion_predictions)]

        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(cv2.dnn.blobFromImage(curr_face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False))
        gender_predictions = gender_cov_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]

        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(cv2.dnn.blobFromImage(curr_face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False))
        age_predictions = age_cov_net.forward()
        age = age_label_list[age_predictions[0].argmax()]

        font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(curr_frame, f"{name_of_person}, {gender}, {age}yrs, {emotion_label}", (left_pos + 6, bottom_pos - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(curr_frame, f"Name: {name_of_person}", (left_pos + 6, bottom_pos + 20), font, 0.8, (0, 0, 255), 1)
        # Display age
        cv2.putText(curr_frame, f"Age: {age} years", (left_pos + 6, bottom_pos + 40), font, 0.8, (255, 0, 0), 1)
        # Display gender
        cv2.putText(curr_frame, f"Gender: {gender}", (left_pos + 6, bottom_pos + 60), font, 0.8, (0, 255, 0), 1)
        # Display emotion
        cv2.putText(curr_frame, f"Emotion: {emotion_label}", (left_pos + 6, bottom_pos + 79), font, 0.8, (255, 255, 255), 1)
    cv2.imshow("Integrated Face Recognition", curr_frame)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()

