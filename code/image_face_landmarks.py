import face_recognition
from PIL import Image,ImageDraw
# There are 68 landmark points
# 1. Load image file
face_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/testing/E&M.jpg")
# 2. Find all face landmarks & Print the list to check
# 2.1 Find all facial landmarks in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(face_image)
# 2.2 print all face landmarks & no. of faces
print(face_landmarks_list)
# 2.3 no. of faces
print(len(face_landmarks_list))
# 4. convert the numoy array image into the pil image object & create a draw object
# convert the numoy array image into the pil image object
pil_image = Image.fromarray(face_image)
# Convert the PIL image to draw object
d = ImageDraw.Draw(pil_image)
i = 0
# 5. Now after getting face landmarks points we will join all the landmarks to draw
# 6. Iterate loop to iterate through each face which contains all face landmarks 
# For each face
while i < len(face_landmarks_list):
    for face_landmarks in face_landmarks_list:
        # 7. Draw or join white line joining each face landmarks points
        d.line(face_landmarks["chin"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["left_eyebrow"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["right_eyebrow"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["nose_bridge"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["nose_tip"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["left_eye"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["right_eye"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["top_lip"] , fill = (255,255,255) , width=2)
        d.line(face_landmarks["bottom_lip"] , fill = (255,255,255) , width=2)
    i += 1
# 8. Display the image
pil_image.show()

# You can also save a copy of the new image to disk
# pil_image.save("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/xyz.jpg") {just give file path & name of the image do you wnt to save}
pil_image.save("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/face_landmarks.jpg")

     