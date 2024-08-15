import face_recognition
from PIL import Image,ImageDraw
# There are 68 landmark points
# 1. Load image file
face_image = face_recognition.load_image_file("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/testing/trump-modi.jpg")
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
# 5. Convert the PIL image to draw object with Alpha mode for Translucencey
d = ImageDraw.Draw(pil_image,"RGBA")
i = 0
# 6. Now after getting face landmarks points we will join all the landmarks to draw
# 7. Iterate loop to iterate through each face which contains all face landmarks 
# For each face
while i < len(face_landmarks_list):
    for face_landmarks in face_landmarks_list:
        # 8. Draw or join white line joining each face landmarks points
        
        # 9.1 Make left & right eyebrows darker 
        # 9.2 Polygon on top & line on bottom with dark color
        d.polygon(face_landmarks["left_eyebrow"] , fill = (68,54,39,128))
        d.polygon(face_landmarks["right_eyebrow"] , fill = (68,54,39,128))
        d.line(face_landmarks["left_eyebrow"] , fill = (68,54,39,150) , width=5)
        d.line(face_landmarks["right_eyebrow"] , fill = (68,54,39,150) , width=5)
        # 10. Fill right & left eyes with Red
        d.polygon(face_landmarks["left_eye"] , fill = (255,0,0,100))
        d.polygon(face_landmarks["right_eye"] , fill = (255,0,0,100))
        # 11. Eyeliner to left & right eyes as lines
        d.line(face_landmarks["left_eye"] + [face_landmarks["left_eye"][0]] , fill = (0,0,0,110) , width=6)
        d.line(face_landmarks["right_eye"] + [face_landmarks["right_eye"][0]] , fill = (0,0,0,110) , width=6)
        # 12. Add lipstick to top & bottom lips with red fill
        d.polygon(face_landmarks["top_lip"] , fill = (150,0,0,128))
        d.polygon(face_landmarks["bottom_lip"] , fill = (150,0,0,128))
        d.line(face_landmarks["top_lip"] , fill = (150,0,0,64) , width=8)
        d.line(face_landmarks["bottom_lip"] , fill = (150,0,0,64) , width=8)
    i += 1
# 13. Display the image
pil_image.show()

# You can also save a copy of the new image to disk
# pil_image.save("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/xyz.jpg") {just give file path & name of the image do you wnt to save}
pil_image.save("C:/Users/ASUS/OneDrive/Desktop/Face Recognition Project/code/images/samples/makeup_face_.jpg")

     