from deepface import DeepFace


# detector_backend = "opencv", "ssd", "dlib", "mtcnn", "retinaface"
# model_name = "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace"
# distance_metric = "cosine", "euclidean", "euclidean_l2"


#face recognition
face_recognition = DeepFace.find(img_path="code/dataset/testing/tom1.jpg",
                                    db_path="code/dataset/training",
                                    detector_backend="opencv",
                                    model_name="VGG-Face",
                                    distance_metric="cosine",
                                    enforce_detection=False)

print(face_recognition)
## After running that the "representations_vgg_face.pkl" named file will save so make sure before running the program delete that fil