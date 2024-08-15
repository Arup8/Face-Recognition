from deepface import DeepFace


#face analysis
face_analysis = DeepFace.analyze(img_path="code/dataset/testing/Arup4.jpg",
                                    actions=['emotion','age','gender','race'])

print(face_analysis)
