from deepface import DeepFace

result = DeepFace.verify(img1_path="1.png", img2_path="2.png")

print(result)
