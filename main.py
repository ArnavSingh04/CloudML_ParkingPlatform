import os
import shutil
import random
from ultralytics import YOLO
from PIL import Image

# This python script just provides the bear minimum on how to load and use the model.
# You shall use the python best practices for exception handling/logging in your own code.

model = YOLO("./model/model.pt")

images_folder = "./images"
result_folder = "./result"

# This only allow jpg. Other image type png should also work with proper handling.
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(".jpg")]

random_image_filename = random.choice(image_files)
random_image_path = os.path.join(images_folder, random_image_filename)

results = model.predict(random_image_path)

result = results[0]

print(len(result.boxes))

# Printing the detected spaces
for box in result.boxes:
	label = result.names[box.cls[0].item()]
	coords = [round(x) for x in box.xyxy[0].tolist()]
	prob = round(box.conf[0].item(), 4)
	print("Object: {}\nCoordinates: {}\nProbability: {}".format(label, coords, prob))

# Copy Original image
print("Copy source image to output folder.")
shutil.copy(random_image_path,f"{result_folder}/input.jpg")

print(f"Annotating result image and save to {result_folder}/output.jpg")
Image.fromarray(result.plot()[:,:,::-1]).save(f"{result_folder}/output.jpg")

print(f"Detection completed, please view the result folder to see the results.")