from ultralytics import YOLO
import cv2
import os 
import numpy as np

model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs = 100, imgsz = 640)
metrics = model.val()  # evaluate model performance on the validation set
print(metrics)

input_image = cv2.imread("ahg_up_1_4.jpeg") #1920. 1080
input_image = cv2.resize(input_image, (300,300))
results = model(input_image)  # predict on an image
im2 = cv2.imread("ahg_up_1_4.jpeg")
#results = model.predict(source=im2, save=True, save_txt=True)
print(str(results[0].boxes))
bounding_box = results[0].boxes.xyxy
x = bounding_box[0]
x1= int(x[0])
x2 = int (x[2])
y1 = int (x[1]) 
y2= int (x[3])
padding = 50
res_plotted = results[0].plot()
cropped = input_image[y1:x2, x1:y2 + padding]
width = cropped.shape[1]
half = int (width / 2) 
left_part = cropped[:, :half]
right_part = cropped[:, half:]
#cv2.imshow("right_half", right_part)
#cv2.imshow("left_half", left_part)
rightSideFinal = np.copy(right_part)
leftSideFinal = np.copy(left_part)
#cv2.imwrite('/home/bwilab/ElevatorBoxImages/rightSideImages.jpeg', rightSideFinal)
#cv2.imwrite('/home/bwilab/ElevatorBoxImages/leftSideImages.jpeg', leftSideFinal)

#cropped = im2[363:1346, 410:659]
cv2.imshow("cropped", cropped)
cv2.imshow("result", res_plotted)
cv2.waitKey(0)