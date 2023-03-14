import cv2
import os
import shapefile
import numpy as np
import matplotlib.pyplot as plt
originalImage = cv2.imread('frame220.jpg')
print(originalImage.shape)
cropped = originalImage[50:350, 212:512]
grayImage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

w = shapefile.Writer('output_shapefile.shp', shapeType=shapefile.POLYLINE)

# Create fields for the shapefile attributes
w.field('ID', 'N')

# Extract the contours from the image
contours, _ = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(blackAndWhiteImage, contours, -1, (90,90,90), 3)
cv2.imshow("",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Iterate over the contours and add them as lines to the shapefile

for i, cnt in enumerate(contours):
    w.line(cnt.tolist())
    w.record(i)


# Save the shapefile
w.close()
sf = shapefile.Reader("output_shapefile.dbf")
shapes = sf.shapes()
