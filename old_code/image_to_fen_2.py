import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

image = cv2.imread('more_chesss.jpg') # opencv reads images as BGR format

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


cv2.imshow("gray", gray_image) # matplotlib expects RGB format 

cv2.waitKey(0)

gaussian_blur = cv2.GaussianBlur(gray_image,(5,5),0)

cv2.imshow("blur", gaussian_blur) 

cv2.waitKey(0)

ret, otsu_binary = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("otsu", otsu_binary) 

cv2.waitKey(0)

canny = cv2.Canny(otsu_binary,20,255)

cv2.imshow("canny", canny) 

cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8) 
  
img_dilation = cv2.dilate(canny, kernel, iterations=1) 

cv2.imshow("img_dilation", img_dilation)

cv2.waitKey(0)

lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=50)

if lines is not None:
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        # draw lines
        cv2.line(img_dilation, (x1, y1), (x2, y2), (255,255,255), 2)

cv2.imshow("houghing around", img_dilation)

cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8) 
  
img_dilation_2 = cv2.dilate(img_dilation, kernel, iterations=1) 

cv2.imshow("second dilation", img_dilation_2)

cv2.waitKey(0)

lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=50)

if lines is not None:
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        # draw lines
        cv2.line(img_dilation_2, (x1, y1), (x2, y2), (255,255,255), 2)

cv2.imshow("more houghing around", img_dilation_2)

cv2.waitKey(0)

# find contours --> img_dilation_2
board_contours, hierarchy = cv2.findContours(img_dilation_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(board_contours))

cv2.drawContours(canny, board_contours, -1, (0,255,0), 9)

cv2.imshow("contours", canny)

cv2.waitKey(0)

square_centers=list()

# draw filtered rectangles to "canny" image for better visualization
board_squared = canny.copy()  

for contour in board_contours:
    if 4000 < cv2.contourArea(contour) < 20000:
        # Approximate the contour to a simpler shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Ensure the approximated contour has 4 points (quadrilateral)
        if len(approx) == 4:
            pts = [pt[0] for pt in approx]  # Extract coordinates

            # Define the points explicitly
            pt1 = tuple(pts[0])
            pt2 = tuple(pts[1])
            pt4 = tuple(pts[2])
            pt3 = tuple(pts[3])

            x, y, w, h = cv2.boundingRect(contour)
            center_x=(x+(x+w))/2
            center_y=(y+(y+h))/2

            square_centers.append([center_x,center_y,pt2,pt1,pt3,pt4])

             

            # Draw the lines between the points
            cv2.line(board_squared, pt1, pt2, (255, 255, 0), 7)
            cv2.line(board_squared, pt1, pt3, (255, 255, 0), 7)
            cv2.line(board_squared, pt2, pt4, (255, 255, 0), 7)
            cv2.line(board_squared, pt3, pt4, (255, 255, 0), 7)


cv2.imshow("board_squared", board_squared)
cv2.waitKey(0)