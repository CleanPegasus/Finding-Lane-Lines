import cv2
import numpy as np
import matplotlib.pyplot as plt



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


def reg_of_interest(image):
    #    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #    canny = cv2.Canny(blur, 50, 150)

    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1000, height), (500, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    masked_img = cv2.bitwise_and(image, mask)

    return masked_img

image = cv2.imread('test_image.jpg')
lane_img = np.copy(image)


canny = canny(lane_img)
cropped_image = reg_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
line_image = display_lines(lane_img, lines)
combo_image = cv2.addWeighted(lane_img, 0.8, line_image, 1, 1)
#cv2.imshow("result", masked_img)
plt.imshow(cropped_image)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('masked_img', 600, 600)
cv2.namedWindow('masked_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)
cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('canny', 600, 600)

cv2.imshow("result", combo_image)

cv2.imshow("image", lane_img)
cv2.imshow("canny", canny)
cv2.imshow("line_image", line_image)

cv2.waitKey(0)