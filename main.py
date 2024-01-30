import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

# initializing the stream
stream = cv2.VideoCapture("assets/vid (1).mp4")

# If we give true boolean value, this will return a debug window to help us find the color value
ball_color_finder = ColorFinder(False)
ball_color_values = {
    "hmin": 0,
    "smin": 136,
    "vmin": 0,
    "hmax": 17,
    "smax": 255,
    "vmax": 255,
}

# ball_color_values = 'red'

while True:
    # Get the image of ball
    success, image = stream.read()
    # image = cv2.imread("assets/Ball.png")

    # cropping the image frame, decreasing the height while width remain same
    image = image[:900, :]

    # detecting color ball
    image_color, mask = ball_color_finder.update(image, ball_color_values)

    # find location of ball
    image_contours, contours = cvzone.findContours(image, mask, minArea=200)

    # display the window
    image = cv2.resize(image, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", image)
    cv2.imshow("Image_Color", image_contours)
    cv2.waitKey(50)
