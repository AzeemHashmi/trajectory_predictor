import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

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

position_list_x, position_list_y = [], []

# image width is 1300
xList = [item for item in range(0, 1300)]

prediction = False


# initializing the stream
stream = cv2.VideoCapture("assets/vid (4).mp4")

while True:
    # Get the image of ball
    success, image = stream.read()
    # image = cv2.imread("assets/Ball.png")

    # cropping the image frame, decreasing the height while width remain same
    image = image[:900, :]

    # detecting color ball
    image_color, mask = ball_color_finder.update(image, ball_color_values)

    # find location of ball
    image_contours, contours = cvzone.findContours(image, mask, minArea=500)

    # contour function has sorted func in it so the biggest contour will be on zero index.
    if contours:
        position_list_x.append(contours[0]["center"][0])
        position_list_y.append(contours[0]["center"][1])

    # polynomial regression --> y = Ax^2 + Bx + C
    # finding coeficients A, B, C
    # 2 is for quadratic eq, if it cubical then it would be 3
    if position_list_x:
        A, B, C = np.polyfit(position_list_x, position_list_y, 2)

        # zip is used to run 2 loops at same time, we are traversing 2 list here at same time
        for i, (posX, posY) in enumerate(zip(position_list_x, position_list_y)):
            pos = (posX, posY)
            # it will draw circle in the center of ball
            cv2.circle(image_contours, pos, 10, (0, 255, 0), cv2.FILLED)

            # it will draw connecting line between points
            if i == 0:
                cv2.line(image_contours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(
                    image_contours,
                    pos,
                    (position_list_x[i - 1], position_list_y[i - 1]),
                    (0, 255, 0),
                    2,
                )

        # generating predictive circles
        for x in xList:
            y = int(A * x**2 + B * x + C)
            cv2.circle(image_contours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        # now to confirm the ball will be basket or not, we have determine that basket position in image is
        # 330 to 430 on x-axis and 590 on y-axis
        # as now we have fixed value of y, we will now calculate x from y
        # search solve quadratic expression for x
        if len(position_list_x) < 10:
            a = A
            b = B
            c = C - 590

            x = int((-b - math.sqrt(b**2 - (4 * a * c))) / (2 * a))

            prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(
                image_contours,
                "Basket",
                (50, 100),
                scale=5,
                thickness=5,
                colorR=(0, 200, 0),
                offset=20,
            )
        else:
            cvzone.putTextRect(
                image_contours,
                "No Basket",
                (50, 100),
                scale=5,
                thickness=5,
                colorR=(0, 0, 200),
                offset=20,
            )

    # display the window
    image = cv2.resize(image, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", image)
    cv2.imshow("Image_Color", image_contours)
    cv2.waitKey(60)
