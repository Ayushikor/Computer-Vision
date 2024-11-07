# install Library
# pip install opencv-python
# pip install cvzone
# pip install numpy

# Import Libraries
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np


# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width of webcam
cap.set(4, 720)   # Height of webcam


# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)


# Variables for drawing
brush_thickness = 15
eraser_thickness = 50
draw_color = (255, 0, 0)  # Blue by default
canvas = None
xp, yp = 0, 0  # Previous points


# Color palette (for changing colors)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0


# Create a canvas for drawing
while cap.isOpened():
    # Read image from the webcam
    success, img = cap.read()
    if not success:
        break


    # Flip the image to avoid mirror effect
    img = cv2.flip(img, 1)


    # Create the canvas if not initialized
    if canvas is None:
        canvas = np.zeros_like(img)


    # Detect hand and landmarks
    hands, img = detector.findHands(img, flipType=False)

    if hands:

        hand = hands[0]
        lmList = hand["lmList"]


        x1, y1 = lmList[8][:2]



        fingers = detector.fingersUp(hand)


        # If only the index finger is up - draw
        if fingers[1] == 1 and all(f == 0 for f in fingers[2:]):
            # Eraser mode if the middle finger is also up
            if fingers[2] == 1:
                cv2.circle(img, (x1, y1), eraser_thickness, (0, 0, 0), cv2.FILLED)
                cv2.circle(canvas, (x1, y1), eraser_thickness, (0, 0, 0), cv2.FILLED)
            else:
                # Brush mode - Draw on the canvas
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                xp, yp = x1, y1


        # If all fingers are up - change color (reset position)
        if all(f == 1 for f in fingers):
            color_index = (color_index + 1) % len(colors)
            draw_color = colors[color_index]
            xp, yp = 0, 0  # Reset drawing start position


        # If no fingers are up - reset previous point (not drawing)
        if fingers[1] == 0:
            xp, yp = 0, 0



    # Combine the original image with the canvas
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    # Display the result
    cv2.imshow("Hand Gesture Paint", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
