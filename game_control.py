import cv2
from pynput.keyboard import Controller
import handtrackingmodule2 as htm
import matplotlib.pyplot as plt

##############################################

# Video capture settings
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

#############################################

# Hand tracking module initialization
detector = htm.handDetector(maxHands=1, detectionCon=0.8, trackCon=0.8)

# Keyboard controller initialization
keyboard = Controller()

# Matplotlib setup
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
img_display = None  # Placeholder for the plot image

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video")
        break

    # Detect hands
    img = detector.findHands(img, draw=True)
    lmList, bbox = detector.findPosition(img)

    # Logic for finger detection
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Get the status of fingers
        fingers = detector.fingersUp()

        # Perform keyboard actions based on finger status
        if fingers[1] == 1:  # Index finger up
            keyboard.release('s')
            keyboard.press("w")
        else:
            keyboard.press("s")
            keyboard.release('w')

        if fingers[0] == 1:  # Thumb up
            keyboard.press("a")
        else:
            keyboard.release("a")

        if fingers[2] == 1:  # Middle finger up
            keyboard.press('d')
        else:
            keyboard.release('d')

    # Convert the image from BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Update or initialize the plot
    if img_display is None:
        img_display = ax.imshow(img_rgb)
        ax.axis("off")  # Hide axis for better display
    else:
        img_display.set_data(img_rgb)

    # Refresh the plot
    plt.pause(0.001)

