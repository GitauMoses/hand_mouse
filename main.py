import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize hand detector and drawing utility
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# Capture video and get screen dimensions
# j
#kk
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Variables for smoothing
prev_x, prev_y = 0, 0
smoothing_factor = 1.2 # Adjust for smoother movement

# Gesture control flags
click_flag = False
drag_flag = False

while True:
    # Capture frame and flip horizontally
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand tracking
    output = hand_detector.process(rgb)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw landmarks on hand
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Get coordinates for index finger and thumb
            index_finger = landmarks[8]
            thumb_tip = landmarks[4]
            middle_finger = landmarks[12]

            # Calculate coordinates on frame
            x = int(index_finger.x * frame_width)
            y = int(index_finger.y * frame_height)

            # Highlight the index finger point on the video feed
            cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)

            # Scale coordinates to screen dimensions with adjusted scaling factors
            index_x = screen_width / frame_width * x * 1.2
            index_y = screen_height / frame_height * y * 1.2

            # Apply smoothing to reduce jitter
            smoothed_x = prev_x + (index_x - prev_x) * smoothing_factor
            smoothed_y = prev_y + (index_y - prev_y) * smoothing_factor

            # Clamp coordinates to screen bounds
            smoothed_x = max(0, min(screen_width, smoothed_x))
            smoothed_y = max(0, min(screen_height, smoothed_y))

            # Move the mouse to the smoothed coordinates
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y

            # Calculate distance between index finger and thumb to detect a "click" gesture
            thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            distance = np.hypot(thumb_x - x, thumb_y - y)

            # Clicking gesture (when index finger and thumb come close together)
            if distance < 30:
                if not click_flag:
                    click_flag = True
                    pyautogui.click()  # Perform click action
            else:
                click_flag = False

            # Dragging gesture (when index and middle finger are close)
            middle_x, middle_y = int(middle_finger.x * frame_width), int(middle_finger.y * frame_height)
            drag_distance = np.hypot(middle_x - x, middle_y - y)
            if drag_distance < 40:
                if not drag_flag:
                    drag_flag = True
                    pyautogui.mouseDown()  # Start dragging
                pyautogui.moveTo(smoothed_x, smoothed_y)  # Update position while dragging
            else:
                if drag_flag:
                    pyautogui.mouseUp()  # End dragging
                    drag_flag = False

            # Scrolling gesture (move hand up or down)
            if thumb_tip.y < 0.3:
                pyautogui.scroll(5)  # Scroll up
            elif thumb_tip.y > 0.7:
                pyautogui.scroll(-5)  # Scroll down

    # Display the video feed
    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
