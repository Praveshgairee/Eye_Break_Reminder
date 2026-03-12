import cv2

# 0 usually means the default built-in camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("Press q to quit.")

while True:
    ret, frame = cap.read()

    # ret tells whether the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Show the live camera video in a window
    cv2.imshow("MacBook Camera Test", frame)

    # Wait for key press for 1 millisecond
    # If q is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera so other apps can use it
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
