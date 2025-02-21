import cv2
import os

# Create a directory to store images
dataset_path = "dataset/notmyself"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

count = 0  # Image counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract face ROI
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (200, 200))

        # Save the image
        file_path = os.path.join(dataset_path, f"face_{count}.jpg")
        cv2.imwrite(file_path, face_resized)
        count += 1

    cv2.imshow("Face Capture", frame)

    # Break if 'q' is pressed or 50 images are captured
    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Collected {count} images and saved to {dataset_path}")
