import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tkinter import Tk, filedialog

plt.ion()  # Turn on interactive mode

# Load YOLO model
yolo_model = YOLO("last.pt")

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 's' to capture webcam image, 'u' to upload image, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Predict on live webcam frame
    results = yolo_model.predict(frame, conf=0.5)
    annotated_frame = results[0].plot()

    # Convert to RGB for matplotlib
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Show live frame
    plt.imshow(frame_rgb)
    plt.title("Live Webcam Detection")
    plt.axis('off')
    plt.pause(0.001)
    plt.clf()

    # Ask user for input
    key = input("Press 's' to capture, 'u' to upload image, 'q' to quit: ").strip().lower()

    if key == 's':
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured and saved as 'captured_image.jpg'.")
        break

    elif key == 'u':
        Tk().withdraw()  # Hide the root Tkinter window
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            upload_img = cv2.imread(file_path)

            if upload_img is None:
                print("Failed to load the selected image.")
                continue

            # Predict on uploaded image
            results = yolo_model.predict(upload_img, conf=0.5)
            upload_annotated = results[0].plot()
            upload_rgb = cv2.cvtColor(upload_annotated, cv2.COLOR_BGR2RGB)

            # Display result
            plt.imshow(upload_rgb)
            plt.title("Detection on Uploaded Image")
            plt.axis('off')
            plt.show(block=True)  # Wait until user closes it
        else:
            print("No image selected.")

    elif key == 'q':
        print("Exiting.")
        break

# Release resources
cap.release()
plt.ioff()
plt.close()
