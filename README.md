# -Face-Recognition
1.INTRODUCTION:
Face recognition is a biometric technology that identifies or verifies an individual's identity by analyzing and comparing facial features from an image or video. It is widely used across industries for security, authentication, and personalization purposes.Google Teachable Machine is a user-friendly, web-based tool that allows anyone to train machine learning models without coding expertise. It is ideal for beginners exploring face recognition.

2.OVERVIEW OF GOOGLE TEACHABLE MACHINE:
Google Teachable Machine is a web-based platform that democratizes machine learning by allowing users to create, train, and deploy models without requiring coding or advanced technical expertise. It's particularly suited for beginners who want to explore machine learning concepts, including face recognition, in an interactive and accessible manner.

3.Key Features
Detects faces in images and in real-time too.
Utilizes a pre-trained model from Google Teachable Machine.
Beginner-friendly and easy to customize.

4.Steps to Build a Face Recognition Model The following steps were taken to create a face recognition model:
Data Collection: Captured multiple images of faces under various conditions to improve accuracy. Categorized images into classes for training (e.g., Person A, Person B).
![Screenshot_14-1-2025_161025_teachablemachine withgoogle com](https://github.com/user-attachments/assets/fc131e1a-904c-49a1-b513-98d1b7709014)
Model Training: Imported images into Teachable Machine. Adjusted parameters (e.g., epochs, learning rate) to optimize performance. Trained the model within the platform, receiving real-time feedback on accuracy. 
![Screenshot_14-1-2025_162124_teachablemachine withgoogle com](https://github.com/user-attachments/assets/8c62bfb3-0666-44f6-a3ae-529049f939b3)
Testing: Validated the model with unseen face images. Evaluated accuracy, precision, and recall metrics. 
![image](https://github.com/user-attachments/assets/1fb4d061-c971-4188-9ded-0b270975bcd1)
Exporting and Deployment: Exported the trained model in TensorFlow.js format. Integrated the model into a Python project using tensorflow or JavaScript for web applications.
Advantages
Free and open for everyone.
No need for advanced hardware or technical expertise.
Encourages experimentation and creativity in AI.
Prerequisites
Python 3.7 or higher
OpenCV library
NumPy library
A pre-trained face model from Google Teachable Machine

Import Required Libraries python from keras.models import load_model # TensorFlow required for Keras import cv2 # For accessing the webcam and processing images import numpy as np # For numerical operations load_model: Loads a pre-trained deep learning model for prediction. cv2 (OpenCV): Handles webcam input and image display. numpy: Facilitates image preprocessing, such as resizing and normalizing.

Disable Scientific Notation np.set_printoptions(suppress=True) Prevents numpy from displaying numbers in scientific notation, ensuring more readable output.

Load the Pre-trained Model and Labels model = load_model("C:/Users/Drishty/PycharmProjects/PythonProject/face_recogonization/model.h5", compile=False) class_names = open("labels.txt", "r").readlines() Model: A deep learning model trained for face recognition. It predicts the probability of the input belonging to each class. Labels: Contains the class names corresponding to the model’s output. For example, labels.txt might look like: 0 John 1 Mary 2 Alice

Access the Webcam camera = cv2.VideoCapture(0) Opens the webcam for capturing live video feed. The 0 specifies the default webcam. Change to 1 if using an external camera.

Main Loop for Real-time Prediction 5.1 Capture and Resize Webcam Image ret, image = camera.read() image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) cv2.imshow("Webcam Image", image) Captures a single frame (image) from the webcam. Resizes it to 224x224 pixels, matching the model's expected input dimensions. Displays the resized image in a window named "Webcam Image".

5.2 Preprocess the Image image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3) image = (image / 127.5) - 1 Converts the image into a numpy array and reshapes it to the format (1, 224, 224, 3) (batch size of 1). Normalizes pixel values to the range [-1, 1] for compatibility with the model.

5.3 Make Predictions prediction = model.predict(image) index = np.argmax(prediction) class_name = class_names[index] confidence_score = prediction[0][index] model.predict: Predicts probabilities for each class. np.argmax: Finds the class with the highest probability. Retrieves the corresponding class name and its confidence score.

5.4 Display Results print("Class:", class_name[2:], end="") print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%") Prints the recognized class name (skipping the first 2 characters for formatting) and the confidence score as a percentage.

Keyboard Input for Exit keyboard_input = cv2.waitKey(1) if keyboard_input == 27: # ESC key break Waits for keyboard input. Terminates the loop if the ESC key (ASCII 27) is pressed.

Release Resources camera.release() cv2.destroyAllWindows() Releases the webcam resource. Closes all OpenCV windows opened during execution.

Output Example Console Output: yaml Class: John Confidence Score: 98 % Webcam Feed: Displays a real-time video feed, resized for prediction. Key Features Real-time Face Recognition: Continuously captures frames and processes them for predictions. Pre-trained Model Integration: Uses a .h5 model trained for face recognition. Confidence Display: Outputs the name of the recognized person with the associated confidence score. Interactive Exit: Allows termination via the ESC key.
