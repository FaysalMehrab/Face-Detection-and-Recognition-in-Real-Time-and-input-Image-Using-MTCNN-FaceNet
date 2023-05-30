import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing import image
from scipy.spatial.distance import cosine
import tensorflow as tf
from keras_facenet import FaceNet

# Load the FaceNet model
facenet_model = FaceNet()

# Load the embeddings from the .npz file
data = np.load('ss-embd.npz')
embeddings_array = data['arr_0']
names = data['arr_1']

# Load the MTCNN face detection model
mtcnn_detector = MTCNN()

# Load the test image
test_image = cv2.imread('test/mes.jpg')

# Detect faces in the test image using MTCNN
faces = mtcnn_detector.detect_faces(test_image)

# Iterate over detected faces
for face in faces:
    # Extract the face coordinates
    x, y, w, h = face['box']
    
    # Extract the face ROI from the test image
    face_roi = test_image[y:y+h, x:x+w]
    
    # Preprocess the face ROI
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_roi = cv2.resize(face_roi, (160, 160))
    face_roi = face_roi.astype('float32')
    # detections = facenet_model.extract(threshold = 0.95)
    # mean, std = face_roi.mean(), face_roi.std()
    # face_roi = (face_roi - mean) / std
    
    # Convert the preprocessed face ROI to a 4D tensor
    face_tensor = np.expand_dims(face_roi, axis=0)
    
    # Generate embeddings using the FaceNet model
    face_embedding = facenet_model.embeddings(face_tensor)[0]
    
    # Compare the face embedding with stored embeddings
    distances = [cosine(face_embedding, emb) for emb in embeddings_array]
    min_distance = min(distances)
    min_distance_idx = np.argmin(distances)
    
    # Define a threshold for face recognition
    threshold = 0.75
    
    # Check if the minimum distance is below the threshold
    if min_distance < threshold:
        # Identify the recognized face
        recognized_face = embeddings_array[min_distance_idx]
        recognized_name = names[min_distance_idx]
        
        # Perform further actions for the recognized face
        # ...
        
        # Print the recognized face name
        print("Recognized face:", recognized_name)
        confidence = 100 - min_distance*100

            # Draw bounding box and label for the recognized face with confidence level
        label = f"{recognized_name} ({confidence:.2f})"
        
        # Display the recognized face with name
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(test_image, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # The face is not recognized
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(test_image, 'Unknown face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print("Unknown face")

# Display the test image with recognized faces
cv2.imshow("Recognized Faces", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
