import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

detector = MTCNN()

# Load the FaceNet model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Load the .npz file containing the embeddings
npzfile = np.load('test-embeddings.npz')
embeddings = npzfile['embeddings']

# Create a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Get the next frame from the video
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Loop over the detected faces
    for face in faces:
        x,y,w,h = face['box']
        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]

        # Preprocess the face image
        face = cv2.resize(face, (224, 224))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        # Extract the embedding of the face
        embedding = model.predict(face)

        # Find the closest match to the embedding in the .npz file
        best_match = np.argmin(np.linalg.norm(embeddings-embedding, axis=1))

        # Display the name of the person corresponding to the closest match on the screen
        cv2.putText(frame, npzfile['names'][best_match], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame on the screen
    cv2.imshow('Face Recognition', frame)

    # If the user presses the `q` key, stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all open windows
cv2.destroyAllWindows()
