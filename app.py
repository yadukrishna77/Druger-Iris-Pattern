import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('/content/drive/MyDrive/model.keras')

# Define the image processing functions
def convert_to_nir(image):
    nir_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nir_image = np.stack((nir_image,) * 3, axis=-1)
    return nir_image

def preprocess_image(image):
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)

# Streamlit UI
st.title("Iris Classification Model")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read and process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    nir_image = convert_to_nir(image)
    preprocessed_image = preprocess_image(nir_image)

    # Make a prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Display result
    class_names = ['Sober', 'Drunk_30min', 'Drunk_60min', 'Eye sickness']
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicted class:", class_names[predicted_class])
