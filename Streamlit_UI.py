import faiss
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load the Keras model
model = load_model('/Users/sridharkandi/Desktop/3rd Sem/Capstone/Saved_Models/h5_final_model.h5')

# Class names
class_names = [
    "Apple___Apple_scab", "Strawberry___Leaf_scorch", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew", 
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy",
    "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Apple___Black_rot",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Load LLM data
with open('/Users/sridharkandi/Desktop/3rd Sem/Capstone/llm_data.txt', 'r') as file:
    llm_data = [line.strip() for line in file if line.strip()]

# Validate LLM data
assert len(llm_data) > 0, "LLM data is empty. Please check the file content."

# Set up OpenAI API key
openai.api_key = 'OPENAI_API_KEY'

# Convert llm_data to embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(llm_data)

# Create a FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Function to preprocess the uploaded image
# def preprocess_image(uploaded_file):
#     image = Image.open(uploaded_file).convert('RGB')
#     image = image.resize((224, 224))  # Resize to the model's input size
#     image_array = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    # Open the file-like object as a PIL image
    image = Image.open(uploaded_file).convert('RGB')
    # Resize the image to the expected size
    image = image.resize((224, 224))
    # Normalize pixel values and convert to NumPy array
    image_array = np.array(image) / 255.0  # Normalized to [0, 1]
    # Add a batch dimension (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array


# Function to predict the class of the image
def predict_image_class(model, image_array):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

# Function to query the FAISS index and OpenAI LLM
def query_llm(prompt, disease_name):
    try:
        query_embedding = embedding_model.encode([prompt])
        _, indices = index.search(query_embedding, 5)
        relevant_texts = [llm_data[i] for i in indices[0]]
        context = "\n".join(relevant_texts[:1500])  # Limit context length

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are an expert in {disease_name} and can answer questions based on your knowledge and additional provided data."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": context}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error during LLM query: {e}")
        return "Unable to fetch response."

# Streamlit app
st.set_page_config(layout="wide")
st.title("PlantGuard: AI Disease Detection & Real-Time Plant Advice")

uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        with st.spinner('Classifying...'):
            image_array = preprocess_image(uploaded_file)
            predicted_class, confidence = predict_image_class(model, image_array)
            predicted_class_name = class_names[predicted_class]
            st.success(f"The classified disease is: {predicted_class_name}")
            # st.info(f"Confidence: {confidence:.2f}")

    with col2:
        st.write(f"You can ask questions related to {predicted_class_name} below.")
        if 'question_history' not in st.session_state:
            st.session_state['question_history'] = []

        for i, (q, r) in enumerate(st.session_state['question_history']):
            st.text_input(f"Question {i+1}", value=q, disabled=True)
            st.text_area(f"Response {i+1}", value=r, disabled=True)

        new_query = st.text_input("Ask a new question:")
        if new_query:
            with st.spinner('Fetching response...'):
                response = query_llm(new_query, predicted_class_name)
                st.session_state['question_history'].append((new_query, response))
                st.write("**LLM Response:**")
                st.write(response)