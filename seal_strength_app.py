import streamlit as st
from transformers import pipeline
import numpy as np

st.set_page_config(page_title="Seal Strength Predictor with AI Assistant", layout="wide")
st.title("🔩 Seal Strength Predictor")

# Sidebar for GenAI Assistant
st.sidebar.title("🤖 AI Assistant")
st.sidebar.markdown("Ask questions about the model, inputs, or interpretation of results.")

@st.cache_resource
def load_genai_pipeline():
    return pipeline("text-generation", model="distilgpt2")

genai = load_genai_pipeline()

user_query = st.sidebar.text_area("Ask me anything:", height=100)
if st.sidebar.button("Get Help"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            response = genai(user_query, max_length=100, do_sample=True)[0]['generated_text']
            st.sidebar.markdown("**AI Response:**")
            st.sidebar.write(response)
    else:
        st.sidebar.warning("Please enter a question.")

# Main UI for model input
st.subheader("Enter Material Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=300.0, value=100.0)
    pressure = st.number_input("Pressure (kPa)", min_value=50.0, max_value=500.0, value=150.0)

with col2:
    dwell_time = st.number_input("Dwell Time (s)", min_value=0.1, max_value=10.0, value=2.0)
    material_thickness = st.number_input("Material Thickness (mm)", min_value=0.01, max_value=5.0, value=0.5)

def predict_seal_strength(temp, press, time, thickness):
    return 0.8 * np.log(temp + 1) + 0.5 * np.sqrt(press) + 1.2 * time - 0.3 * thickness**2

if st.button("Predict Seal Strength"):
    result = predict_seal_strength(temperature, pressure, dwell_time, material_thickness)
    st.success(f"🔍 Predicted Seal Strength: {result:.2f} N")
    st.markdown("You can ask the AI Assistant in the sidebar to help interpret this result or suggest improvements.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and HuggingFace Transformers")
