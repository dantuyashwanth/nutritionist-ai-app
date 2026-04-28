import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Configuration & API Setup
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_gemini_response(prompt_template, image_data, user_input):
    """
    Calls the Gemini API using the latest supported model.
    """
    # Updated to the latest model to fix the 404 error
    model = genai.GenerativeModel("gemini-3-flash-preview") 
    
    # Passing components: [Template, Image Part, Additional user text]
    response = model.generate_content([prompt_template, image_data[0], user_input])
    return response.text

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for the API."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

def main():
    # 2. Streamlit UI Setup
    st.set_page_config(page_title="Nutritionist-Food-Recognition-APP", page_icon="🍲")
    st.header("Your Dietitian and Nutritionist")
    
    selected_language = st.selectbox("Select Language:", ["English"])
    
    # Prompt Definitions
    if selected_language == "English":
        input_prompt1 = "Identify this dish, its origins, and list ingredients pointwise."
        input_prompt2 = "Provide a step-by-step cooking guide and expert tips for this dish."
        input_prompt3 = "Provide nutritional tables for this dish (Calories, Protein, Fat, Carbs)."
        input_prompt4 = "Suggest 2 vegetarian and 2 non-vegetarian alternatives with similar nutrition."

    input_text = st.text_input("Additional Instructions (Optional): ", key="input")
    uploaded_file = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)
        
    col1, col2 = st.columns(2)
    submit1 = col1.button("Get Dish Name & Ingredients")
    submit2 = col1.button("How to Cook")
    submit3 = col2.button("Nutritional Value")
    submit4 = col2.button("Similar Alternatives")

    # 3. Logic Handling
    button_map = {
        submit1: input_prompt1,
        submit2: input_prompt2,
        submit3: input_prompt3,
        submit4: input_prompt4
    }

    for button, prompt in button_map.items():
        if button:
            if uploaded_file is not None:
                with st.spinner("Analyzing image..."):
                    try:
                        image_data = input_image_setup(uploaded_file)
                        response = get_gemini_response(prompt, image_data, input_text)
                        st.subheader("The Response is")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload a dish image first.")

    st.markdown("---")
    st.caption("Disclaimer: AI-generated data is an estimate. Consult a professional for medical advice.")

if __name__ == "__main__":
    main()
