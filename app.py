import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
import streamlit as st
from dotenv import dotenv_values, load_dotenv
from PIL import Image, UnidentifiedImageError


APP_TITLE = "AI Food Calorie Tracker"
BASE_DIR = Path(__file__).resolve().parent
PREFERRED_GEMINI_MODELS = (
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.0-flash-lite",
)


def preprocess_image(uploaded_file, max_size: Tuple[int, int] = (1024, 1024)) -> Tuple[Image.Image, bytes]:
    """Validate, resize, and convert an uploaded/captured image to JPEG bytes."""
    if uploaded_file is None:
        raise ValueError("Please upload or capture a food image first.")

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The selected file is not a valid image. Please upload a JPG, JPEG, or PNG file.") from exc

    image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return image, buffer.getvalue()


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI using weight in kilograms and height in meters."""
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    if weight_kg <= 0:
        raise ValueError("Weight must be greater than zero.")
    return weight_kg / (height_m**2)


def classify_bmi(bmi: float) -> str:
    """Classify BMI into standard adult BMI categories."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _get_api_key() -> str:
    """Read Gemini API key, preferring the local project .env file."""
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path, override=True)
    file_key = dotenv_values(env_path).get("GOOGLE_API_KEY", "") if env_path.exists() else ""

    try:
        secret_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        secret_key = ""

    return (file_key or secret_key or os.getenv("GOOGLE_API_KEY", "")).strip().strip('"').strip("'")


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON from Gemini output, including responses wrapped in markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "dish_name": "Could not confidently parse dish name",
            "ingredients": [],
            "nutrition": {"calories": "N/A", "protein": "N/A", "fat": "N/A", "carbohydrates": "N/A"},
            "recipe": cleaned,
            "alternatives": [],
            "notes": "Gemini returned free-form text instead of structured JSON.",
        }


def _select_gemini_model() -> str:
    """Choose the first available Gemini model that supports generateContent."""
    available_models = {
        model.name
        for model in genai.list_models()
        if "generateContent" in getattr(model, "supported_generation_methods", [])
    }

    for model_name in PREFERRED_GEMINI_MODELS:
        if model_name in available_models:
            return model_name

    gemini_models = sorted(name for name in available_models if "gemini" in name.lower())
    if gemini_models:
        return gemini_models[0]

    raise RuntimeError("No Gemini model that supports image analysis was found for this API key.")


def generate_ai_response(image_bytes: bytes, user_context: str = "") -> Dict[str, Any]:
    """Send the preprocessed food image to Gemini Vision and return structured analysis."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key is missing. Add GOOGLE_API_KEY to your .env file or Streamlit secrets.")

    genai.configure(api_key=api_key)
    selected_model = _select_gemini_model()
    model = genai.GenerativeModel(selected_model)

    prompt = f"""
You are an expert nutritionist and food recognition assistant.
Analyze the food image and return only valid JSON with this exact schema:
{{
  "dish_name": "string",
  "confidence": "High/Medium/Low",
  "ingredients": ["ingredient 1", "ingredient 2"],
  "nutrition": {{
    "calories": "estimated kcal per serving",
    "protein": "estimated grams per serving",
    "fat": "estimated grams per serving",
    "carbohydrates": "estimated grams per serving"
  }},
  "recipe": ["step 1", "step 2", "step 3"],
  "alternatives": ["alternative 1", "alternative 2", "alternative 3"],
  "portion_assumption": "brief serving-size assumption"
}}

Keep nutrition estimates realistic, mention uncertainty when the image is unclear,
and do not include markdown outside the JSON.
Additional user context: {user_context or "None"}
"""

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
        ]
    )

    if not response or not getattr(response, "text", None):
        raise RuntimeError("Gemini did not return a response. Please try another image.")

    return _extract_json(response.text)


def _nutrition_number(value: Any) -> float:
    """Best-effort extraction of the first number from a nutrition field."""
    if isinstance(value, (int, float)):
        return float(value)
    digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in str(value))
    for token in digits.split():
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def generate_recommendation(bmi_category: str, food_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create a personalized recommendation from BMI category and food nutrition estimates."""
    nutrition = food_analysis.get("nutrition", {})
    calories = _nutrition_number(nutrition.get("calories", 0))
    fat = _nutrition_number(nutrition.get("fat", 0))
    carbs = _nutrition_number(nutrition.get("carbohydrates", 0))
    protein = _nutrition_number(nutrition.get("protein", 0))

    score = 0
    reasons: List[str] = []

    if calories >= 700:
        score += 2
        reasons.append("high estimated calories")
    elif calories >= 450:
        score += 1
        reasons.append("moderate-to-high estimated calories")

    if fat >= 30:
        score += 1
        reasons.append("high estimated fat")
    if carbs >= 80:
        score += 1
        reasons.append("high estimated carbohydrates")
    if protein >= 20:
        score -= 1
        reasons.append("good estimated protein")

    if bmi_category in {"Overweight", "Obese"}:
        score += 1
    elif bmi_category == "Underweight":
        score -= 1

    if score <= 0:
        verdict = "Recommended"
    elif score <= 2:
        verdict = "Moderate consumption"
    else:
        verdict = "Avoid"

    modifications = [
        "Reduce oil, butter, cream, or cheese where possible.",
        "Keep the portion size controlled and pair it with a salad or cooked vegetables.",
        "Choose grilled, baked, steamed, or air-fried preparation instead of deep-fried.",
        "Use whole grains, lean protein, and low-fat dairy alternatives when they fit the dish.",
        "Limit sugary drinks or desserts with this meal to keep total calories balanced.",
    ]

    if bmi_category == "Underweight":
        modifications.append("Add nutrient-dense sides such as lentils, yogurt, nuts, or eggs for healthy weight gain.")
    if bmi_category in {"Overweight", "Obese"}:
        modifications.append("Prioritize half-plate vegetables and avoid second servings of refined carbs.")

    return {
        "verdict": verdict,
        "reason": ", ".join(reasons) if reasons else "balanced estimated nutrition for your BMI profile",
        "modifications": modifications,
    }


def _list_or_text(items: Any) -> None:
    """Render lists cleanly, with graceful fallback for text."""
    if isinstance(items, list) and items:
        for item in items:
            st.markdown(f"- {item}")
    elif items:
        st.write(items)
    else:
        st.caption("Not available")


def _render_card(title: str, value: str, help_text: str = "") -> None:
    st.markdown(f"**{title}**")
    st.markdown(f"<div class='metric-card'>{value}</div>", unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":fork_and_knife:", layout="wide")

    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 2rem; max-width: 1180px;}
        .metric-card {
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #ffffff;
            color: #111827;
            font-size: 1.05rem;
            font-weight: 600;
            min-height: 3.1rem;
        }
        .recommendation {
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #d7e4d0;
            background: #f6fbf3;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.caption("Personalized Gemini-powered food recognition, calorie estimation, and nutrition guidance.")

    with st.sidebar:
        st.header("Your Profile")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Female", "Male", "Other / Prefer not to say"])
        height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01, format="%.2f")
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")

        bmi = calculate_bmi(weight, height)
        bmi_category = classify_bmi(bmi)
        st.divider()
        st.metric("BMI", f"{bmi:.1f}", bmi_category)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Food Image")
        input_mode = st.radio("Image source", ["Upload image", "Use camera"], horizontal=True)
        uploaded_file = (
            st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])
            if input_mode == "Upload image"
            else st.camera_input("Capture food image")
        )
        user_context = st.text_area(
            "Optional notes",
            placeholder="Example: homemade, one bowl, extra cheese, less oil, restaurant serving...",
            height=90,
        )

    image = None
    image_bytes = None
    if uploaded_file is not None:
        try:
            image, image_bytes = preprocess_image(uploaded_file)
            with left:
                st.image(image, caption="Image ready for Gemini analysis", use_column_width=True)
        except ValueError as exc:
            st.error(str(exc))

    with right:
        st.subheader("Analysis")
        analyze = st.button("Analyze Food", type="primary", use_container_width=True)

        if analyze:
            if image_bytes is None:
                st.warning("Please upload or capture a valid food image before analysis.")
                return

            profile_context = (
                f"Age: {age}, Gender: {gender}, Height: {height:.2f} m, "
                f"Weight: {weight:.1f} kg, BMI: {bmi:.1f} ({bmi_category}). {user_context}"
            )

            with st.spinner("Gemini is analyzing the food and estimating nutrition..."):
                try:
                    analysis = generate_ai_response(image_bytes, profile_context)
                except Exception as exc:
                    st.error(f"AI analysis failed: {exc}")
                    st.info("Check your Gemini API key, internet connection, and image quality, then try again.")
                    return

            st.session_state["analysis"] = analysis
            st.session_state["recommendation"] = generate_recommendation(bmi_category, analysis)

    analysis = st.session_state.get("analysis")
    recommendation = st.session_state.get("recommendation")

    if analysis:
        st.divider()
        st.subheader("Detected Food")
        confirm_col, confidence_col = st.columns([2, 1])
        detected_name = analysis.get("dish_name", "Unknown dish")
        with confirm_col:
            confirmed_food = st.text_input("Confirm or edit detected food", value=detected_name)
        with confidence_col:
            _render_card("Gemini confidence", str(analysis.get("confidence", "N/A")))

        nutrition = analysis.get("nutrition", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _render_card("Calories", str(nutrition.get("calories", "N/A")))
        with c2:
            _render_card("Protein", str(nutrition.get("protein", "N/A")))
        with c3:
            _render_card("Fat", str(nutrition.get("fat", "N/A")))
        with c4:
            _render_card("Carbohydrates", str(nutrition.get("carbohydrates", "N/A")))

        st.caption(f"Portion assumption: {analysis.get('portion_assumption', 'Not provided')}")

        tab1, tab2, tab3, tab4 = st.tabs(["Ingredients", "Recipe", "Alternatives", "Personalized Guidance"])

        with tab1:
            st.markdown(f"### {confirmed_food} ingredients")
            _list_or_text(analysis.get("ingredients"))

        with tab2:
            st.markdown("### How to cook")
            _list_or_text(analysis.get("recipe"))

        with tab3:
            st.markdown("### Similar food alternatives")
            _list_or_text(analysis.get("alternatives"))

        with tab4:
            st.markdown("### BMI summary")
            b1, b2 = st.columns(2)
            with b1:
                _render_card("BMI value", f"{bmi:.1f}")
            with b2:
                _render_card("BMI category", bmi_category)

            if recommendation:
                st.markdown("### Recommendation")
                st.markdown(
                    f"<div class='recommendation'><strong>{recommendation['verdict']}</strong><br>"
                    f"Reason: {recommendation['reason']}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("### Healthy Modification Suggestions")
                _list_or_text(recommendation["modifications"])

    st.divider()
    st.caption(
        "Disclaimer: AI nutrition estimates can be inaccurate and are not medical advice. "
        "Consult a qualified professional for medical or diet-specific decisions."
    )


if __name__ == "__main__":
    main()
