import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# -----------------------------------
# Load environment variables
# -----------------------------------
load_dotenv()

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------------
# App Header & Description
# -----------------------------------
st.title("🤖 AI Document Assistant")
st.write("Ask any question and get an AI-generated answer powered by Google Gemini.")

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ **Gemini API key not found.** Please add `GEMINI_API_KEY` to your `.env` file.")
    st.stop()

# Candidate models order to prevent quota or deprecation failures
CANDIDATE_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.0-flash-lite"
]

def generate_response(prompt: str) -> tuple[str, str]:
    """Generates content using the google.genai SDK with automatic model fallbacks."""
    client = genai.Client(api_key=api_key)
    last_exception = None

    for model_name in CANDIDATE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text, model_name
        except Exception as e:
            last_exception = e
            continue

    raise last_exception or RuntimeError("All candidate models failed to generate content.")

# -----------------------------------
# User Input & Interaction
# -----------------------------------
question = st.text_input(
    "Ask me anything:",
    placeholder="Type your question here..."
)

if st.button("Generate Answer", type="primary"):
    if question.strip():
        try:
            with st.spinner("Generating answer..."):
                answer, model_used = generate_response(question)

            st.subheader("Answer")
            st.caption(f"Model used: `{model_used}`")
            st.write(answer)

        except APIError as err:
            st.error(f"API Error ({err.code}): {err.message}")
            if "RESOURCE_EXHAUSTED" in str(err) or err.code == 429:
                st.info("💡 Free tier API rate limit reached. Please wait a moment and try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question first.")