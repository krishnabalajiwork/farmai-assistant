import google.generativeai as genai
import streamlit as st

def test_google_connection():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: No API Key found in Streamlit Secrets.")
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test prompt
        response = model.generate_content("Is the API working? Answer with 'YES' and a farming fact.")
        print(f"✅ Success! Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_google_connection()
