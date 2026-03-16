import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="API Connectivity Test", page_icon="🔌")
st.title("🔌 Google Gemini API Test")

def check_api():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ No API Key found!")
        return

    try:
        genai.configure(api_key=api_key)
        
        # We try 'gemini-pro' first as it's the most stable
        with st.spinner("Connecting to Google Gemini Pro..."):
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Is the API working? Say YES.")
            
        st.success("✅ Success! Your API key is active.")
        st.info(f"Gemini says: {response.text}")
        
    except Exception as e:
        st.error(f"❌ Connection Failed!")
        st.write("Trying secondary model...")
        try:
            # Fallback to the newest version if pro fails
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Is the API working? Say YES.")
            st.success("✅ Success with Gemini 1.5 Pro!")
        except Exception as e2:
            st.exception(e2)

if __name__ == "__main__":
    if st.button("Run Connection Test"):
        check_api()
