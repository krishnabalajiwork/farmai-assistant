import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="API Connectivity Test", page_icon="🔌")
st.title("🔌 Google Gemini API Test")

def check_api():
    # 1. Check if the secret exists
    api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("❌ No API Key found! Please add 'GOOGLE_API_KEY' to your Streamlit Secrets.")
        return

    try:
        # 2. Configure the API
        genai.configure(api_key=api_key)
        
        # 3. Try to list models (this checks if the key is valid)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 4. Attempt a simple generation
        with st.spinner("Testing connection to Google..."):
            response = model.generate_content("Is the API working? Answer with 'YES' and one short farming tip.")
            
        st.success("✅ API Connection Successful!")
        st.write("---")
        st.subheader("Response from Gemini:")
        st.info(response.text)
        
    except Exception as e:
        st.error(f"❌ Connection Failed!")
        st.exception(e)

if __name__ == "__main__":
    if st.button("Run Connection Test"):
        check_api()
    else:
        st.info("Click the button above to test your GOOGLE_API_KEY.")
