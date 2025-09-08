import streamlit as st
import os
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch for the event loop issue
nest_asyncio.apply()

# Import classes for Google Gemini and the community package for FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ==============================================================================
# PART 1: DATA LOADER CODE
# ==============================================================================
def load_agricultural_data() -> List[Dict[str, Any]]:
    """
    Load extensive agricultural knowledge data including pests, diseases,
    pest management, cultivation practices, integrated pest management, 
    nutritional recommendations, soil health guidelines, and environmental 
    stewardship, across major crops like tomato, rice, maize, wheat, 
    plus additional crops, pests and best practices.

    The content entries feature rich, multi-paragraph descriptions 
    to reach 1000+ lines when fully populated.
    """
    knowledge_base = [
        {
            "title": "Tomato Disease and Pest Management Detailed Guide",
            "content": """
Tomato Disease Management:
1. Tomato Blight (Early Blight and Late Blight):
- Symptoms: Early blight appears as brown concentric spots primarily on older leaves; late blight causes dark water-soaked lesions usually on leaves and stems accompanied by white fungal growth under moist conditions.
- Causal Agents: Alternaria solani (early blight), Phytophthora infestans (late blight).
- Epidemiology: Both thrive in humid, warm conditions from 18 to 30°C; late blight spreads rapidly during cool, wet weather.
- Management Strategy:
  A. Use certified disease-free seeds and resistant varieties like 'Roma VF' and 'Cherry VF'.
  B. Employ crop rotation and avoid planting solanaceous plants consecutively.
  C. Implement fungicide schedules starting at first symptom detection, favoring copper-based fungicides and systemic protectants.
  D. Maintain adequate spacing for airflow; prune lower leaves to prevent humidity buildup.
  E. Irrigate at soil level, avoiding leaf wetness.

2. Tomato Mosaic Virus (ToMV):
- Symptoms: Mottled light and dark green leaf patterns, leaf curling, stunting, reduced fruit yield.
- Transmission: Mechanical injury, aphids, farm tools.
- Control Measures:
  A. Destroy infected plants immediately.
  B. Clean tools and equipment regularly.
  C. Use resistant cultivars and virus-free seeds.
  D. Control aphid vector populations with insecticides or biological controls like lady beetles.

3. Fusarium Wilt:
- Characterized by vascular discoloration and wilting.
- Soilborne fungus persists in soil for several years.
- Control by soil solarization, resistant cultivars, and organic amendments.

4. Whitefly Management:
- Whiteflies cause direct feeding damage and transmit Tomato Yellow Leaf Curl Virus.
- Control through yellow sticky traps, biological agents (Encarsia spp., Eretmocerus spp.), and insecticidal soaps.

5. Nutrient Management:
- Balanced fertilization with potassium to enhance plant resistance.
- Avoid excess nitrogen which promotes vegetative growth susceptible to fungal infection.

6. Post-Harvest Disease Management:
- Proper sanitation of storage facilities.
- Use of fungicide treatments on harvested fruits before storage.

... (Expand with detailed subtopics: nematodes, bacterial spot, powdery mildew, physiological disorders, yield optimization, and climate adaptability)

""",
            "source": "Comprehensive Agricultural Extension Publications, Research Journals",
            "category": "disease_management",
            "crop": "tomato"
        },
        {
            "title": "Rice Crop Pest and Disease Management Extended Manual",
            "content": """
Pest and Disease Complex in Rice:
1. Stem Borer Biology and Control:
- Life cycle details of Yellow, Pink, Striped stem borers with diagrams.
- Action thresholds detailed for various growth stages.
- Recommended pheromone trap deployment densities.
- Resistant varieties including 'Swarna', 'IR64'.

2. Blast Disease (Magnaporthe oryzae):
- Detailed pathogen life cycle and spore dissemination.
- Symptoms on leaf, collar, neck, and panicle blast.
- Integrated management including nitrogen timing, water management, and fungicide application protocols.

3. Planthopper Management:
- Population dynamics with weather correlations.
- Use of systemic insecticides and biopesticides.
- Biological control agents descriptions including spiders and predatory bugs.

4. Nutrient Management:
- Precision fertilization to prevent excess nitrogen use promoting pests.
- Soil testing best practices.
- Use of green manures and biofertilizers like Azolla.

5. Climate Resilience:
- Flood and drought resilient varieties.
- Impact of climate change on pest lifecycles and mitigation measures.

6. Farm Mechanization and Crop Management Best Practices:
- Best timings for transplanting.
- Water management for alternate wetting and drying.
- Weed control integrated approaches including chemical and mechanical.

... (More on storage pests, post-harvest losses, grain quality preservation)

""",
            "source": "IRRI, National Plant Protection Programs, Peer-reviewed Publications",
            "category": "crop_management",
            "crop": "rice"
        },
        {
            "title": "Maize Integrated Pest Management and Cultivation Guidelines",
            "content": """
Major Maize Production Challenges:
1. Stem Borer Management:
- Detailed larval developmental stages and behavior.
- Cultural practices: destruction of crop residues to break lifecycle.
- Use of parasitoids (Cotesia flavipes) and insect growth regulators.
- Field sanitation and trap crops.

2. Leaf Blight Diseases:
- Turcicum and Maydis Leaf Blights: symptoms, pathogenic fungi characteristics.
- Fungicide types, optimal application timing and intervals.
- Use of resistant hybrids and seed treatment.

3. Earworm (Helicoverpa armigera):
- Monitoring methods including pheromone traps.
- Economic thresholds and decision-making models.
- Biological controls: Bacillus thuringiensis-based products, entomopathogenic fungi.

4. Fertilizer Recommendations:
- Soil fertility profiles for maize zones.
- Split applications of nitrogen.
- Micronutrients needed for optimal harvest.

5. Water Management:
- Irrigation scheduling for minimizing stress.
- Effect of drought stress on pest susceptibility.

6. Post-harvest Management:
- Grain drying, storage conditions, and pest prevention.

... (Detailed appendices with case studies, pest lifecycle charts, regional adaptation guides)

""",
            "source": "NCIPM Publications, Agricultural Research Journals",
            "category": "integrated_pest_management",
            "crop": "maize"
        },
        {
            "title": "Wheat Disease and Crop Management Comprehensive Guide",
            "content": """
Wheat Crop Agronomy and Protection:
1. Rust Diseases:
- Stem rust, Leaf rust, Stripe rust detailed pathogenesis.
- Resistant cultivar development.
- Surveillance and prediction models for rust epidemics.
- Fungicide program suggestions.

2. Powdery Mildew and Spot Blotch:
- Recognizing symptoms and differentiating diseases.
- Integrated management including varietal resistance and fungicide protocols.

3. Pest Management:
- Aphids, Armyworm biology and control.
- Use of pheromone traps and natural enemy augmentation.

4. Crop Nutrition:
- Macro and micronutrient balance.
- Foliar feeding and soil amendment techniques.

5. Harvest and Post-Harvest:
- Timely harvest to avoid shattering.
- Grain storage best practices to minimize losses.

6. Conservation Agriculture Practices:
- Tillage reduction, crop residue retention.
- Benefits in pest and disease suppression.

... (Long-term soil health guidelines, climate adaptation strategies, seed treatment protocols)

""",
            "source": "Wheat Research Council, Punjab Agriculture University, FAO",
            "category": "crop_management",
            "crop": "wheat"
        },
        {
            "title": "General Principles of Integrated Pest Management (IPM)",
            "content": """
IPM: Principles and Practices
- Introduction to IPM: Definition and importance in sustainable agriculture.
- Pest identification and monitoring techniques.
- Detailed explanation of Economic Threshold Levels (ETL).
- Cultural methods: crop rotation, trap crops, intercropping.
- Biological control agents: predators, parasitoids, pathogens with lifecycle illustrations.
- Chemical controls: usage guidelines to avoid resistance, safe application techniques.
- Farm safety and environmental protection practices.
- Farmer Field Schools and participatory training methods.
- Case studies on successful IPM implementation across regions and crops.
- Advances in IPM: Use of pheromones, biotechnologies, remote sensing.

... (Additional sections on legislation, pesticide registration, market access impacts)

""",
            "source": "FAO, National IPM Programs, IPM Global Initiative",
            "category": "general_management",
            "crop": "all"
        }
        # Add more crops, pest species, soil health, nutrients, climate resilience etc., to reach 1000+ lines
    ]
    return knowledge_base


# ==============================================================================
# PART 2: RAG AND AGENT SYSTEM (Simplified for Google Gemini)
# ==============================================================================
class FarmAISystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def build_knowledge_base(self, documents: List[Dict[str, Any]]):
        try:
            # Use Google's embedding model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )

            langchain_docs = [Document(page_content=doc['content'], metadata={'title': doc.get('title', '')}) for doc in documents]
            vectorstore = FAISS.from_documents(langchain_docs, embeddings)

            # Use Google's Gemini Pro model
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.1,
                google_api_key=self.api_key
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
            )
            return True
        except Exception as e:
            st.error(f"Error building knowledge base: {e}")
            return False

    def query(self, question: str):
        if not self.qa_chain:
            return "The system is not ready. Please check the logs."
        try:
            result = self.qa_chain.invoke(question)
            return result.get('result', "Sorry, I couldn't find an answer.")
        except Exception as e:
            return f"An error occurred: {e}"

# ==============================================================================
# PART 3: MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="FarmAI Knowledge Assistant", page_icon="🌾", layout="wide")
st.markdown('<h1 style="text-align: center; color: #2E8B57;">🌾 FarmAI Knowledge Assistant</h1>', unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Load data and build the RAG system."""
    # Use the new GOOGLE_API_KEY secret
    api_key = st.secrets.get("GOOGLE_API_KEY")

    if not api_key:
        return None, False

    documents = load_agricultural_data()
    farm_ai_system = FarmAISystem(api_key=api_key)
    success = farm_ai_system.build_knowledge_base(documents)

    return farm_ai_system, success

farm_ai, success = initialize_system()

if not success:
    st.error("Failed to initialize the FarmAI system. Please ensure your Google API key is correct in Streamlit Secrets and refresh the page.")
else:
    st.success("✅ FarmAI system initialized successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your farming questions today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("FarmAI is thinking..."):
                response = farm_ai.query(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
