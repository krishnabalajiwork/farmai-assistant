import os
import json
from typing import List, Dict, Any
import requests
import logging

def load_agricultural_data() -> List[Dict[str, Any]]:
    """Load agricultural knowledge data from various sources"""
    documents = []
    
    # Load built-in agricultural knowledge
    documents.extend(load_builtin_knowledge())
    
    # Load from data files if available
    documents.extend(load_from_files())
    
    # If no documents loaded, use fallback data
    if not documents:
        documents = get_fallback_data()
    
    logging.info(f"Loaded {len(documents)} agricultural knowledge documents")
    return documents

def load_builtin_knowledge() -> List[Dict[str, Any]]:
    """Load comprehensive built-in agricultural knowledge"""
    knowledge_base = [
        {
            "title": "Tomato Disease Management Guide",
            "content": """
            Common Tomato Diseases and Management:
            
            1. Tomato Blight (Early and Late Blight):
            - Symptoms: Brown spots on leaves with concentric rings (early blight) or dark lesions with white fuzzy growth (late blight)
            - Causes: Fungal infections (Alternaria solani for early blight, Phytophthora infestans for late blight)
            - Management: Apply copper-based fungicides, ensure good air circulation, avoid overhead watering, rotate crops
            - Prevention: Plant resistant varieties, maintain proper spacing, remove infected plant debris
            
            2. Tomato Mosaic Virus:
            - Symptoms: Mottled yellow and green leaves, stunted growth, reduced fruit production
            - Causes: Viral infection spread by insects or contaminated tools
            - Management: Remove infected plants, control insect vectors, use virus-resistant varieties
            - Prevention: Sanitize tools, control aphids and whiteflies, avoid touching plants when wet
            
            3. Bacterial Wilt:
            - Symptoms: Sudden wilting of plants during hot weather, brown discoloration in stem vascular tissue
            - Causes: Bacterial infection (Ralstonia solanacearum)
            - Management: Remove and destroy infected plants, improve soil drainage, use bacterial-resistant varieties
            - Prevention: Crop rotation, avoid overwatering, maintain soil pH between 6.0-7.0
            
            4. Fusarium Wilt:
            - Symptoms: Yellowing of lower leaves, wilting during hot weather, brown streaks in stem
            - Causes: Soil-borne fungus (Fusarium oxysporum)
            - Management: Use resistant varieties, improve soil drainage, apply beneficial microorganisms
            - Prevention: Crop rotation with non-solanaceous plants, soil solarization
            """,
            "source": "Agricultural Extension Service",
            "category": "disease_management",
            "crop": "tomato"
        },
        {
            "title": "Rice Cultivation Best Practices",
            "content": """
            Rice Cultivation Guidelines:
            
            1. Land Preparation:
            - Plow field 2-3 times after harvesting previous crop
            - Apply organic matter (2-3 tons per hectare)
            - Level the field properly for uniform water distribution
            - Create bunds to retain water
            
            2. Seed Selection and Treatment:
            - Use certified seeds of high-yielding varieties
            - Treat seeds with fungicide (Carbendazim 2g per kg seed)
            - Soak seeds for 24 hours before sowing
            - Use 20-25 kg seeds per hectare for direct seeding
            
            3. Water Management:
            - Maintain 2-5 cm water level during vegetative stage
            - Increase to 5-10 cm during reproductive stage
            - Drain field 15-20 days before harvest
            - Alternate wetting and drying for water conservation
            
            4. Nutrient Management:
            - Apply NPK in ratio 120:60:40 kg per hectare
            - Split nitrogen application: 50% at planting, 25% at tillering, 25% at panicle initiation
            - Apply phosphorus and potassium as basal dose
            - Use micronutrient spray if deficiency symptoms appear
            
            5. Pest and Disease Management:
            - Monitor for brown plant hopper, stem borer, and leaf folder
            - Apply integrated pest management practices
            - Use pheromone traps for stem borer control
            - Spray fungicide for blast and sheath blight diseases
            """,
            "source": "Rice Research Institute",
            "category": "crop_management",
            "crop": "rice"
        },
        {
            "title": "Wheat Disease Identification and Control",
            "content": """
            Major Wheat Diseases:
            
            1. Wheat Rust (Yellow, Brown, Black):
            - Symptoms: Rust-colored pustules on leaves and stems
            - Yellow rust: Yellow stripes on leaves
            - Brown rust: Brown pustules scattered on leaves
            - Black rust: Black pustules on stems
            - Management: Apply fungicides (Propiconazole), use resistant varieties
            - Prevention: Early sowing, balanced fertilization, crop rotation
            
            2. Wheat Blast:
            - Symptoms: Bleached wheat heads, shriveled grains
            - Causes: Fungal infection (Magnaporthe oryzae)
            - Management: Apply fungicides at heading stage, harvest early
            - Prevention: Use resistant varieties, avoid late planting
            
            3. Powdery Mildew:
            - Symptoms: White powdery growth on leaves and stems
            - Causes: Fungal infection favored by high humidity
            - Management: Apply sulfur-based fungicides, improve air circulation
            - Prevention: Plant resistant varieties, avoid overcrowding
            
            4. Septoria Leaf Spot:
            - Symptoms: Small brown spots with dark borders on leaves
            - Causes: Fungal infection during wet conditions
            - Management: Apply fungicides, remove crop residues
            - Prevention: Crop rotation, avoid overhead irrigation
            """,
            "source": "Wheat Research Institute",
            "category": "disease_management",
            "crop": "wheat"
        },
        {
            "title": "Integrated Pest Management Strategies",
            "content": """
            Integrated Pest Management (IPM) Principles:
            
            1. Prevention:
            - Use resistant crop varieties
            - Maintain crop rotation
            - Practice good field sanitation
            - Time planting to avoid peak pest periods
            
            2. Monitoring and Identification:
            - Regular field scouting for pests and beneficial insects
            - Use pheromone traps for early detection
            - Identify economic threshold levels
            - Keep detailed pest monitoring records
            
            3. Biological Control:
            - Encourage natural enemies (predators and parasites)
            - Use beneficial insects like ladybugs and lacewings
            - Apply microbial pesticides (Bt, NPV)
            - Plant banker plants to support beneficial insects
            
            4. Cultural Control:
            - Adjust planting dates to avoid pest outbreaks
            - Use trap crops to divert pests
            - Implement proper irrigation management
            - Remove crop residues that harbor pests
            
            5. Chemical Control (Last Resort):
            - Use selective pesticides that target specific pests
            - Rotate pesticides with different modes of action
            - Follow label instructions and safety guidelines
            - Apply only when economic threshold is reached
            
            6. Organic Alternatives:
            - Neem oil for soft-bodied insects
            - Diatomaceous earth for crawling insects
            - Soap sprays for aphids and mites
            - Essential oil-based repellents
            """,
            "source": "IPM Guidelines",
            "category": "pest_management",
            "crop": "general"
        },
        {
            "title": "Soil Health and Fertility Management",
            "content": """
            Soil Health Management:
            
            1. Soil Testing and Analysis:
            - Test soil pH, nutrient levels, and organic matter content
            - Conduct tests every 2-3 years or before major crops
            - Test for micronutrients if deficiency symptoms appear
            - Maintain soil pH between 6.0-7.5 for most crops
            
            2. Organic Matter Management:
            - Add compost or well-rotted manure (5-10 tons per hectare)
            - Practice green manuring with leguminous crops
            - Retain crop residues to improve soil structure
            - Use cover crops during fallow periods
            
            3. Nutrient Management:
            - Follow soil test recommendations for fertilizer application
            - Use balanced NPK fertilizers based on crop requirements
            - Apply micronutrients (Zn, B, Fe) if soil tests indicate deficiency
            - Time fertilizer application to match crop uptake patterns
            
            4. Physical Soil Health:
            - Avoid soil compaction by limiting heavy machinery use
            - Practice minimum tillage to preserve soil structure
            - Create proper drainage to prevent waterlogging
            - Add organic amendments to improve water retention
            
            5. Biological Soil Health:
            - Encourage beneficial soil microorganisms
            - Apply mycorrhizal inoculants for better nutrient uptake
            - Use organic fertilizers to feed soil biology
            - Avoid excessive use of chemical pesticides
            
            6. Erosion Control:
            - Plant cover crops on slopes
            - Create terraces and contour farming on sloped land
            - Maintain vegetation buffers near water bodies
            - Use mulching to protect soil surface
            """,
            "source": "Soil Science Department",
            "category": "soil_management",
            "crop": "general"
        },
        {
            "title": "Climate-Smart Agriculture Practices",
            "content": """
            Climate-Smart Agriculture Strategies:
            
            1. Water Conservation:
            - Implement drip irrigation systems for efficient water use
            - Practice rainwater harvesting and storage
            - Use mulching to reduce evaporation
            - Apply deficit irrigation strategies for drought tolerance
            
            2. Heat Stress Management:
            - Plant heat-tolerant varieties
            - Provide shade structures for sensitive crops
            - Adjust planting times to avoid extreme temperatures
            - Use cooling techniques like misting systems
            
            3. Drought Resilience:
            - Select drought-resistant crop varieties
            - Improve soil water retention through organic matter
            - Practice conservation tillage to reduce water loss
            - Implement early warning systems for drought conditions
            
            4. Flood Management:
            - Create proper drainage systems
            - Plant flood-tolerant varieties in prone areas
            - Use raised bed cultivation in flood-prone regions
            - Implement early warning systems for flooding
            
            5. Carbon Sequestration:
            - Practice agroforestry to capture carbon
            - Use cover crops to build soil organic matter
            - Reduce tillage to prevent carbon loss
            - Apply compost and organic amendments
            
            6. Adaptation Strategies:
            - Diversify crops to spread climate risks
            - Use weather-based crop insurance
            - Implement flexible planting schedules
            - Adopt precision agriculture technologies
            """,
            "source": "Climate Change Research Center",
            "category": "climate_adaptation",
            "crop": "general"
        }
    ]
    
    return knowledge_base

def load_from_files() -> List[Dict[str, Any]]:
    """Load data from external files if available"""
    documents = []
    data_dir = "data"
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            documents.extend(file_data)
                        elif isinstance(file_data, dict):
                            documents.append(file_data)
                except Exception as e:
                    logging.warning(f"Error loading {filename}: {str(e)}")
    
    return documents

def get_fallback_data() -> List[Dict[str, Any]]:
    """Provide fallback data if no other sources are available"""
    return [
        {
            "title": "Basic Agricultural Principles",
            "content": """
            Fundamental principles of agriculture include proper soil preparation, 
            appropriate seed selection, timely planting, adequate water management, 
            balanced nutrition, and integrated pest management. Success in farming 
            requires understanding local climate conditions, soil characteristics, 
            and market demands.
            """,
            "source": "Agricultural Basics",
            "category": "general",
            "crop": "general"
        }
    ]

def validate_document(doc: Dict[str, Any]) -> bool:
    """Validate document structure"""
    required_fields = ['title', 'content']
    return all(field in doc for field in required_fields)

def enrich_document_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Add default metadata to documents"""
    defaults = {
        'source': 'Agricultural Knowledge Base',
        'category': 'general',
        'crop': 'general',
        'language': 'en'
    }
    
    for key, value in defaults.items():
        if key not in doc:
            doc[key] = value
    
    return doc