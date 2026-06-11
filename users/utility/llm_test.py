import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# ========== 1. Configure Gemini Flash ==========
API_KEY = 'AIzaSyDAYQ5uPqYqF4C97eRnH0QlUgPb2EIIVb0'

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

def query_gemini(prompt):
    """Send query to Gemini Flash and return response text."""
    response = gemini_model.generate_content(prompt)
    return response.text

# ========== 2. Load BERT for Symptom Classification ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=5
)

# Map condition codes to human-readable labels
condition_labels = {
    0: "Respiratory infection (cold, flu, COVID-like symptoms)",
    1: "Gastrointestinal issue (food poisoning, stomach bug)",
    2: "Cardiovascular issue (mild hypertension symptoms)",
    3: "Neurological issue (headache, dizziness)",
    4: "General or unclear condition"
}

def classify_symptoms(text):
    """Classify symptoms using BERT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).detach().numpy()
    predicted_class = int(np.argmax(probs))
    return predicted_class, probs

# ========== 3. Synthetic Data Generation (Stub) ==========
def generate_synthetic_patient_data(n=10):
    """Generate random patient vitals as placeholder for GAN/CVAE output."""
    synthetic_data = []
    for i in range(n):
        synthetic_data.append({
            "heart_rate": np.random.randint(60,100),
            "bp": f"{np.random.randint(110,130)}/{np.random.randint(70,90)}",
            "symptom": "headache"
        })
    return synthetic_data

# ========== 4. Personalized Recommendation Engine ==========
def personalized_recommendation(symptoms):
    """Classify symptoms + ask Gemini for advice."""
    cls, conf = classify_symptoms(symptoms)
    condition_name = condition_labels.get(cls, "Unknown condition")

    gemini_response = query_gemini(
        f"Patient symptoms: {symptoms}. "
        f"Predicted condition: {condition_name}. "
        f"Give a brief personalized treatment advice."
    )

    return {
        "condition_code": cls,
        "condition_name": condition_name,
        "confidence": conf.tolist(),
        "recommendation": gemini_response
    }

# ========== 5. Run an Example ==========
if __name__ == "__main__":
    patient_symptoms = "persistent cough, mild fever, and fatigue"
    print("---- Personalized Recommendation ----")
    result = personalized_recommendation(patient_symptoms)
    print("Condition Code:", result["condition_code"])
    print("Condition Name:", result["condition_name"])
    print("Confidence:", result["confidence"])
    print("LLM Advice:", result["recommendation"])

    print("\n---- Synthetic Data Example ----")
    print(generate_synthetic_patient_data(5))
