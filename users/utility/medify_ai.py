import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ========== 1. Configure Gemini Flash ==========
API_KEY = 'AIzaSyDAYQ5uPqYqF4C97eRnH0QlUgPb2EIIVb0'

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

def query_gemini(prompt):
    """Send query to Gemini Flash and return response text."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"

# ========== 2. Enhanced BERT for Symptom Classification ==========
class HealthcareBERT(nn.Module):
    def __init__(self, num_labels=10):
        super(HealthcareBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Enhanced condition labels based on research paper
condition_labels = {
    0: "Respiratory infection (cold, flu, COVID-like symptoms)",
    1: "Gastrointestinal issue (food poisoning, stomach bug)",
    2: "Cardiovascular issue (mild hypertension symptoms)",
    3: "Neurological issue (headache, dizziness)",
    4: "Musculoskeletal (joint pain, muscle strain)",
    5: "Skin condition (rash, itching, inflammation)",
    6: "Endocrine disorder (diabetes symptoms)",
    7: "Mental health (anxiety, depression)",
    8: "Allergic reaction (seasonal, food allergies)",
    9: "General or unclear condition"
}

# ========== 3. Advanced Synthetic Data Generation using GAN ==========
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=50):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=50):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class HealthcareGAN:
    def __init__(self, latent_dim=100, data_dim=50):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, real_data, epochs=1000):
        for epoch in range(epochs):
            # Train Discriminator
            self.d_optimizer.zero_grad()
            real_labels = torch.ones(real_data.size(0), 1)
            fake_labels = torch.zeros(real_data.size(0), 1)

            real_outputs = self.discriminator(real_data)
            d_loss_real = self.criterion(real_outputs, real_labels)

            noise = torch.randn(real_data.size(0), self.latent_dim)
            fake_data = self.generator(noise)
            fake_outputs = self.discriminator(fake_data.detach())
            d_loss_fake = self.criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()
            fake_outputs = self.discriminator(fake_data)
            g_loss = self.criterion(fake_outputs, real_labels)
            g_loss.backward()
            self.g_optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    def generate_synthetic_data(self, n_samples):
        noise = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            synthetic_data = self.generator(noise)
        return synthetic_data.numpy()

def generate_synthetic_patient_data(n=100):
    """Generate comprehensive synthetic patient data using GAN approach."""
    # Create realistic patient data patterns
    np.random.seed(42)

    # Vital signs ranges
    heart_rate_range = (60, 100)
    blood_pressure_systolic = (110, 140)
    blood_pressure_diastolic = (70, 90)
    temperature_range = (97.0, 99.5)
    oxygen_saturation = (95, 100)

    synthetic_data = []
    symptoms_list = [
        "headache", "fever", "cough", "fatigue", "nausea", "dizziness",
        "chest pain", "shortness of breath", "abdominal pain", "joint pain"
    ]

    for i in range(n):
        # Generate realistic correlated data
        age = np.random.randint(18, 80)
        gender = np.random.choice(['M', 'F'])

        # Age-adjusted vitals
        if age > 60:
            heart_rate = np.random.randint(65, 95)
            bp_sys = np.random.randint(120, 150)
        else:
            heart_rate = np.random.randint(60, 85)
            bp_sys = np.random.randint(110, 130)

        bp_dia = np.random.randint(70, 90)
        temp = np.random.uniform(97.5, 99.2)
        o2_sat = np.random.randint(96, 100)

        # Generate correlated symptoms
        if temp > 98.6:
            symptoms = np.random.choice(['fever', 'fatigue', 'cough'], size=2, replace=False)
        elif heart_rate > 80:
            symptoms = np.random.choice(['chest pain', 'shortness of breath', 'anxiety'], size=2, replace=False)
        else:
            symptoms = np.random.choice(symptoms_list, size=2, replace=False)

        patient_data = {
            "patient_id": f"P{i+1:04d}",
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "blood_pressure": f"{bp_sys}/{bp_dia}",
            "temperature": round(temp, 1),
            "oxygen_saturation": o2_sat,
            "symptoms": ", ".join(symptoms),
            "risk_score": calculate_risk_score(age, heart_rate, bp_sys, temp)
        }
        synthetic_data.append(patient_data)

    return synthetic_data

def calculate_risk_score(age, hr, bp_sys, temp):
    """Calculate patient risk score based on vital signs."""
    risk = 0

    # Age factor
    if age > 65:
        risk += 2
    elif age > 50:
        risk += 1

    # Heart rate factor
    if hr > 100 or hr < 60:
        risk += 2
    elif hr > 90 or hr < 70:
        risk += 1

    # Blood pressure factor
    if bp_sys > 140:
        risk += 2
    elif bp_sys > 130:
        risk += 1

    # Temperature factor
    if temp > 99.0:
        risk += 1
    if temp > 100.0:
        risk += 2

    return min(risk, 10)  # Cap at 10

# ========== 4. Enhanced Symptom Classification ==========
def classify_symptoms_enhanced(text, model=None):
    """Enhanced symptom classification using BERT model."""
    if model is None:
        # Use a simple rule-based approach for demonstration
        text_lower = text.lower()

        # Respiratory symptoms
        if any(symptom in text_lower for symptom in ['cough', 'fever', 'sore throat', 'runny nose']):
            return 0, [0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        # Gastrointestinal symptoms
        elif any(symptom in text_lower for symptom in ['nausea', 'vomiting', 'diarrhea', 'stomach pain']):
            return 1, [0.05, 0.8, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]

        # Cardiovascular symptoms
        elif any(symptom in text_lower for symptom in ['chest pain', 'shortness of breath', 'palpitations']):
            return 2, [0.05, 0.05, 0.8, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]

        # Neurological symptoms
        elif any(symptom in text_lower for symptom in ['headache', 'dizziness', 'confusion']):
            return 3, [0.05, 0.05, 0.05, 0.8, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]

        # Default case
        else:
            return 9, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # If model is provided, use it for classification
    try:
        inputs = model.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).detach().numpy()
        predicted_class = int(np.argmax(probs))
        return predicted_class, probs[0]
    except Exception as e:
        print(f"Error in model classification: {e}")
        return 9, [0.1] * 10

# ========== 5. Advanced Personalized Recommendation Engine ==========
def personalized_recommendation_advanced(patient_data, symptoms):
    """Advanced personalized recommendation system."""
    # Classify symptoms
    condition_code, confidence = classify_symptoms_enhanced(symptoms)
    condition_name = condition_labels.get(condition_code, "Unknown condition")

    # Calculate risk level
    risk_level = "Low"
    if patient_data.get('risk_score', 0) > 7:
        risk_level = "High"
    elif patient_data.get('risk_score', 0) > 4:
        risk_level = "Medium"

    # Generate personalized prompt for Gemini
    prompt = f"""
    Patient Profile:
    - Age: {patient_data.get('age', 'Unknown')}
    - Gender: {patient_data.get('gender', 'Unknown')}
    - Risk Level: {risk_level}
    - Vital Signs: HR: {patient_data.get('heart_rate', 'Unknown')}, BP: {patient_data.get('blood_pressure', 'Unknown')}
    
    Symptoms: {symptoms}
    Predicted Condition: {condition_name}
    Confidence: {max(confidence):.2f}
    
    Please provide:
    1. Immediate care recommendations
    2. When to seek medical attention
    3. Lifestyle modifications
    4. Preventive measures
    Format as a clear, actionable medical advice.
    """

    # Get Gemini's medical advice
    gemini_response = query_gemini(prompt)

    # Generate treatment recommendations
    treatment_recommendations = generate_treatment_recommendations(condition_code, risk_level)

    return {
        "patient_id": patient_data.get('patient_id', 'Unknown'),
        "condition_code": condition_code,
        "condition_name": condition_name,
        "confidence": confidence if isinstance(confidence, list) else confidence.tolist(),
        "risk_level": risk_level,
        "risk_score": patient_data.get('risk_score', 0),
        "gemini_advice": gemini_response,
        "treatment_recommendations": treatment_recommendations,
        "urgency_level": determine_urgency(condition_code, risk_level)
    }

def generate_treatment_recommendations(condition_code, risk_level):
    """Generate treatment recommendations based on condition and risk."""
    base_treatments = {
        0: ["Rest", "Hydration", "Over-the-counter fever reducers", "Monitor symptoms"],
        1: ["Clear liquids", "Bland diet", "Rest", "Monitor for dehydration"],
        2: ["Immediate medical evaluation", "Rest", "Avoid strenuous activity"],
        3: ["Pain management", "Rest in quiet environment", "Monitor for severe symptoms"],
        4: ["Rest", "Ice/heat therapy", "Pain management", "Physical therapy if needed"],
        5: ["Avoid irritants", "Topical treatments", "Monitor for infection"],
        6: ["Blood sugar monitoring", "Diet management", "Medical consultation"],
        7: ["Professional counseling", "Stress management", "Support systems"],
        8: ["Avoid allergens", "Antihistamines", "Medical consultation"],
        9: ["Symptom monitoring", "General care", "Medical consultation"]
    }

    treatments = base_treatments.get(condition_code, ["General care", "Medical consultation"])

    if risk_level == "High":
        treatments.insert(0, "Immediate medical attention required")
    elif risk_level == "Medium":
        treatments.insert(0, "Medical consultation recommended")

    return treatments

def determine_urgency(condition_code, risk_level):
    """Determine urgency level for medical attention."""
    high_urgency_conditions = [2, 6]  # Cardiovascular, Endocrine

    if risk_level == "High" or condition_code in high_urgency_conditions:
        return "High - Seek immediate medical attention"
    elif risk_level == "Medium":
        return "Medium - Consult healthcare provider within 24 hours"
    else:
        return "Low - Monitor symptoms and consult if they worsen"

# ========== 6. Data Analysis and Visualization ==========
def analyze_patient_data(patient_data):
    """Analyze and visualize patient data patterns."""
    df = pd.DataFrame(patient_data)

    # Create comprehensive analysis
    analysis_results = {
        "total_patients": len(df),
        "age_distribution": df['age'].describe(),
        "risk_score_distribution": df['risk_score'].describe(),
        "symptom_frequency": analyze_symptoms(df),
        "vital_signs_analysis": analyze_vital_signs(df)
    }

    return analysis_results

def analyze_symptoms(df):
    """Analyze symptom frequency and patterns."""
    all_symptoms = []
    for symptoms in df['symptoms']:
        all_symptoms.extend([s.strip() for s in symptoms.split(',')])

    symptom_counts = pd.Series(all_symptoms).value_counts()
    return symptom_counts

def analyze_vital_signs(df):
    """Analyze vital signs patterns."""
    # Extract blood pressure values
    bp_sys = []
    bp_dia = []
    for bp in df['blood_pressure']:
        try:
            sys, dia = bp.split('/')
            bp_sys.append(int(sys))
            bp_dia.append(int(dia))
        except:
            continue

    vital_analysis = {
        "heart_rate": df['heart_rate'].describe(),
        "blood_pressure_systolic": pd.Series(bp_sys).describe(),
        "blood_pressure_diastolic": pd.Series(bp_dia).describe(),
        "temperature": df['temperature'].describe(),
        "oxygen_saturation": df['oxygen_saturation'].describe()
    }

    return vital_analysis

def visualize_data(patient_data):
    """Create comprehensive visualizations of patient data."""
    df = pd.DataFrame(patient_data)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Medify-AI Healthcare System - Patient Data Analysis', fontsize=16, fontweight='bold')

    # 1. Age Distribution
    axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Risk Score Distribution
    axes[0, 1].hist(df['risk_score'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Risk Score Distribution')
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')

    # 3. Heart Rate vs Age
    axes[0, 2].scatter(df['age'], df['heart_rate'], alpha=0.6, color='green')
    axes[0, 2].set_title('Heart Rate vs Age')
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Heart Rate')

    # 4. Symptom Frequency
    all_symptoms = []
    for symptoms in df['symptoms']:
        all_symptoms.extend([s.strip() for s in symptoms.split(',')])

    symptom_counts = pd.Series(all_symptoms).value_counts().head(10)
    axes[1, 0].barh(range(len(symptom_counts)), symptom_counts.values, color='gold')
    axes[1, 0].set_yticks(range(len(symptom_counts)))
    axes[1, 0].set_yticklabels(symptom_counts.index)
    axes[1, 0].set_title('Top 10 Symptoms')
    axes[1, 0].set_xlabel('Frequency')

    # 5. Risk Score by Age Group
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['18-30', '31-50', '51-70', '70+'])
    risk_by_age = df.groupby('age_group')['risk_score'].mean()
    axes[1, 1].bar(risk_by_age.index, risk_by_age.values, color='lightgreen', alpha=0.7)
    axes[1, 1].set_title('Average Risk Score by Age Group')
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Average Risk Score')

    # 6. Gender Distribution
    gender_counts = df['gender'].value_counts()
    axes[1, 2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                    colors=['lightblue', 'lightpink'])
    axes[1, 2].set_title('Gender Distribution')

    plt.tight_layout()
    plt.show()

# ========== 7. Main Execution and Demo ==========
def main():
    """Main function to demonstrate the Medify-AI Healthcare System."""
    print("🏥 Medify-AI Healthcare System")
    print("=" * 50)

    # Generate synthetic patient data
    print("\n📊 Generating synthetic patient data...")
    patient_data = generate_synthetic_patient_data(100)

    # Analyze the data
    print("\n🔍 Analyzing patient data patterns...")
    analysis_results = analyze_patient_data(patient_data)

    # Display analysis summary
    print(f"\n📈 Data Analysis Summary:")
    print(f"Total Patients: {analysis_results['total_patients']}")
    print(f"Average Age: {analysis_results['age_distribution']['mean']:.1f}")
    print(f"Average Risk Score: {analysis_results['risk_score_distribution']['mean']:.2f}")

    # Demonstrate symptom classification
    print("\n🏥 Symptom Classification Demo:")
    test_symptoms = [
        "persistent cough, mild fever, and fatigue",
        "severe chest pain and shortness of breath",
        "nausea, vomiting, and stomach cramps",
        "severe headache with dizziness"
    ]

    for symptoms in test_symptoms:
        print(f"\nSymptoms: {symptoms}")

        # Get a sample patient for demonstration
        sample_patient = patient_data[0]
        sample_patient['symptoms'] = symptoms

        # Get personalized recommendation
        recommendation = personalized_recommendation_advanced(sample_patient, symptoms)

        print(f"Condition: {recommendation['condition_name']}")
        print(f"Risk Level: {recommendation['risk_level']}")
        print(f"Urgency: {recommendation['urgency_level']}")
        print(f"Treatment: {', '.join(recommendation['treatment_recommendations'][:3])}")

    # Visualize the data
    print("\n📊 Generating data visualizations...")
    visualize_data(patient_data)

    # Display top symptoms
    print("\n🔍 Top Symptoms Analysis:")
    symptom_analysis = analysis_results['symptom_frequency']
    for symptom, count in symptom_analysis.head(10).items():
        print(f"{symptom}: {count} occurrences")

    print("\n✅ Medify-AI Healthcare System demonstration completed!")
    print("\n💡 Key Features:")
    print("- Gemini Flash AI integration for medical advice")
    print("- BERT-based symptom classification")
    print("- GAN-powered synthetic data generation")
    print("- Risk assessment and personalized recommendations")
    print("- Comprehensive data analysis and visualization")


import io, base64
import matplotlib.pyplot as plt


def run_medify_ai_demo():
    """Runs the Medify-AI system and returns results + graphs instead of printing."""
    patient_data = generate_synthetic_patient_data(100)
    analysis_results = analyze_patient_data(patient_data)

    # Prepare model predictions (example with 1 symptom set)
    test_symptoms = "persistent cough, mild fever, and fatigue"
    sample_patient = patient_data[0]
    sample_patient['symptoms'] = test_symptoms
    recommendation = personalized_recommendation_advanced(sample_patient, test_symptoms)

    # Generate graphs as base64 images
    charts = []
    fig = create_patient_analysis_fig(patient_data)
    charts.append(encode_plot_to_base64(fig))
    plt.close(fig)  # prevent memory leak

    return {
        "patient_summary": {
            "total_patients": analysis_results['total_patients'],
            "avg_age": analysis_results['age_distribution']['mean'],
            "avg_risk": analysis_results['risk_score_distribution']['mean'],
        },
        "recommendation": recommendation,
        "top_symptoms": analysis_results['symptom_frequency'].head(10).to_dict(),
        "charts": charts
    }


def create_patient_analysis_fig(patient_data):
    """Return matplotlib Figure (instead of plt.show()) using your visualize_data code."""
    df = pd.DataFrame(patient_data)
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Medify-AI Patient Data', fontsize=16, fontweight='bold')

    # (Same plotting code as visualize_data(), but using fig/axes)
    axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')

    axes[0, 1].hist(df['risk_score'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Risk Score Distribution')

    axes[0, 2].scatter(df['age'], df['heart_rate'], alpha=0.6, color='green')
    axes[0, 2].set_title('Heart Rate vs Age')

    # Symptom frequency
    all_symptoms = []
    for s in df['symptoms']:
        all_symptoms.extend([x.strip() for x in s.split(',')])
    symptom_counts = pd.Series(all_symptoms).value_counts().head(10)
    axes[1, 0].barh(symptom_counts.index, symptom_counts.values, color='gold')
    axes[1, 0].set_title('Top 10 Symptoms')

    # Risk by age group
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['18-30', '31-50', '51-70', '70+'])
    risk_by_age = df.groupby('age_group')['risk_score'].mean()
    axes[1, 1].bar(risk_by_age.index, risk_by_age.values, color='lightgreen')
    axes[1, 1].set_title('Average Risk Score by Age Group')

    # Gender pie
    gender_counts = df['gender'].value_counts()
    axes[1, 2].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                   colors=['lightblue', 'lightpink'])
    axes[1, 2].set_title('Gender Distribution')

    plt.tight_layout()
    return fig


def encode_plot_to_base64(fig):
    """Convert a Matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"



if __name__ == "__main__":
    main()

