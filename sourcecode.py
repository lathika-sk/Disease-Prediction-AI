!pip install gradio fpdf --quiet

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gradio as gr
from fpdf import FPDF
import datetime
import os

# 📄 Load Data
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)
X = df.iloc[:, :-1]
y = df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

print(f"✅ Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

# 🔍 Disease Info
disease_info = {
    'Fungal infection': {
        "risk": "Low",
        "desc": "A fungal infection causes irritation, scaly skin, redness, etc.",
        "action": "Use antifungal cream and maintain hygiene"
    },
    'Allergy': {
        "risk": "Medium",
        "desc": "Allergy is an immune response to substances not typically harmful.",
        "action": "Avoid allergens and take antihistamines"
    },
    # ➕ Add more diseases here
}

symptoms = X.columns.tolist()

# 🧠 Prediction Logic
def predict_disease(user_name, *user_symptoms):
    input_vector = [1 if s else 0 for s in user_symptoms]
    input_scaled = scaler.transform([input_vector])
    pred_probs = model.predict_proba(input_scaled)[0]
    top3_idx = np.argsort(pred_probs)[-3:][::-1]

    results = []
    for i in top3_idx:
        disease = le.inverse_transform([i])[0]
        confidence = pred_probs[i] * 100
        info = disease_info.get(disease, {
            "risk": "Unknown", "desc": "No description available", "action": "Consult a doctor"
        })
        results.append({
            "Disease": disease,
            "Confidence": f"{confidence:.2f}%",
            "Risk Level": info["risk"],
            "Description": info["desc"],
            "Action": info["action"]
        })

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    selected_symptoms = ", ".join([s for s, val in zip(symptoms, user_symptoms) if val])
    history_entry = pd.DataFrame([{
        "Name": user_name,
        "DateTime": timestamp,
        "Disease": results[0]["Disease"],
        "Confidence": results[0]["Confidence"],
        "Symptoms": selected_symptoms
    }])
    if os.path.exists("prediction_history.csv"):
        history_entry.to_csv("prediction_history.csv", mode='a', header=False, index=False)
    else:
        history_entry.to_csv("prediction_history.csv", index=False)

    # 🧾 Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(230, 230, 250)  # Lavender
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.cell(200, 10, txt=f"AI Disease Prediction Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Patient Name: {user_name}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Date: {timestamp}", ln=True, align="L")
    pdf.ln(5)
    for res in results:
        for key, val in res.items():
            pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)
        pdf.ln(4)
    pdf.output("disease_report.pdf")

    summary = f"# 🧠 Prediction Summary for {user_name}\n"
    for res in results:
        summary += f"### 🩺 {res['Disease']}\n"
        summary += f"- 🔢 Confidence: {res['Confidence']}\n"
        summary += f"- ⚠️ Risk Level: {res['Risk Level']}\n"
        summary += f"- 📖 {res['Description']}\n"
        summary += f"- 💊 Advice: {res['Action']}\n\n"

    return summary, "disease_report.pdf"

# 📊 View History Function
def view_history():
    if os.path.exists("prediction_history.csv"):
        return pd.read_csv("prediction_history.csv")
    else:
        return pd.DataFrame(columns=["Name", "DateTime", "Disease", "Confidence", "Symptoms"])

# 🎨 Custom Styling (Lavender + Black)
theme_css = """
body { background-color: lavender; color: black; }
.gradio-container { background-color: lavender !important; color: black !important; }
"""

# 🌐 Interface Setup
inputs = [gr.Textbox(label="Your Name")] + [gr.Checkbox(label=s) for s in symptoms]

interface = gr.Interface(
    fn=predict_disease,
    inputs=inputs,
    outputs=[gr.Markdown(label="Prediction Summary"), gr.File(label="Download PDF")],
    title="🧠 AI Disease Predictor",
    description="Select your symptoms and receive an AI-powered diagnosis with risk level, advice, and a downloadable report.",
    theme="default",
    css=theme_css
)

history_ui = gr.Interface(
    fn=view_history,
    inputs=[],
    outputs="dataframe",
    title="📊 Prediction History",
    description="See all your past predictions."
)

# 🎯 Launch App
gr.TabbedInterface([interface, history_ui], ["Disease Prediction", "Prediction History"]).launch()
