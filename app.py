from flask import Flask, render_template, request, redirect, url_for, flash
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import os
import json
import re
from transformers import pipeline

app = Flask(__name__)
app.secret_key = "secret-key"

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

label2id = {
    'Other': 0,
    'Policy Change': 1,
    'First Party Collection/Use': 2,
    'User Choice/Control': 3,
    'Third Party Sharing/Collection': 4,
    'Data Retention': 5,
    'Do Not Track': 6,
    'International and Specific Audiences': 7,
    'Data Security': 8,
    'User Access, Edit and Deletion': 9
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# using best model from traning (checkpoint 985)
model = BertForSequenceClassification.from_pretrained("./results/checkpoint-985", num_labels=num_labels)
model.to(device)
model.eval()

def predict_categories_bert(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = outputs.logits.argmax(dim=1).item()
    return id2label[pred_id]

# main page when u open the local host (ie the index page where u upload the privcy policy )
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        policy_text = request.form.get("policy_text", "")
        if not policy_text.strip():
            flash("Please enter a privacy policy text.")
            return redirect(url_for("index"))
        return redirect(url_for("analyze", policy_text=policy_text))
    return render_template("index.html")

# after uploading the privacy policy text it takes you to the analyze page 
# cateorgoies policy and allows for summary 
@app.route("/analyze", methods=["GET"])
def analyze():
    policy_text = request.args.get("policy_text", "")
    if not policy_text:
        flash("No policy text provided.")
        return redirect(url_for("index"))
    
    # Split the policy text into senetences
    # if we can figure out how to split by paragraph we can try that too 
    paragraphs = [s.strip() for s in re.split(r'\.\s+', policy_text) if s.strip()]
    paragraphs = [s if s.endswith('.') else s + '.' for s in paragraphs]    
    
    paragraph_predictions = []
    for para in paragraphs:
        pred = predict_categories_bert(para)
        paragraph_predictions.append((para, [pred]))
    
    found_categories = {pred for _, preds in paragraph_predictions for pred in preds}
    available_categories = sorted(list(found_categories))
    
    legend = {
        # i just foudn these descriptions online but we can change them later, just placeholders for now
        # also these are random colors we can chenge them too 
        "First Party Collection/Use": {
            "color": "#A2D5AB",
            "description": "Privacy practice describing data collection or data use by the company/organization owning the website or mobile app."
        },
        "Third Party Sharing/Collection": {
            "color": "#F5B7B1",
            "description": "Privacy practice describing data sharing with third parties or data collection by third parties. A third party is a company/organization other than the first party that owns the website or mobile app."
        },
        "User Choice/Control": {
            "color": "#E0BBE4",
            "description": "Practice that describes general choices and control options available to users."
        },
        "User Access, Edit and Deletion": {
            "color": "#FFDFBA",
            "description": "Privacy practice that allows users to access, edit or delete the data that the company/organization has about them."
        },
        "Data Retention": {
            "color": "#F0E68C",
            "description": "Privacy practice specifying the retention period for collected user information."
        },
        "Data Security": {
            "color": "#ADD8E6",
            "description": "Practice that describes how users’ information is secured and protected (e.g., encryption of stored data and communications)."
        },
        "Policy Change": {
            "color": "#F9E79F",
            "description": "The company’s practices concerning if and how users will be informed of changes to its privacy policy."
        },
        "Do Not Track": {
            "color": "#B0E0E6",
            "description": "Practices that explain if and how Do Not Track signals for online tracking and advertising are honored."
        },
        "International and Specific Audiences": {
            "color": "#C8A2C8",
            "description": "Specific audiences mentioned in the privacy policy (e.g., children or international users) for which special provisions may be provided."
        },
        "Other": {
            "color": "#D6EAF8",
            "description": "Does not relate to any specific privacy category."
        }
    }
    
    colored_paragraphs = []
    for para, preds in paragraph_predictions:
        colors = [legend.get(cat, {"color": "#CCCCCC"})["color"] for cat in preds]
        colored_paragraphs.append({
            "text": para,
            "categories": preds,
            "colors": colors
        })
    
    return render_template("analyze.html", 
                           policy_text=policy_text,
                           colored_paragraphs=colored_paragraphs,
                           available_categories=available_categories,
                           paragraphs_json=json.dumps(paragraph_predictions),
                           legend=legend)


# generates summary based on selected categories
@app.route("/summary", methods=["GET", "POST"])
def summary():
    if request.method == "GET":
        flash("Please submit the form to generate a summary.")
        return redirect(url_for("index"))
    
    selected_categories = request.form.getlist("selected_categories")
    original_text = request.form.get("policy_text")
    
    if not original_text:
        flash("No privacy policy text provided.")
        return redirect(url_for("index"))
    
    paragraphs = [para.strip() for para in original_text.split(".") if para.strip()]
    
    relevant_paragraphs = []
    for para in paragraphs:
        pred = predict_categories_bert(para)
        # only adds senetences from selected catergories 
        if pred in selected_categories:
            relevant_paragraphs.append(para)
    
    filtered_text = "\n\n".join(relevant_paragraphs)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    summary_output = summarizer(filtered_text, max_length=600, min_length=150, do_sample=False)
    summary_text = summary_output[0]['summary_text']
    
    return render_template("summary.html", summary_text=summary_text, selected_categories=selected_categories)

if __name__ == "__main__":
    app.run(debug=True)


