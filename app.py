from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

app = Flask(__name__)

# بارگذاری مدل فارسی
model_name = "HooshvareLab/bert-fa-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# لیست اسناد متنی
documents = [
    "چگونه گربه را به توالت عادت دهیم؟",
    "بهترین روش تربیت سگ چیست؟",
    "آموزش بچه‌گربه برای استفاده از خاک گربه",
    "چگونه زبان انگلیسی را سریع یاد بگیریم؟",
    "پرورش گربه در آپارتمان"
]

# تابع استخراج embedding (از [CLS])
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # فقط بردار [CLS]

# استخراج embedding اسناد
doc_embeddings = torch.stack([get_embedding(text)[0] for text in documents])

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    query_embedding = get_embedding(query)

    # محاسبه شباهت (Cosine)
    similarities = F.cosine_similarity(query_embedding, doc_embeddings)

    # مرتب‌سازی بر اساس شباهت
    scores_texts = sorted(zip(similarities, documents), key=lambda x: x[0], reverse=True)
    response = [{"text": text, "score": float(score)} for score, text in scores_texts]

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
