import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('./movie_review_model')
tokenizer = DistilBertTokenizer.from_pretrained('./movie_review_model')


def predict_rating(review_text):
    inputs = tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = torch.argmax(logits, dim=1).item()

    sentiment = "Positive" if predicted_class >= 5 else "Negative"

    return predicted_class, sentiment
