from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load Pre-trained Model and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

def summarize_text(text, length_factor):
    if not text.strip():
        return "No text provided for summarization."

    # Convert slider value (1-100) into min and max summary lengths
    max_length = int(30 + (length_factor / 100) * 120)  # Scale between 30 and 150
    min_length = int(max_length * 0.5)  # Ensure min length is at least half of max
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    input_text = ""
    length_factor = 50  # Default slider value

    if request.method == "POST":
        input_text = request.form.get("input_text")
        length_factor = int(request.form.get("length_factor", 50))  # Get slider value

        if input_text:
            summary = summarize_text(input_text, length_factor)
    
    return render_template("index.html", input_text=input_text, summary=summary, length_factor=length_factor)

if __name__ == "__main__":
    app.run(debug=True)
