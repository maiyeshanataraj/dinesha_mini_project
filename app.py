from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load model and tokenizer directly
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

@app.route('/', methods=['GET', 'POST'])
def home():
    original_text = ""
    summary = ""
    original_count = 0
    summary_count = 0
    reduction = 0

    if request.method == 'POST':
        original_text = request.form.get('input_text', '').strip()
        
        if original_text:
            # Generate summary
            inputs = tokenizer([original_text], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, do_sample=False)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Word counts and reduction percentage
            original_count = len(original_text.split())
            summary_count = len(summary.split())
            
            if original_count > 0:
                reduction = round(((original_count - summary_count) / original_count) * 100, 2)

    return render_template(
        'index.html',
        original_text=original_text,
        summary=summary,
        original_count=original_count,
        summary_count=summary_count,
        reduction=reduction
    )

if __name__ == '__main__':
    app.run()