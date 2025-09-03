import os
import numpy as np
import pandas as pd
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- Config ---------
# Point this to the folder that contains: config.json, model.safetensors (or pytorch_model.bin),
# tokenizer files (vocab.txt / merges / tokenizer_config.json / special_tokens_map.json, etc.)
# You can also override via environment variable: export MODEL_DIR=/path/to/model
MODEL_DIR = "/path/to/model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


# Ensure label mapping (adjust if your labels differ)
model.config.id2label = {0: "Negative", 1: "Positive"}
model.config.label2id = {"Negative": 0, "Positive": 1}
labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

def predict_one(text: str):
    txt = (text or "").strip()
    if not txt:
        raise gr.Error("Please enter a sentence.")
    with torch.no_grad():
        inputs = tokenizer([txt], return_tensors="pt", truncation=True, padding=True)
        probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
    top = labels[int(np.argmax(probs))]
    return top, probs

def _empty_df():
    return pd.DataFrame({
        "Class": labels,
        "Percent": [0.0] * len(labels),
        "Label": ["0.0%"] * len(labels),
    })

def predict_handler(text):
    text = (text or "").strip()
    if not text:
        return "<div class='result-card neutral'>No input</div>", _empty_df()

    top, probs = predict_one(text)

    cls = "positive" if top.lower().startswith("pos") else "negative"
    result_html = f"<div class='result-card {cls}'>Prediction: {top}</div>"

    classes = labels
    perc = [round(float(v) * 100, 1) for v in probs]
    df = (
        pd.DataFrame({"Class": classes, "Percent": perc, "Label": [f"{p:.1f}%" for p in perc]})
        .sort_values("Percent", ascending=False)
        .reset_index(drop=True)
    )
    return result_html, df

def toggle_btn(text):
    return gr.update(interactive=bool(text and text.strip()))

# --------- UI ---------
with gr.Blocks(title="Sentiment Predictor") as demo:
    gr.HTML("""
    <style>
      .gradio-container, .gr-blocks, .gr-form { max-width:100% !important; width:100% !important; }
      #app-title { text-align:center; font-size:2.4rem; font-weight:800; margin:10px 0 6px; }
      #app-sub   { text-align:center; color:#6b7280; margin-bottom:16px; }
      #input-box textarea { border-radius:12px !important; box-shadow:0 6px 18px rgba(0,0,0,.08); }
      #btn-predict { background:#3b82f6 !important; color:#fff !important; border:none !important; }
      #btn-predict:hover { filter:brightness(1.05); }
      #btn-clear { background:#e5e7eb !important; color:#111827 !important; border:none !important; }
      #btn-clear:hover { filter:brightness(0.98); }
      .result-card { text-align:center; font-weight:800; font-size:1.3rem; padding:18px; border-radius:14px; color:white;
                     box-shadow:0 10px 24px rgba(0,0,0,.15); margin-top:6px; margin-bottom:18px; }
      .result-card.positive { background:#22c55e; }
      .result-card.negative { background:#ef4444; }
      .result-card.neutral  { background:#6b7280; }
      .section-title { margin:6px 0 8px; font-weight:700; }
    </style>
    """)

    # Header
    gr.Markdown(
        "<h1 style='text-align:center; font-size:2.5rem;'>Sentiment Analysis</h1>"
        "<p style='text-align:center; color:gray;'>Type a sentence and see if it's "
        "<b style='color:green;'>Positive</b> or <b style='color:red;'>Negative</b></p>"
    )

    # Input
    txt = gr.Textbox(
        label="Your sentence",
        placeholder="Type something here…",
        lines=1,
        max_lines=1,
        autofocus=True,
        elem_id="input-box"
    )

    with gr.Row():
        btn_predict = gr.Button("Predict", elem_id="btn-predict")
        btn_clear = gr.ClearButton([txt], value="Clear", elem_id="btn-clear")

    txt.input(toggle_btn, inputs=txt, outputs=btn_predict)

    # Result card
    out_result = gr.HTML("")

    gr.Markdown("<div class='section-title'>Analysis of the result</div>")

    out_bars = gr.BarPlot(
        value=pd.DataFrame({"Class": labels, "Percent": [0.0]*len(labels), "Label": ["0.0%"]*len(labels)}),
        x="Percent",
        y="Class",
        vertical=False,
        x_lim=(0, 100),
        title=None,
        interactive=False,
        tooltip=["Class"]
    )

    # Submit (button + Enter)
    btn_predict.click(predict_handler, inputs=txt, outputs=[out_result, out_bars])
    txt.submit(predict_handler, inputs=txt, outputs=[out_result, out_bars])

    # Clear
    btn_clear.click(
        lambda: ("", pd.DataFrame({"Class": labels, "Percent": [0.0]*len(labels), "Label": ["0.0%"]*len(labels)})),
        inputs=None,
        outputs=[out_result, out_bars]
    )

# --------- Launch ---------
if __name__ == "__main__":
    demo.launch(share = True)
