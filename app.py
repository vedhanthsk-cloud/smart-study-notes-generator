from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import BartForConditionalGeneration, BartTokenizer

app = FastAPI()

tokenizer = None
model = None
model_loaded = False

def load_model():
    global tokenizer, model, model_loaded
    if model_loaded:
        return
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model_loaded = True

def count_words(text: str) -> int:
    return len(text.strip().split())

def summarize_text(text: str, max_len: int = 150, min_len: int = 30) -> str:
    load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_len = inputs["input_ids"].shape[1]
    max_len = min(max_len, input_len - 1)
    min_len = min(min_len, max_len - 1)

    summary_ids = model.generate(
        **inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_key_points(summary: str, max_points: int = 5) -> list[str]:
    sentences = [s.strip() for s in summary.replace("\n", " ").split(".") if s.strip()]
    return sentences[:max_points]

BASE_STYLE = """
<style>
  :root {
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --bg: #f9fafb;
    --card-bg: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --success: #10b981;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0; padding: 0; line-height: 1.6;
  }
  .container {
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
  }
  h1, h2 {
    margin: 0 0 12px 0;
    color: #1f2937;
  }
  h1 {
    font-size: 2rem;
    text-align: center;
    margin-bottom: 8px;
  }
  .subtitle {
    text-align: center;
    color: var(--muted);
    margin-bottom: 28px;
    font-size: 0.95rem;
  }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
  }
  label {
    display: block;
    font-weight: 600;
    margin-bottom: 8px;
  }
  textarea {
    width: 100%;
    min-height: 160px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.95rem;
    font-family: inherit;
    resize: vertical;
  }
  textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
  }
  button {
    background: var(--primary);
    color: #fff;
    border: none;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s ease;
  }
  button:hover {
    background: var(--primary-dark);
  }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-top: 12px;
  }
  .stat-box {
    background: #f3f4f6;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }
  .stat-label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .stat-value {
    font-weight: 700;
    font-size: 1.05rem;
  }
  .summary-box, .points-box, .original-box {
    margin-top: 12px;
  }
  .section-title {
    font-weight: 700;
    margin-bottom: 6px;
    color: #1f2937;
  }
  ul {
    margin: 6px 0 0 18px;
    padding: 0;
  }
  li {
    margin-bottom: 6px;
  }
  .footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 28px;
    margin-bottom: 40px;
  }
</style>
"""

def make_page(title: str, body_html: str) -> str:
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        f"  <meta charset=\"utf-8\">\n"
        f"  <title>{title}</title>\n"
        f"{BASE_STYLE}\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )

def home_body() -> str:
    return (
        '<div class="container">\n'
        '  <h1>Smart Study Notes Generator</h1>\n'
        '  <p class="subtitle">Generate concise summaries and key points from any paragraph using AI</p>\n'
        '  <div class="card">\n'
        '    <form method="post">\n'
        '      <label for="text">Enter a paragraph</label>\n'
        '      <textarea name="text" id="text" placeholder="Paste or type your paragraph here..." required></textarea>\n'
        '      <div style="margin-top: 14px;">\n'
        '        <button type="submit">Generate Notes</button>\n'
        '      </div>\n'
        '    </form>\n'
        '  </div>\n'
        '  <div class="footer">Mini Project – Generative AI</div>\n'
        '</div>\n'
    )

def result_body(
    text: str,
    summary: str,
    points_html: str,
    orig_words: int,
    sum_words: int,
    reduction_pct: float,
) -> str:
    return (
        '<div class="container">\n'
        '  <h1>Smart Study Notes Generator</h1>\n'
        '  <p class="subtitle">Generate concise summaries and key points from any paragraph using AI</p>\n'
        '  <div class="card">\n'
        '    <form method="post">\n'
        '      <label for="text">Enter a paragraph</label>\n'
        f'      <textarea name="text" id="text" required>{text}</textarea>\n'
        '      <div style="margin-top: 14px;">\n'
        '        <button type="submit">Generate Notes</button>\n'
        '      </div>\n'
        '    </form>\n'
        '  </div>\n'
        '  <div class="card">\n'
        '    <h2>Results</h2>\n'
        '    <div class="original-box">\n'
        '      <div class="section-title">Original text</div>\n'
        '      <div style="white-space: pre-wrap; background: #f9fafb; border: 1px solid var(--border); border-radius: 8px; padding: 12px;">\n'
        f'        {text}\n'
        '      </div>\n'
        '    </div>\n'
        '    <div class="summary-box">\n'
        '      <div class="section-title" style="margin-top: 16px;">Summary</div>\n'
        '      <div style="white-space: pre-wrap; background: #f0fdf4; border: 1px solid #dcfce7; border-radius: 8px; padding: 12px;">\n'
        f'        {summary}\n'
        '      </div>\n'
        '    </div>\n'
        '    <div class="points-box">\n'
        '      <div class="section-title" style="margin-top: 16px;">Key points</div>\n'
        '      <ul>\n'
        f'        {points_html}\n'
        '      </ul>\n'
        '    </div>\n'
        '    <div class="stats-grid">\n'
        '      <div class="stat-box">\n'
        '        <div class="stat-label">Original words</div>\n'
        f'        <div class="stat-value">{orig_words}</div>\n'
        '      </div>\n'
        '      <div class="stat-box">\n'
        '        <div class="stat-label">Summary words</div>\n'
        f'        <div class="stat-value">{sum_words}</div>\n'
        '      </div>\n'
        '      <div class="stat-box">\n'
        '        <div class="stat-label">Text reduction</div>\n'
        f'        <div class="stat-value" style="color: var(--success);">{reduction_pct}%</div>\n'
        '      </div>\n'
        '    </div>\n'
        '  </div>\n'
        '  <div class="footer">Mini Project – Generative AI</div>\n'
        '</div>\n'
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    html = make_page("Smart Study Notes Generator", home_body())
    return HTMLResponse(html)

@app.post("/", response_class=HTMLResponse)
async def generate_notes(text: str = Form(...)):
    if not text.strip():
        html = make_page("Smart Study Notes Generator", home_body())
        return HTMLResponse(html)

    try:
        summary = summarize_text(text)
    except Exception:
        summary = "Could not generate summary (input may be too short or model error)."

    key_points = extract_key_points(summary)
    points_html = "".join(f"<li>{p}</li>" for p in key_points)

    orig_words = count_words(text)
    sum_words = count_words(summary)
    reduction_pct = round((1 - sum_words / orig_words) * 100, 2) if orig_words > 0 else 0.0

    body = result_body(text, summary, points_html, orig_words, sum_words, reduction_pct)
    html = make_page("Smart Study Notes Generator - Results", body)
    return HTMLResponse(html)