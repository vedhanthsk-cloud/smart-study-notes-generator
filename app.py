from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import BartForConditionalGeneration, BartTokenizer

app = FastAPI()

# Load model and tokenizer once at startup
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def count_words(text: str) -> int:
    return len(text.strip().split())

def summarize_text(text: str, max_len: int = 150, min_len: int = 30) -> str:
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

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

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
    margin: 0;
    padding: 0;
    line-height: 1.6;
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

HTML_FORM = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smart Study Notes Generator</title>
  {BASE_STYLE}
</head>
<body>
  <div class="container">
    <h1>Smart Study Notes Generator</h1>
    <p class="subtitle">Generate concise summaries and key points from any paragraph using AI</p>

    <div class="card">
      <form method="post">
        <label for="text">Enter a paragraph</label>
        <textarea name="text" id="text" placeholder="Paste or type your paragraph here..." required></textarea>
        <div style="margin-top: 14px;">
          <button type="submit">Generate Notes</button>
        </div>
      </form>
    </div>

    <div class="footer">
      Mini Project – Generative AI
    </div>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML_FORM)

@app.post("/", response_class=HTMLResponse)
async def generate_notes(text: str = Form(...)):
    if not text.strip():
        return HTMLResponse(HTML_FORM)

    try:
        summary = summarize_text(text)
    except Exception:
        summary = "Could not generate summary (input may be too short or model error)."

    key_points = extract_key_points(summary)

    orig_words = count_words(text)
    sum_words = count_words(summary)

    if orig_words > 0:
        reduction_pct = round((1 - sum_words / orig_words) * 100, 2)
    else:
        reduction_pct = 0.0

    points_html = "".join(f"<li>{p}</li>" for p in key_points)

    rendered = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smart Study Notes Generator</title>
  {BASE_STYLE}
</head>
<body>
  <div class="container">
    <h1>Smart Study Notes Generator</h1>
    <p class="subtitle">Generate concise summaries and key points from any paragraph using AI</p>

    <div class="card">
      <form method="post">
        <label for="text">Enter a paragraph</label>
        <textarea name="text" id="text" placeholder="Paste or type your paragraph here..." required>{text}</textarea>
        <div style="margin-top: 14px;">
          <button type="submit">Generate Notes</button>
        </div>
      </form>
    </div>

    <div class="card">
      <h2>Results</h2>

      <div class="original-box">
        <div class="section-title">Original text</div>
        <div style="white-space: pre-wrap; background: #f9fafb; border: 1px solid var(--border); border-radius: 8px; padding: 12px;">
{text}
        </div>
      </div>

      <div class="summary-box">
        <div class="section-title" style="margin-top: 16px;">Summary</div>
        <div style="white-space: pre-wrap; background: #f0fdf4; border: 1px solid #dcfce7; border-radius: 8px; padding: 12px;">
{summary}
        </div>
      </div>

      <div class="points-box">
        <div class="section-title" style="margin-top: 16px;">Key points</div>
        <ul>
          {points_html}
        </ul>
      </div>

      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Original words</div>
          <div class="stat-value">{orig_words}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Summary words</div>
          <div class="stat-value">{sum_words}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Text reduction</div>
          <div class="stat-value" style="color: var(--success);">{reduction_pct}%</div>
        </div>
      </div>
    </div>

    <div class="footer">
      Mini Project – Generative AI
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(rendered)