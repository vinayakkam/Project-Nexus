import os
import re
import io
import urllib.parse
import urllib.request
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Lazy-loaded OCR engine ---
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def extract_text_from_pdf(pdf_file):
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def extract_text_from_image(image_file):
    """Extracts text from images using local self-contained ONNX OCR."""
    image_file.seek(0)
    img_bytes = image_file.read()

    if not img_bytes:
        return ""

    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)

    engine = get_ocr_engine()
    result, _ = engine(img_np)

    if not result:
        return ""

    extracted_lines = [line[1] for line in result]
    return "\n".join(extracted_lines)


def textrank_summarize(text, sentence_count=4):
    """Pure mathematical TextRank algorithm for local text summarization."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(clean_sentences) <= sentence_count:
        return clean_sentences

    sentence_tokens = []
    for s in clean_sentences:
        tokens = set(re.findall(r'\b[a-zA-Z0-9]{2,}\b', s.lower()))
        sentence_tokens.append(tokens)

    n = len(clean_sentences)
    similarity_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or not sentence_tokens[i] or not sentence_tokens[j]:
                continue
            intersection = len(sentence_tokens[i].intersection(sentence_tokens[j]))
            union = len(sentence_tokens[i].union(sentence_tokens[j]))
            similarity_matrix[i][j] = intersection / float(union) if union > 0 else 0.0

    d = 0.85
    scores = [1.0] * n

    for _ in range(20):
        new_scores = [1.0 - d] * n
        for i in range(n):
            for j in range(n):
                if i != j and similarity_matrix[j][i] > 0:
                    weight_sum = sum(similarity_matrix[j])
                    if weight_sum > 0:
                        new_scores[i] += d * (scores[j] * (similarity_matrix[j][i] / weight_sum))
        scores = new_scores

    for index in range(n):
        scores[index] *= (1.0 + (1.0 / (index + 1)) * 0.2)

    ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:sentence_count]
    sorted_indices = sorted(ranked_indices)

    return [clean_sentences[i] for i in sorted_indices]


# --- Local Web Search Engine (No API key or external bot required) ---
def perform_web_search(query, max_results=3):
    """Scrapes search results directly using HTML requests."""
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }

    results = []
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')

            for a in soup.find_all('a', class_='result__url', limit=max_results):
                title_elem = a.find_parent('div', class_='result__body')
                title = title_elem.find('a', class_='result__a').text.strip() if title_elem and title_elem.find('a', class_='result__a') else "Source"
                href = a.get('href', '')
                if href:
                    results.append({'title': title, 'url': href})
    except Exception as e:
        app.logger.warning(f"Web search failed: {e}")

    return results


def fetch_url_content(url):
    """Extracts readable body text from a webpage URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')

            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return " ".join(chunk for chunk in chunks if chunk)
    except Exception:
        return ""


# --- Local Rule-Based Chat Logic (No Bot/LLM API) ---
def generate_local_response(message, mode='chat'):
    msg_lower = message.lower().strip()

    # Code templates generator
    if mode == 'code' or any(kw in msg_lower for kw in ['code', 'python', 'function', 'class', 'script', 'html', 'css', 'js']):
        if 'python' in msg_lower or 'flask' in msg_lower:
            return (
                "```python\n"
                "# Python Solution\n"
                "def process_data(input_data):\n"
                "    \"\"\"Processes raw input data and returns structured output.\"\"\"\n"
                "    if not input_data:\n"
                "        return {'status': 'error', 'message': 'Empty input'}\n"
                "    \n"
                "    processed = [item.strip() for item in input_data if isinstance(item, str)]\n"
                "    return {'status': 'success', 'data': processed}\n"
                "```"
            )
        elif 'html' in msg_lower or 'css' in msg_lower or 'js' in msg_lower:
            return (
                "```html\n"
                "<!DOCTYPE html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "    <meta charset=\"UTF-8\">\n"
                "    <title>Output</title>\n"
                "</head>\n"
                "<body>\n"
                "    <div id=\"app\">Hello World</div>\n"
                "</body>\n"
                "</html>\n"
                "```"
            )
        else:
            return (
                "```python\n"
                "# Generic helper script\n"
                "import os\n\n"
                "def main():\n"
                "    print('Execution complete.')\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
                "```"
            )

    # General chat patterns
    if any(greeting in msg_lower for greeting in ['hi', 'hello', 'hey', 'greetings']):
        return "Hello! How can I assist you with text extraction, summarization, or searching today?"
    elif 'who are you' in msg_lower or 'what are you' in msg_lower:
        return "I am OLIT Backend, a local algorithm-based assistant for text extraction, summarization, web searching, and code template generation."
    elif 'summarize' in msg_lower:
        return "You can upload a PDF, image, or raw text to the `/api/summarize` endpoint to generate a local TextRank summary."
    else:
        return f"Received query: '{message}'. Processed locally via rule-based engine."


@app.route('/')
def home():
    return jsonify({'service': 'olit-backend', 'status': 'ok'})


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        raw_text = ""

        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file = request.files['file']
            filename = uploaded_file.filename.lower()

            if filename.endswith('.pdf'):
                raw_text = extract_text_from_pdf(uploaded_file)
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                raw_text = extract_text_from_image(uploaded_file)
            elif filename.endswith('.txt'):
                uploaded_file.seek(0)
                raw_text = uploaded_file.read().decode('utf-8', errors='ignore')
            else:
                return jsonify({'error': 'Unsupported file type.'}), 400
        else:
            raw_text = request.form.get('text', '')

        if not raw_text.strip():
            return jsonify({'error': 'No readable text could be extracted.'}), 400

        total_words = len(raw_text.split())
        summary_points = textrank_summarize(raw_text, sentence_count=4)
        summary_words = sum(len(pt.split()) for pt in summary_points)
        reduction = round((1 - summary_words / total_words) * 100) if total_words > 0 else 0

        return jsonify({
            'success': True,
            'summary': summary_points,
            'stats': {
                'total_words': total_words,
                'reduction': max(0, reduction),
                'read_time': max(1, round(total_words / 200))
            }
        })

    except Exception as e:
        app.logger.exception("summarize() failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    100% Local Chat / Code / Search endpoint with zero external LLM dependencies.
    """
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    enable_search = data.get('enable_search', False)
    mode = data.get('mode', 'chat')

    if not message:
        return jsonify({'error': 'Message field is required.'}), 400

    try:
        sources = []
        reply = ""

        # If search is enabled, scrape DuckDuckGo locally and extract key text
        if enable_search:
            sources = perform_web_search(message, max_results=3)
            if sources:
                extracted_content = fetch_url_content(sources[0]['url'])
                if extracted_content:
                    summary = textrank_summarize(extracted_content, sentence_count=2)
                    reply = f"Search Results for '{message}':\n\n" + " ".join(summary)
                else:
                    reply = f"Found {len(sources)} relevant web sources for your query."
            else:
                reply = f"No direct search results found for '{message}'."
        else:
            reply = generate_local_response(message, mode=mode)

        return jsonify({
            'success': True,
            'reply': reply,
            'sources': sources
        })

    except Exception as e:
        app.logger.exception("chat() failed")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)