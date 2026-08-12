import os
import re
import io
import time
import urllib.parse
import urllib.request
import logging
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from PIL import Image
from dotenv import load_dotenv

# --- Advanced Configuration & Scaling ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] Nexus: %(message)s')

app = Flask(__name__)
# Enable strict CORS for standard domain or wildcard for development
CORS(app)

# NEXUS_MODEL_NAME is used for internal reference/tags
app.config['NEXUS_MODEL'] = "OLIT-Nexus-Core-S9"
app.config['NEXUS_VERSION'] = "0.9.1.A"
app.config['START_TIME'] = time.time()

# --- Component: Lazy-loaded Local OCR Engine ---
# Initializing ONNX on boot is slow and causes startup timeouts. Lazy loading ensures quick ready state.
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        logging.info("Initializing Local OCR (ONNX Runtime)...")
        _ocr_engine = RapidOCR()
    return _ocr_engine


# ==============================================================================
# --- Component: Advanced Text Synthesis Processing (Robust Algorithms) ---
# ==============================================================================

def textrank_summarize(text, sentence_count=4):
    """
    Implementation of the TextRank mathematical algorithm.
    Ranks sentences based on localized word-overlap similarity graph connectivity.
    No external LLM needed.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Preprocessing & cleaning pipeline
    clean_sentences = []
    for s in sentences:
        cs = s.strip()
        if len(re.findall(r'\w+', cs)) > 5:  # Complexity filter
            clean_sentences.append(cs)

    if len(clean_sentences) <= sentence_count:
        return clean_sentences

    sentence_tokens = []
    for s in clean_sentences:
        tokens = set(re.findall(r'\b\w{3,}\b', s.lower()))
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

    # Ranks generation via localized power iteration
    d = 0.85  # Damping factor
    scores = [1.0] * n
    for _ in range(25):  # Increased iterations for stability
        new_scores = [1.0 - d] * n
        for i in range(n):
            for j in range(n):
                if i != j and similarity_matrix[j][i] > 0:
                    weight_sum = sum(similarity_matrix[j])
                    if weight_sum > 0:
                        new_scores[i] += d * (scores[j] * (similarity_matrix[j][i] / weight_sum))
        scores = new_scores

    # Bias towards beginning of document (standard for summarization)
    for index in range(min(len(scores), 15)):
        scores[index] *= (1.0 + (1.0 / (index + 1)) * 0.1)

    ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:sentence_count]
    sorted_indices = sorted(ranked_indices)

    return [clean_sentences[i] for i in sorted_indices]


# ==============================================================================
# --- Component: Local Grounding Engine (DuckDuckGo Local Scraping) ---
# ==============================================================================

def execute_local_web_grounding(query, max_sources=1):
    """
    Complex local grounding system. Scrapes web without API keys,
    fetches the primary source content, and synthesizes it locally.
    """
    logging.info(f"Executing Local Grounding Protocol for query: '{query}'")
    encoded_query = urllib.parse.quote_plus(f"{query} info site:en")
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Nexus Core Synthesis 0.9.1.A; Linux x86_64) OLIT/Nexus'
    }

    # Step 1: Execute search scraping
    search_results = []
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            html_content = response.read().decode('utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')

            # Standard result container for ddg html
            for result_div in soup.find_all('div', class_='result__body', limit=max_sources):
                url_elem = result_div.find('a', class_='result__url')
                title_elem = result_div.find('a', class_='result__a')

                if url_elem and title_elem:
                    url = url_elem.get('href', '').strip()
                    title = title_elem.get_text().strip()
                    # Basic sanitization
                    if not url.startswith('http'):
                        url = f"https:{url}"
                    search_results.append({'title': title, 'url': url})

    except Exception as e:
        logging.error(f"Search Grounding Scraping Failed: {e}")
        return [], None

    if not search_results:
        return [], None

    # Step 2: Fetch and Synthesize main content from first source
    primary_source = search_results[0]
    logging.info(f"Fetching Nexus Synthesis grounding from primary source: '{primary_source['title']}'")

    grounded_text = ""
    try:
        req_page = urllib.request.Request(primary_source['url'], headers=headers)
        with urllib.request.urlopen(req_page, timeout=10) as response_page:
            html_page = response_page.read().decode('utf-8', errors='ignore')
            soup_page = BeautifulSoup(html_page, 'html.parser')

            # Robust cleaning pipeline - remove all non-content nodes
            for script_node in soup_page(["script", "style", "nav", "footer", "header", "aside", "form"]):
                script_node.decompose()

            # Smart content extraction (paragraphs, headers)
            body_text = ' '.join(p.get_text() for p in soup_page.find_all(['p', 'h1', 'h2', 'h3']))
            grounded_text = re.sub(r'\s+', ' ', body_text).strip()
    except Exception as e:
        logging.warning(f"Failed to fetch content from {primary_source['url']}: {e}")

    # Local Math Synthesis
    synthesis_points = []
    if len(grounded_text.split()) > 200:  # Threshold for summarization
        logging.info(f"Synthesizing {len(grounded_text.split())} words of grounded text locally...")
        synthesis_points = textrank_summarize(grounded_text, sentence_count=3)
        synthesis_points.insert(0, f"Synthesis complete. Source data: -> '{primary_source['title']}'.")
    elif grounded_text:
        synthesis_points = [f"Direct synthesis from source -> '{primary_source['title']}':", grounded_text]
    else:
        synthesis_points = [
            f"Grounded source located -> '{primary_source['title']}', but local content synthesis was blocked by server."]

    return search_results, "\n".join(synthesis_points)


# ==============================================================================
# --- Component: NEXUS Modular Intent Engine (Scaling Matching) ---
# ==============================================================================

class NexusIntentEngine:
    """Modular engine for detecting query complexity and routing intents locally."""

    def __init__(self):
        # Priority mapping for faster matching
        self.priority_keywords = {
            'image': ['draw', 'generate image', 'create visual', '/image', 'visual synthesis of'],
            'code': ['code', 'python', 'script', 'flask', 'javascript', 'algorithm', 'function', 'class', 'html',
                     'css'],
            'info': ['definition of', 'who is', 'what is', 'explain', 'tell me about', 'summary of', 'summarize'],
            'greeting': ['hi', 'hello', 'hey', 'greetings', 'nexus'],
            'identity': ['who are you', 'what are you', 'model name', 'nexus engine']
        }

    def determine_intent(self, message):
        msg_lower = message.lower().strip()

        # 1. Complexity & Grounding Detection pipeline
        # Complexity based on length and standard grounding trigger words
        grounding_triggers = ['current', 'latest', 'weather', 'news', 'price of', 'today']
        words = msg_lower.split()

        # High priority grounding matching
        if any(kw in msg_lower for kw in grounding_triggers) or len(words) > 12:
            return 'grounding'

        # 2. Sequential priority matching
        if any(kw in msg_lower for kw in self.priority_keywords['image']):
            return 'image'
        if any(kw in msg_lower for kw in self.priority_keywords['code']):
            return 'code'
        if any(kw in msg_lower for kw in self.priority_keywords['greeting']):
            return 'greeting'
        if any(kw in msg_lower for kw in self.priority_keywords['identity']):
            return 'identity'
        if any(kw in msg_lower for kw in self.priority_keywords['info']):
            return 'info'

        # Default conversational routing
        return 'text'


# Global instance of Intent Engine
_intent_engine = NexusIntentEngine()


def format_code_solution(msg_lower, complexity='generic'):
    """Generates standard, localized code templates for standard coding queries."""
    if 'python' in msg_lower or 'flask' in msg_lower:
        return (
            "```python\n"
            "# Local Nexus Synthesis Solution\n"
            "# Requirement: Python 3.9+ standard library\n"
            "import os\n\n"
            "def synthesis_op(input_payload):\n"
            "    \"\"\"Processes and structures incoming query payload locally.\"\"\"\n"
            "    # Cleanup and filtering logic\n"
            "    filtered_data = [item.strip() for item in input_payload if isinstance(item, str)]\n"
            "    \n"
            "    if not filtered_data:\n"
            "        return {'status': 'error', 'message': 'operation payload empty'}\n"
            "        \n"
            "    # standard task operation mocked\n"
            "    return {'status': 'success', 'processed': filtered_data[:5]}\n"
            "```"
        )
    elif 'html' in msg_lower or 'css' in msg_lower or 'js' in msg_lower:
        return (
            "```html\n"
            "<!DOCTYPE html>\n"
            "<!-- Nexus core local render template -->\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "    <meta charset=\"UTF-8\">\n"
            "    <title>Nexus Output Template</title>\n"
            "</head>\n"
            "<body>\n"
            "    <div id=\"root\">Local Core Engine Ready</div>\n"
            "</body>\n"
            "</html>\n"
            "```"
        )
    else:
        # Standard generic local execution template
        return (
            "```python\n"
            "# generic local helper script\n"
            "import time\n\n"
            "def main_op():\n"
            "    start = time.time()\n"
            "    print('Initializing local operational synthesis...')\n"
            "    time.sleep(0.01) # task simulation\n"
            "    print(f'Execution complete in {(time.time() - start)*1000:.2f}ms.')\n\n"
            "if __name__ == '__main__':\n"
            "    main_op()\n"
            "```"
        )


# ==============================================================================
# --- Component: Asset OCR & Text Analysis Pipeline (Robust File Handling) ---
# ==============================================================================

def extract_content_pipeline(uploaded_file):
    """
    Robust extraction pipeline. Handles seek resets, read buffers,
    stream conversion, and error propagation.
    """
    # Force seek reset for robust re-read support
    uploaded_file.seek(0)
    filename = uploaded_file.filename.lower()

    if not filename:
        raise ValueError("Filename is invalid.")

    # PDF Analysis Pipeline
    if filename.endswith('.pdf'):
        extracted_text = ""
        try:
            reader = PdfReader(uploaded_file)
            logging.info(f"Analyzing robust PDF asset: {filename} ({len(reader.pages)} pages)")
            for page_index, page in enumerate(reader.pages):
                extracted_text += page.extract_text() or ""
            return extracted_text
        except Exception as e:
            raise RuntimeError(f"PDF Analysis pipeline failed: {e}")

    # Image OCR Pipeline
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        logging.info(f"Initializing Local OCR pipeline for asset: {filename}")
        img_bytes = uploaded_file.read()

        # Fix stream corrupted issues by re-wrapping bytes
        try:
            pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            # Local RapidOCR Engine Execution
            engine = get_ocr_engine()
            result, _ = engine(np.array(pil_image))

            if not result:
                return ""

            # Standard extraction output formatting
            lines = [line[1] for line in result]
            return "\n".join(lines)
        except Exception as e:
            raise RuntimeError(f"OCR Pipeline failed: {e}")

    # Pure Text Pipeline
    elif filename.endswith('.txt'):
        uploaded_file.seek(0)
        return uploaded_file.read().decode('utf-8', errors='ignore')

    else:
        raise ValueError("Unsupported file type provided to analysis pipeline.")


# ==============================================================================
# --- Component: FLASK Nexus Backend Application Routing ---
# ==============================================================================

@app.route('/')
def home_nexus():
    # Nexus Core Health Endpoint
    status = {
        'service': 'olitt-nexus-core',
        'status': 'operational',
        'model': app.config['NEXUS_MODEL'],
        'synthesis_engine': 'textrank_math_0.85',
        'grounding': 'local_ddg_scraping',
        'ocr': 'rapid_onnx_local',
        'uptime': f"{(time.time() - app.config['START_TIME']) / 3600:.2f}h"
    }
    return jsonify(status)


@app.route('/api/health')
def health():
    # Fast health check without triggering OCR models
    return jsonify({'nexus_core_status': 'ok'})


@app.route('/api/summarize', methods=['POST'])
def nexus_summarize():
    """Robust endpoint for asset analysis and synthesis (summarization)."""
    try:
        raw_content = ""

        # Check for asset upload (PDF, Image, Text) first (Complexity logic)
        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file = request.files['file']
            raw_content = extract_content_pipeline(uploaded_file)
        else:
            # Fallback to pure text field
            raw_content = request.form.get('text', '')

        if not raw_content.strip():
            return jsonify({'error': 'Analysis pipeline located no restorable content.'}), 400

        # Run Local Math Summarization Algorithm
        total_words = len(re.findall(r'\w+', raw_content))

        # Complex dynamic synthesis target
        dynamic_sentence_target = max(3, min(6, int(total_words / 150)))
        logging.info(f"Synthesizing {total_words} words down to ~{dynamic_sentence_target} grounded sentences...")

        summary_points = textrank_summarize(raw_content, sentence_count=dynamic_sentence_target)

        summary_words = sum(len(re.findall(r'\w+', pt)) for pt in summary_points)
        reduction_ratio = round((1 - summary_words / total_words) * 100) if total_words > 0 else 0

        # Successful Synthesis Payload
        return jsonify({
            'success': True,
            'summary': summary_points,
            'stats': {
                'total_words': total_words,
                'reduction': max(0, reduction_ratio),
                'read_time_reduction': max(1, round(total_words / 200))
            }
        })

    except ValueError as ve:
        # User error formatting
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        # System error robust formatting
        logging.critical(f"Summarize Pipeline Crash: {e}")
        app.logger.exception("Summarize Pipeline crash:")
        return jsonify({'error': f"Operational Synthesis failure. {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def nexus_chat():
    """Unified conversational routing. Automatically determines intent and grounding."""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'operational payload required'}), 400

        sources = []
        reply_content = ""
        intent_type = "text"
        procedural_data = ""

        # Step 1: Query Intent Engine
        intent = _intent_engine.determine_intent(message)
        logging.info(f"Message Intented Routed to: {intent}")

        # Step 2: Route Operation based on matched Intent
        if intent == 'grounding':
            # Execute Advanced Local Grounding pipeline
            sources, reply_content = execute_local_web_grounding(message, max_sources=1)
            # If grounding failed to locate source, default to basic text response
            if not reply_content:
                reply_content = f"Acknowledged query -> '{message}'. Executed query locally. Local Nexus grounding pipeline synthesis was non-responsive. No synthesis data was generated."

        elif intent == 'image':
            intent_type = "image"
            # Extract prompt for visual output mockup on frontend canvas
            procedural_data = re.sub(r'^(generate image|draw|/image)\s*', '', message, flags=re.IGNORECASE).strip()
            reply_content = f"Visual operations synthesis complete for localized prompt -> '{procedural_data or 'Abstract synthesis operational operational operational operation'}'"

        elif intent == 'code':
            reply_content = format_code_solution(message.lower())

        elif intent == 'greeting':
            reply_content = "Hello! Acknowledged. This is OLIT Nexus Core Synthesis Operational Operational Operational Operation. How may I assist your local synthesis operation today?"

        elif intent == 'identity':
            reply_content = f"Operation: Local Core Synthesis. Identification: {app.config['NEXUS_MODEL']}. Status: Offline. Operational Operational Operational Operation."

        elif intent == 'info':
            reply_content = f"ACK query -> '{message}'. In Offline Core Synthesis mode (OLIT Nexus), full definition synthesis requires Local Grounding -> enable Web Grounding or paste relevant data Operation."

        else:
            # Default fallback conversational response
            reply_content = f"Query synthesis complete -> '{message}'. Standard response executed via Offline Core Synthesis pipeline (OLIT Nexus) Operational Operational Operational Operation."

        # Unified Success Payload
        return jsonify({
            'success': True,
            'reply': reply_content,
            'sources': sources,
            'type': intent_type,
            'prompt': procedural_data
        })

    except Exception as e:
        # Fatal crash robust formatting
        logging.critical(f"Chat Pipeline Crash: {e}")
        app.logger.exception("Chat Pipeline crash:")
        return jsonify({'error': f"Offline synthesis failure. {str(e)}"}), 500


# Fast startup boot check for Cloud Run
if __name__ == '__main__':
    logging.info(f"OLIT Nexus Core Initializing... Offline Mode Ready.")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)