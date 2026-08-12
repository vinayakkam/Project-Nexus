import os
import re
import math
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
from pypdf import PdfReader
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize local self-contained OCR engine (No tesseract.exe needed!)
ocr_engine = RapidOCR()


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def extract_text_from_image(image_file):
    """Extracts embedded text from images using self-contained local ONNX OCR."""
    img = Image.open(image_file).convert('RGB')
    img_np = np.array(img)

    # Perform OCR
    result, _ = ocr_engine(img_np)

    if not result:
        return ""

    # Extract line text from OCR result matrix
    extracted_lines = [line[1] for line in result]
    return "\n".join(extracted_lines)


def textrank_summarize(text, sentence_count=4):
    """
    Pure mathematical TextRank algorithm.
    Ranks sentences using word-overlap graph connectivity.
    """
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


@app.route('/')
def home():
    return render_template('index.html')


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
                raw_text = uploaded_file.read().decode('utf-8', errors='ignore')
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
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)