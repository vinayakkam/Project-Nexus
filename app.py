import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizers import Tokenizer
from nexus_llm import NexusLLM

app = Flask(__name__)
CORS(app)

TOKENIZER_PATH = "tokenizer.json"
MODEL_PATH = "model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if os.path.exists(TOKENIZER_PATH) and os.path.exists(MODEL_PATH):
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        vocab_size = tokenizer.get_vocab_size()
        model = NexusLLM(vocab_size=vocab_size, dim=256, n_head=8, n_layer=4).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

load_model()

@app.route('/')
def home():
    return jsonify({'service': 'nexus-llm', 'status': 'ready'})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Message required.'}), 400

    if not model or not tokenizer:
        return jsonify({'error': 'Model weights missing. Run train.py first.'}), 500

    try:
        prompt = f"User: {message}\nBot:"
        encoded_ids = tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([encoded_ids], dtype=torch.long).to(device)

        eos_id = tokenizer.token_to_id("<|endoftext|>")
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=90,
            temperature=0.6,
            top_p=0.85,
            eos_token_id=eos_id
        )

        full_text = tokenizer.decode(output_ids[0].cpu().numpy().tolist())
        reply = full_text[len(prompt):].split("<|endoftext|>")[0].split("User:")[0].strip()

        return jsonify({
            'success': True,
            'reply': reply,
            'sources': []
        })

    except Exception as e:
        app.logger.exception("Inference failed")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)