import os
import sys
import torch
import subprocess
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


def ensure_model_artifacts():
    """Checks if model weights and tokenizer exist. If missing (e.g. on Cloud Run),
    runs train.py automatically on backend startup."""
    if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(MODEL_PATH):
        print("[Nexus Core] Model weights or tokenizer missing. Initializing automated boot training...")
        try:
            # Execute train.py using the current Python executable
            result = subprocess.run([sys.executable, "train.py"], check=True, capture_output=True, text=True)
            print("[Nexus Core] Training stdout:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[Nexus Core] Automated training failed:\n{e.stderr}")
            raise RuntimeError(f"Failed to auto-train model artifacts: {e.stderr}")


def load_model():
    global tokenizer, model

    # 1. Ensure model files exist before loading
    ensure_model_artifacts()

    # 2. Load tokenizer and model weights into memory
    if os.path.exists(TOKENIZER_PATH) and os.path.exists(MODEL_PATH):
        print(f"[Nexus Core] Loading Tokenizer from {TOKENIZER_PATH}...")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        vocab_size = tokenizer.get_vocab_size()

        print(f"[Nexus Core] Initializing NexusLLM architecture on device: {device}...")
        model = NexusLLM(vocab_size=vocab_size, dim=256, n_head=8, n_layer=4).to(device)

        print(f"[Nexus Core] Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("[Nexus Core] Custom LLM fully initialized and operational!")


# Initialize the model on boot
load_model()


@app.route('/')
def home():
    return jsonify({
        'service': 'nexus-llm',
        'status': 'ready',
        'device': device,
        'model_loaded': model is not None
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Message payload required.'}), 400

    if not model or not tokenizer:
        return jsonify({'error': 'Custom LLM failed to load weights.'}), 500

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

        # Clean response payload
        reply = full_text[len(prompt):].split("<|endoftext|>")[0].split("User:")[0].strip()

        return jsonify({
            'success': True,
            'reply': reply,
            'sources': []
        })

    except Exception as e:
        app.logger.exception("Inference processing error")
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)