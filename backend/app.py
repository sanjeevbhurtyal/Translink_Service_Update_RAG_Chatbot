from flask import Flask, request, jsonify
from flask_cors import CORS
import rag

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    model = data.get('model', 'qwen3:8b') # Default model
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        response = rag.query_rag(user_query, model=model)
        return jsonify({"response": response}), 200
    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Translink Chatbot Server...")
    print("Make sure your Qwen LLM is running (e.g., via Ollama)")
    app.run(port=5000, debug=True)
