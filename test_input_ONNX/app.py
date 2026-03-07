from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# Import functions from the NLPTest.py file we just wrote
from NLPTest import load_ws_model, segment_text, tag_features

app = Flask(__name__)
CORS(app) # Allow cross-origin requests

# Ensure the model is loaded once when the app starts
print("初始化 Flask 伺服器...")
ws_pipeline = load_ws_model()

@app.route('/')
def home():
    # Render the main HTML page
    return render_template('test_nlp.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '請輸入文字'}), 400
            
        # 1. Segment the text
        segmented_words = segment_text(ws_pipeline, text)
        
        # 2. Extract features
        features = tag_features(segmented_words)
        
        # 3. Clean up None values or empty lists for a cleaner API response
        cleaned_features = {k: v for k, v in features.items() if v}
        
        return jsonify({
            'success': True,
            'original_text': text,
            'segmented_words': segmented_words,
            'features': cleaned_features
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # Start the Flask development server on port 5000
    app.run(debug=True, port=5000)
