from flask import Flask, render_template, request, jsonify
import logging
from main import TextGenerator
import os
import re

def validate_input(user_input):
    # Check for special characters
    if re.search(r'[^a-zA-Z0-9]', user_input):
        raise ValueError("Error: Special characters are not allowed.")

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the text generator
generator = TextGenerator(hidden_size=256)

# File paths
input_file = 'training_data.txt'
model_save_path = 'model.pth'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        prompt = request.json.get('prompt', '')
        max_length = request.json.get('max_length', 200)  # Default to 200 if not specified
        validate_input(prompt)
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        if not isinstance(max_length, int) or max_length < 50 or max_length > 1000:
            return jsonify({'error': 'Invalid maximum length. Must be between 50 and 1000 characters'}), 400

        # Generate text
        generated_text = generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=0.8
        )

        # Count words and characters
        words = len(generated_text.split())
        chars = len(generated_text)

        return jsonify({
            'text': generated_text,
            'stats': {
                'words': words,
                'characters': chars
            }
        })

    except Exception as e:
        logging.error(f"Error generating text: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Try to load existing model
        if os.path.exists(model_save_path):
            logging.info("Loading existing model...")
            generator.load_model(model_save_path)
        else:
            logging.info("Training new model...")
            # Read training data
            with open(input_file, 'r', encoding='utf-8') as f:
                training_text = f.read()
            
            if not training_text:
                raise ValueError("Training file is empty")
            
            # Train the model
            generator.train(
                text=training_text,
                sequence_length=50,
                batch_size=16,
                num_epochs=100
            )
            generator.save_model(model_save_path)
            logging.info("Model training completed and saved")

        # Start the Flask app
        app.run(debug=True)
        
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        logging.error("Traceback:", exc_info=True)
