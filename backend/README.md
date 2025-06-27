# Backend

This is the backend of an AI chatbot to generate bash scripts that serves a Flask API using a fine-tuned DeepSeek-Coder model with LoRA adapters to process natural language requests and generate executable bash scripts.

## Setup Instructions

**Terminal 1: Start the Backend (Flask API)**
Open Terminal 1.
Activate the backend virtual environment:
~~~
source ~/path/ai-goal/backend/venv_backend/bin/activate
~~~
Navigate to the backend directory:
~~~
cd backend
~~~
Run the Flask application:
~~~
python app.py
~~~
Wait for the model to finish loading. This will take a few minutes for the DeepSeek-Coder model. You will see print statements like "Loading model and tokenizer for Flask API..." and eventually "Model and tokenizer loaded successfully for API!" followed by * Running on http://0.0.0.0:5000/.

Keep this terminal window open while the chatbot is in use.
