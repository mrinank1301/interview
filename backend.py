from flask import Flask, request, jsonify, render_template
import azure.cognitiveservices.speech as speechsdk
import os
import openai
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import time
import json
from textblob import TextBlob
import vaderSentiment.vaderSentiment
import random

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)


logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    
    
    OPENAI_API_TYPE = "azure"  
    OPENAI_API_KEY = "22Mj7xKp5fPvOKQDZ54xncvwHCUUt27nPBhmgI89k60HJ3do1kgTJQQJ99ALACYeBjFXJ3w3AAABACOGOy3V"
    OPENAI_ENDPOINT_URL = "https://jobspringai.openai.azure.com/"
    OPENAI_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
    
    AZURE_API_KEY = "2XJ6aPd30utg3LgzQtxd3peUvYK92O42pt1zZySHWSXUB9OLyVS5JQQJ99ALACYeBjFXJ3w3AAAYACOGufOC"
    AZURE_endpoint = "  https://eastus.api.cognitive.microsoft.com/"
    
    
    SPEECH_SYNTHESIS_VOICE = "en-US-JennyNeural"
    SPEECH_SYNTHESIS_RATE = "+0%"
    SPEECH_SYNTHESIS_PITCH = "+0Hz"

app.config.from_object(Config)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file):
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_text_from_docx(file):
    try:
        from docx import Document
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8").strip()
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {str(e)}")
        raise


def analyze_sentiment_textblob(text):
    try:
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        if sentiment_score > 0.3:
            sentiment = "Positive"
        elif sentiment_score < -0.3:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "polarity_score": round(sentiment_score, 2),
            "subjectivity": round(blob.sentiment.subjectivity, 2)
        }
    except Exception as e:
        logger.error(f"Error in TextBlob sentiment analysis: {str(e)}")
        return {"error": "TextBlob sentiment analysis failed"}


def analyze_sentiment_vader(text):
    try:
        vader_analyzer = vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer()
        sentiment_score = vader_analyzer.polarity_scores(text)

        if sentiment_score['compound'] > 0.3:
            sentiment = "Positive"
        elif sentiment_score['compound'] < -0.3:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "compound_score": round(sentiment_score['compound'], 2),
            "positive_score": round(sentiment_score['pos'], 2),
            "negative_score": round(sentiment_score['neg'], 2),
            "neutral_score": round(sentiment_score['neu'], 2)
        }
    except Exception as e:
        logger.error(f"Error in VADER sentiment analysis: {str(e)}")
        return {"error": "VADER sentiment analysis failed"}
    
# Simulate emotion values
def get_emotions():
    emotions = {
        'happiness': random.randint(0, 100),
        'confidence': random.randint(0, 100),
        'engagement': random.randint(0, 100),
        'nervousness': random.randint(0, 100)
    }
    return emotions



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    try:
        data = request.json
        if not data or 'answer' not in data:
            return jsonify({"error": "No answer provided"}), 400
        
        answer = data['answer']
        
       
        textblob_result = analyze_sentiment_textblob(answer)
        vader_result = analyze_sentiment_vader(answer)

        return jsonify({
            "textblob": textblob_result,
            "vader": vader_result
        })
    
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Please upload a PDF, DOCX, or TXT file"}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        file.save(filepath)
        
        try:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(file)
            elif filename.endswith('.txt'):
                text = extract_text_from_txt(file)
            
            os.remove(filepath)
            
            sentiment_analysis = analyze_sentiment_textblob(text)
            
            return jsonify({
                "resume_text": text,
                "sentiment_analysis": sentiment_analysis,
                "word_count": len(text.split())
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


@app.route('/get_emotions', methods=['GET'])
def get_emotion_data():
    # Get simulated emotion values
    emotions = get_emotions()
    return jsonify(emotions)

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        resume_text = data.get('resume_text', '').strip()
        num_questions = min(max(int(data.get('num_questions', 5)), 1), 10)
        difficulty_level = data.get('difficulty_level', 'Simple')
        job_description = data.get('job_description', '').strip()
        
        if not resume_text:
            return jsonify({"error": "Resume text is required"}), 400

        if difficulty_level not in ['Simple', 'Moderate', 'Advanced']:
            return jsonify({"error": "Invalid difficulty level"}), 400

        openai.api_type = "azure"
        openai.api_base = app.config['OPENAI_ENDPOINT_URL']
        openai.api_version = "2024-05-01-preview"
        openai.api_key = app.config['OPENAI_API_KEY']

        chat_prompt = [
            {
                "role": "system",
                "content": (
                    f"You are a professional interviewer and act as real time interviwer for conducting a real-time interview. "
                    f"Generate {num_questions} {difficulty_level.lower()} {job_description}interview questions "
                    "based on the candidate's resume. Focus on their experience, skills, "
                    "and potential contributions. Make the questions engaging and relevant."
                )
            },
            {"role": "user", "content": resume_text}
        ]

        completion = openai.ChatCompletion.create(
            engine=app.config['OPENAI_DEPLOYMENT_NAME'],
            messages=chat_prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )

        questions = completion.choices[0].message["content"]
        
        return jsonify({
            "questions": questions,
            "metadata": {
                "difficulty": difficulty_level,
                "num_questions": num_questions,
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Question generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def recognize_speech_with_timeout(timeout_ms=30000, max_retries=3):
    """
    This function listens for a speech input for the specified timeout duration.
    It will retry up to max_retries times if no speech is recognized.
    
    timeout_ms: Maximum time (in milliseconds) to listen for speech.
    max_retries: Number of times to retry listening if no speech is captured.
    """
    attempt = 0
    while attempt < max_retries:
        try:
          
            speech_config = speechsdk.SpeechConfig(subscription=app.config['AZURE_API_KEY'], region=app.config['AZURE_SPEECH_REGION'])
            speech_config.speech_recognition_language = "en-US"
            
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
          
            result = recognizer.recognize_once_async().get(timeout=timeout_ms / 1000)  # Timeout in seconds
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            else:
                logger.warning(f"Attempt {attempt + 1} failed to recognize speech.")
                attempt += 1
                time.sleep(2)  
        except Exception as e:
            logger.error(f"Speech recognition error on attempt {attempt + 1}: {str(e)}")
            attempt += 1
            time.sleep(2)
    
    return None  


def synthesize_speech(text):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=app.config['AZURE_API_KEY'], region=app.config['AZURE_SPEECH_REGION'])
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        synthesizer.voice_name = app.config['SPEECH_SYNTHESIS_VOICE']
        synthesizer.speak_text_async(text)
    except Exception as e:
        logger.error(f"Speech synthesis error: {str(e)}")

@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        question = request.json.get('question', '')
        if not question:
            return jsonify({"error": "No question provided"}), 400

        synthesize_speech(f"Question: {question}")
        time.sleep(5) 

 
        answer = recognize_speech_with_timeout(timeout_ms=40000, max_retries=4)  # 30 seconds timeout and 3 retries
        
        if not answer:
            return jsonify({"error": "Failed to capture an answer from speech after multiple attempts."}), 400

        synthesize_speech(f"Answer: {answer}")
        textblob_result = analyze_sentiment_textblob(answer)
        vader_result = analyze_sentiment_vader(answer)

        return jsonify({
            "answer": answer,
            "textblob": textblob_result,
            "vader": vader_result
        })

    except Exception as e:
        logger.error(f"Ask question error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
