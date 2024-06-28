from flask import Flask, render_template, request, jsonify
from scipy.io.wavfile import write
import threading

# Import your custom modules
from NewTechem_LLM.LLM_API import HuggingFaceLLM_api
from NewTechem_LLM.stt import MyRecognizer
from NewTechem_LLM.tts import CoquaiSpeaker, PyttsxSpeaker

app = Flask(__name__)
llm = HuggingFaceLLM_api()

def recognize(audio) -> str:
    sr, y = audio
    transcribed_audio_filename = 'static/user_query.wav'
    write(transcribed_audio_filename, sr, y)

    recognizer = MyRecognizer()
    transcribed_query = recognizer.recognize_google(transcribed_audio_filename, language='zh-CN')
    print("Transcribed user query:", transcribed_query)
    return transcribed_query

def ai_speak(text: str):
    tts = PyttsxSpeaker()
    audio_output_filename = tts(text)
    return audio_output_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_route():
    audio = request.files['audio']
    chat_history = request.form.getlist('history')
    sr, y = audio
    question = recognize((sr, y))

    response = llm(question, user_lang='zh')

    threading.Thread(target=ai_speak, args=(response,)).start()

    user_gif = "/static/user.gif"
    bot_gif = "/static/bot.gif"

    chat_history.append(f'<div><img src="{user_gif}" alt="User" width="50" height="50"/> User: {question}</div>')
    chat_history.append(f'<div><img src="{bot_gif}" alt="Bot" width="50" height="50"/> Bot: {response}</div>')
    combined_text = "<br>".join(chat_history)

    return jsonify(combined_text)

if __name__ == "__main__":
    app.run(debug=True)
