import gradio as gr
from LLM_API import HuggingFaceLLM_api
# from transformers import pipeline
import numpy as np
from stt import MyRecognizer
from tts import CoquaiSpeaker, PyttsxSpeaker
from scipy.io.wavfile import write
import threading

    
# from local_model import local_model


def recognize(audio) -> str:
    sr, y = audio
    # Save the audio back to a .wav file
    transcribed_audio_filename = '../data/audio_cache/user_query.wav'
    write(transcribed_audio_filename, sr, y)

    recognizer = MyRecognizer()
    transcribed_query = recognizer.recognize_google(transcribed_audio_filename, language='zh-CN')
    print("Transcribed user qquery:", transcribed_query)
    return transcribed_query


def ai_speak(text:str):
    # tts = CoquaiSpeaker()
    tts = PyttsxSpeaker()
    audio_output_filename = tts(text)
    return audio_output_filename



# def chat(question:str, history:list):
def chat(text_input, audio_input, chat_history=[]):
    global llm, bot_img_path
    still_gif_path = "../data/icon/still.gif"
    speak_gif_path = "../data/icon/speak.gif"

    match text_input, audio_input:
        case None, None:
            return "您没有提出问题，请利用文字输入或语音输入向聊天机器人提问~"
        case None, _: # only audio
            # user stt
            question = recognize(audio_input)
        case _, None:  # only text
            question = text_input
        case _, _:
            question = recognize(audio_input)
            question +=  text_input

    bot_img_path = speak_gif_path

    # llm inference
    response = llm(question, user_lang='zh')


    # Start a new thread to speak the response while returning the chat history
    threading.Thread(target=ai_speak, args=(response,)).start()
    # response_audio_path = ai_speak(response)


    combined_text = response + "\n"

    bot_img_path = still_gif_path

    # combine input & response
    # combined_text = "\n\n".join(chat_history)

    return combined_text






if __name__ == "__main__":
    # llm = local_model()
    llm = HuggingFaceLLM_api()
    bot_img_path = "data/icon/stil.gif"


    # User Interface
    blocks = gr.Blocks(css="styles.css")

    with blocks as app:
        with gr.Row():
            bot_img_block = gr.HTML(f'<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fgiphy.com%2Fexplore%2Fcute-robot&psig=AOvVaw1ocPVibc9hIZItyfXdFro3&ust=1719691393746000&source=images&cd=vfe&opi=89978449&ved=0CBAQjRxqFwoTCKCh9eaL_4YDFQAAAAAdAAAAABAh" alt="Bot" width="50" height="50">')
            chat_area = gr.TextArea(label="请多多了解我们的产品吧~")
        
        with gr.Row():
            audio_input = gr.Audio(sources=["microphone"], show_label=True, label="语音输入")
            text_input = gr.Textbox(placeholder="向我提问有关NewTechem产品的问题，我将为您提供答案。", label="文字输入")
            submit_btn = gr.Button("送出", scale=0.2)

        clear = gr.ClearButton([text_input, audio_input])

        with gr.Row():
            example_btn_1 = gr.Button("产品资讯1")
            example_btn_2 = gr.Button("产品资讯2")
            example_btn_3 = gr.Button("产品资讯3")

        def example_response(w:str):
            prepared_response = f"{w}"
            threading.Thread(target=ai_speak, args=(prepared_response,)).start()
            return prepared_response
        

        example_btn_1.click(example_response, [example_btn_1], chat_area)
        example_btn_2.click(example_response, [example_btn_2], chat_area)
        example_btn_3.click(example_response, [example_btn_3], chat_area)

        submit_btn.click(chat, [text_input, audio_input], [chat_area])

    app.launch()
    
    # inputs kwarg match the function input number
    # outputs kwarg match the function output number
    # demo = gr.Interface(
    #     fn=chat, 
    #     inputs=gr.Audio(sources=["microphone"]),
    #     outputs=[gr.HTML(label="Chat Box")],
    #     examples = [{"text":"产品资讯1"}, {"text":"产品资讯2"}, {"text":"产品资讯3"}],
    #     title="NewTechem 产品咨询AI机器人",
    #     description="向我提问有关NewTechem产品的问题，我将为您提供答案。",
    # )


    

   


# .launch(share=True)
