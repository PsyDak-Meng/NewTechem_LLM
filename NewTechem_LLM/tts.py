# from transformers import pipeline
# from datasets import load_dataset
# import soundfile as sf
# import torch
# import pyttsx3
import torch
from TTS.api import TTS
import pyttsx3

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"




# * coquai tts
class CoquaiSpeaker:
    def __init__(self):
        # List available 🐸TTS models
        print(TTS().list_models())
        # Init TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.output_filename = 'coquai_speech.wav'
    
    def __call__(self, text:str):
        # Save wav to demo.wav
        print("Text to spech...\n", text)

        # Run TTS
        # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
        # Text to speech list of amplitude values as output
        # wav = self.tts.tts(text=text, speaker_wav="NewTechem_LLM\user_query.wav", language="en")
        # Text to speech to a file
        self.tts.tts_to_file(text=text, speaker_wav="../data/audio_cache/user_query.wav", language="en", file_path=self.output_filename)
        print("Voice file saved.")

        return self.output_filename

#! Test
# tts = CoquaiSpeaker()
# tts(text="哈啰! 我很乐意协助你介绍我们公司的产品之一 根据所提供的背景,我可以介绍R-PPPE,一种专门的无卤和无害环境的阻燃剂。 常规 PPPE是一种白色粉末成分,无毒,不含APP或TPP. 它具有较低的粉尘飞行特性,在燃烧时不产生腐蚀性气体,有效抑制产生烟雾,并且可以通过UL-V级阻燃剂.")




# * pyttsx3
class PyttsxSpeaker():
    def __init__(self):
        # Initialize the speech engine
        self.engine = pyttsx3.init()

        # Set properties before adding anything to speak
        self.engine.setProperty('rate', 180)  # Speed percent (can go over 100)
        self.engine.setProperty('volume', 0.9)  # Volume 0-1

        # Get available voices and set the voice to Chinese
        voices = self.engine.getProperty('voices')
        for voice in voices:
            # print(voice)
            if 'zh-' in voice.id.lower():  # Check if the voice is for Chinese
                # print("Chinese voice found.")
                self.engine.setProperty('voice', voice.id)
                break
        else:
            print("Chinese voice not found. Please ensure you have the appropriate language pack installed.")
            exit()

    def __call__(self, text:str):
        # Add text to the speech queue
        self.engine.say(text)

        self.engine.save_to_file(text, "../data/audio_cache/pyttsx_speech.wav")

        # Process and play the speech
        self.engine.runAndWait()

#! Test
# tts = PyttsxSpeaker()
# tts(text="哈啰! 我很乐意协助你介绍我们公司的产品之一 根据所提供的背景,我可以介绍R-PPPE,一种专门的无卤和无害环境的阻燃剂。 常规 PPPE是一种白色粉末成分,无毒,不含APP或TPP. 它具有较低的粉尘飞行特性,在燃烧时不产生腐蚀性气体,有效抑制产生烟雾,并且可以通过UL-V级阻燃剂.")






# # * Meta TTS model
# synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embedding = torch.tensor(embeddings_dataset[7406]["xvector"]).unsqueeze(0)
# # You can replace this embedding with your own as well.

# speech = synthesiser("Hello, here is our star product!", forward_params={"speaker_embeddings": speaker_embedding})

# sf.write("meta_speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

