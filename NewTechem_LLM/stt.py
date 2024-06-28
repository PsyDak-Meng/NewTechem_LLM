import speech_recognition as sr

class MyRecognizer(sr.Recognizer):
    def __init__(self):
        super().__init__()
        self.r = sr.Recognizer()

    def recognize_google(self, audio_file:str, language:str='en-US') -> str:
        speech = sr.AudioFile(audio_file)

        with speech as source:
            audio = self.r.record(source)

        text = self.r.recognize_google(audio, language=language)
        print("Google cloud speech recognized user input:", text)
        return text
    

#! Test
# recognizer = MyRecognizer()
# print(recognizer.recognize_google(audio_file = 'meta_speech.wav'))