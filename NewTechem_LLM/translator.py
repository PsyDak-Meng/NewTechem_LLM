# from googletrans import Translator
# class GoogleTranslator:
#     def __init__(self):
#         self.translator = Translator(service_urls=[
#             'translate.google.com',
#             'translate.google.co.kr',
#             ])
        
#     def translate(self, text, dest:str='en', src:str='auto'):
#         # src = self.translator.detect(text)
#         return self.translator.translate(text, dest=dest, src=src).strip()
    

# from translate import Translator
# class MyTranslator:
#     def __init__(self, to_lang:str='en'):
#         # self.translator = Translator(provider='microsoft', to_lang=to_lang, secret_access_key=secret, pro=True)
#         self.translator = Translator(to_lang=to_lang)

#     def translate(self, text):
#         return self.translator.translate(text).strip()


import argostranslate.package
import argostranslate.translate #https://github.com/argosopentech/argos-translate

class MyTranslator:
    def __init__(self, from_code:str='zh', to_code:str='en'):
        self.from_code = from_code
        self.to_code = to_code

        # Download and install Argos Translate package
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == self.from_code and x.to_code == self.to_code, available_packages
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

    def translate(self, text):
        return argostranslate.translate.translate(text, self.from_code, self.to_code).strip()


# ! Test
# translator  = MyTranslator()
# print(translator.translate("化学药剂HG-380是界面活性剂。"))