import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
import requests
from NewTechem_LLM.translator import MyTranslator
from huggingface_hub import login
import json
# from langchain_core.runnables.base import RunnableSequence
# from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub
# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate


from NewTechem_LLM.RAG import RAG
from NewTechem_LLM.utils import timeit, filter_english


login(token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'))
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"



class serverless_llm:
    def __init__(self, API_URL:str):
        self.API_URL = API_URL
        self.headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}"}

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()
	
    def __call__(self, user_query:str, prompt:str, context:str):
        context = filter_english(context)
        print("filtered context: ", context)
        payload = {
            "inputs": prompt.format(context=context, question=user_query),
        }
        # payload = {
        #         "inputs": {
        #             "question": user_query,
        #             "context": prompt.format(context=context)
        #         },
        #     }
        print("The input chain is:\n ", json.dumps(payload))
        output = self.query(payload)
        # print("The answer is:\n ", json.dumps(output))

        # output formatting
        output = output[0]["generated_text"].split("Give your helpful answer below:")[1].strip()
        print('Splitted answer: \n',output)
        return output



class HuggingFaceLLM_api:
    def __init__(self, repo_id:str=None,
                 model_kwargs={"temperature":1e-10}):
        
        print('Initializing Hugging Face API token...')
        self.HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
        

        prompt_template = """You are a chemical professional, introduce the product in the Context for our company--NewTechem. Please follow the following rules:
        1. If you don't know the answer and don't see the answer in the Context, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links". You will be 
        2. If you find the answer, state the product's full name and introduce it thoroughly and concisely in a description within five sentences maximum.
        3. If you cannot find a prooper answer, refer user to Newtechem company link: www.new-techem.com.
        Context: {context}
        Question: {question}

        Give your helpful answer below:
        """

        # self.PROMPT = PromptTemplate(
        #     template = prompt_template,
        #     input_variables=["context", "question"],
        # )

        self.PROMPT = prompt_template
        print('Prompt template setup!')



        # * useing HuggingFaceHub
        # repo_id = "mistralai/Mistral-7B-v0.3"
        # self.llm = HuggingFaceHub(
        #     repo_id=repo_id,
        #     model_kwargs={"temperature":0.1, "max_length":500})
        # * using serverless api
        self.llm = serverless_llm(API_URL=API_URL)
        print('HuggingFace model connected successfully!')

        print("Setting up RAG...")
        self.rag = RAG()
        print('RAG setup in llm_api!')

        self.en2zh_translator = MyTranslator(from_code='en', to_code='zh')
        self.zh2en_translator = MyTranslator(from_code='zh', to_code='en')
        print('Translators setup!')

        

    @timeit
    def __call__(self, user_query:str,  user_lang:str, default_lang:str='en'):
        
        def clip_output(output:str):
             puncs = ['.','!','?']
            #  print(output)
             lengths = [len(output.split(punc)) for punc in puncs]
             if not output.endswith(tuple(puncs)): # don't clip if there is no punctuation
                print('ends with:', output[-1])
                punc = puncs[lengths.index((max(lengths)))]
                output = punc.join(output.split(punc)[:-1])
                output += '.'
                print("clipped output.")
            #  print('clipped output:', output)
             return output
         
        def inference(user_query:str):
            # output = self.rag.retrievalQA(llm=self.llm, PROMPT=self.PROMPT, query=user_query)
            output = self.llm(user_query, self.PROMPT, self.rag.retrieve(user_query))
            output = clip_output(output)
            return output
        


        # if input zh, trnalslate to model default language
        match user_lang, default_lang:
                case 'zh', default_lang:
                    user_query = self.zh2en_translator.translate(user_query)
                case _:
                    pass

        output = inference(user_query)
        output = self.en2zh_translator.translate(output)
        print('Output:', output)

        return output



# ! Test
# user_query = "可以介绍一种阻燃剂给我吗？"
# llm_chain = HuggingFaceLLM_api()
# print(llm_chain(user_query=user_query, user_lang='zh'))


