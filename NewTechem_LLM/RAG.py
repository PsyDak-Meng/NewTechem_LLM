import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.llms import HuggingFacePipeline
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
import glob
from typing import List
# from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from tqdm import tqdm
import docx2txt, re


from NewTechem_LLM.utils import timeit

class docx_loader:
    def __init__(self):
        pass
    def load(self, file_path) -> str:
        # extract text
        text = docx2txt.process(file_path)
        # Regex pattern to match one or more consecutive newline characters
        pattern = r'\n+'

        # Replace multiple newlines with a single newline
        text = [re.sub(pattern, '\n', txt).strip() for txt in text.split('\n') if len(txt)>0]
        # print(text)
        return '\n'.join(text)


class CustomDirectoryLoader:
    def __init__(self, directory_path: str, glob_pattern: str = "*.*", mode: str = "single"):
        """
        Initialize the loader with a directory path and a glob pattern.
        :param directory_path: Path to the directory containing files to load.
        :param glob_pattern: Glob pattern to match files within the directory.
        :param mode: Mode to use with UnstructuredFileLoader ('single', 'elements', or 'paged').
        """
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.mode = mode

    def load(self) -> str:
        """
        Load all files matching the glob pattern in the directory using UnstructuredFileLoader.
        :return: List of Document objects loaded from the files.
        """
        texts = ""

        # Construct the full glob pattern
        full_glob_pattern = f"{self.directory_path}/{self.glob_pattern}"
        # print(glob.glob(full_glob_pattern))

        # Iterate over all files matched by the glob pattern
        for file_path in tqdm(glob.glob(full_glob_pattern)):
            # Use UnstructuredFileLoader to load each file
            # loader = UnstructuredFileLoader(file_path=file_path, mode=self.mode)
            loader = docx_loader()
            docs = loader.load(file_path)
            # documents.extend(docs)
            # texts.append({'text':docs})
            texts += docs
             
        return texts
    
# ! Test
# # directory_loader = CustomDirectoryLoader(directory_path="./documents", glob_pattern="*.@(pdf|txt|csv|doc|docx)", mode="elements")
# directory_loader = CustomDirectoryLoader(directory_path="./documents", glob_pattern="*.docx", mode="elements")
# print('loader created')
# documents = directory_loader.load()
# print('documents loaded')
# print(documents[0], len(documents))

# ! Different loaders
# from langchain.document_loaders import DirectoryLoader
# from langchain.document_loaders.pdf import PyMuPDFLoader
# from langchain.document_loaders.xml import UnstructuredXMLLoader
# from langchain.document_loaders.csv_loader import CSVLoader

# # Define a dictionary to map file extensions to their respective loaders
# loaders = {
#     '.pdf': PyMuPDFLoader,
#     '.xml': UnstructuredXMLLoader,
#     '.csv': CSVLoader,
# }

# # Define a function to create a DirectoryLoader for a specific file type
# def create_directory_loader(file_type, directory_path):
#     return DirectoryLoader(
#         path=directory_path,
#         glob=f"**/*{file_type}",
#         loader_cls=loaders[file_type],
#     )

# # Create DirectoryLoader instances for each file type
# pdf_loader = create_directory_loader('.pdf', '/path/to/your/directory')
# xml_loader = create_directory_loader('.xml', '/path/to/your/directory')
# csv_loader = create_directory_loader('.csv', '/path/to/your/directory')

# # Load the files
# pdf_documents = pdf_loader.load()
# xml_documents = xml_loader.load()
# csv_documents = csv_loader.load()



class RAG():
    def __init__(self, directory:str='../data'):
        # * Load Texts
        loader = CustomDirectoryLoader(directory_path=directory, glob_pattern="*.docx", mode="elements")
        texts = loader.load()
        print(f"RAG loaded raw texts: has {len(texts)} characters")

        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        pages = text_splitter.split_text(texts)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_after_split = text_splitter.create_documents(pages)

        # loader = Docx2txtLoader.load('documents\PDS(N) R-970P V1 CN 20240522.docx')
        # docs_before_split = loader.load()
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size = 700,
        #     chunk_overlap  = 50,
        # )
        # docs_after_split = text_splitter.split_documents(docs_before_split)

        # print(f"Sample document chunk has {len(docs_after_split[0].page_content)} characters as follows:\n {docs_after_split[0]}")


        # * Text Embedding
        huggingface_embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",  # BAAI/bge-small-en-v1.5 or alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
            model_kwargs={'device':'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        sample_embedding = np.array(huggingface_embeddings.embed_query(docs_after_split[0].page_content))
        # print("Sample embedding of a document chunk: ", sample_embedding)
        print("Size of a sample document chunk embedding: ", sample_embedding.shape)


        # * Vectorstore
        self.vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

        # ! Test retrieval
        # query = """What were the trends in median household income across
        #    different states in the United States between 2021 and 2022."""  
        # # Sample question, change to other questions you are interested in.
        # relevant_documents = vectorstore.similarity_search(query)
        # print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
        # print(relevant_documents[0].page_content)

        # Use similarity searching algorithm and return 3 most relevant documents.
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    def retrievalQA(self, llm, PROMPT:str, query:str) -> str:
        retrievalQA = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = retrievalQA.invoke({"query": query})
        print(result['result'])

        # retreival logs
        relevant_docs = result['source_documents']
        print(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')
        print("*" * 100)
        for i, doc in enumerate(relevant_docs):
            print(f"Relevant Document #{i+1}:\nSource file: {doc.metadata['source']}, Page: {doc.metadata['page']}\nContent: {doc.page_content}")
            print("-"*100)
            print(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')

        return result['results']
   
    def retrieve(self, query:str) -> str:
        relevant_documents = self.vectorstore.similarity_search(query)
        print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
        # print(relevant_documents[0].page_content)
        retrieved_documents = " ".join([doc.page_content.strip() for doc in relevant_documents])
        print(f"RAG retrieved {len(retrieved_documents)} characters.")
        return retrieved_documents




# ! Test
# rag = RAG()
