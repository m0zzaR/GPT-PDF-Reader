import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from pdfminer.high_level import extract_text
import os

class InvalidAPIKey(Exception):
    pass

# Asks for OpenAI API Key. Outputs error if incorrect API Key length.
while True:
    try:
        api_key = input("Input API, key (If invalid, code will produce an error): ")
        if len(api_key) < 30 or len(api_key) > 128:
            raise InvalidAPIKey("Api key not correct length")
        break
    except InvalidAPIKey as fnf_error:
        print(fnf_error)

os.environ["OPEN_API_KEY"] = api_key

# Asks for directory of PDF file. Outputs error if incorrect file type or directory notation.
while True:
    try:
        root_dir = input("Input directory of pdf (Do not include quotes): ")
        if not os.path.isfile(root_dir) or not root_dir.endswith(".pdf"):
            raise FileNotFoundError("The provided directory does not exist or is not a pdf.")
        break
    except FileNotFoundError as fnf_error:
        print(fnf_error)

# Extracts text from PDF
reader = extract_text(root_dir)

# Splits PDF text into multiple text files so that ChatGPT token limit is not reached.
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 2000,
    chunk_overlap = 100,
    length_function = len,
    )
texts = text_splitter.split_text(reader)

# Turns text files into vector based embeddings so that ChatGPT can read all files at once. 
# (Not to sure how exactly this works, the magic of LangChain is at work here)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Takes in user prompt in order to interact with ChatGPT
print("At any time, use 'close' to stop running the program")
user_input = ''
while user_input != 'close':
    print("-----------------------------------------")
    user_input = input("What would you like to ask about the pdf: ")
    if user_input == 'close':
        break
    docs = docsearch.similarity_search(user_input)
    print(f"ChatGPT: {chain.run(input_documents=docs, question=user_input)}")
os._exit(0)





