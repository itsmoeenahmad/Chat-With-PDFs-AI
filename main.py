# Importing Packages
import os 
from dotenv import load_dotenv
import streamlit as streamlit
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Loading the environment variables & initializing the Gemini For Accessing Its Models
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY") 
openai_api_key = os.getenv("OPENAI_API_KEY")

# Fucntion For Converting the PDF Data into text
def pdf_to_text(pdfs):
    text = ''
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


# Function For Converting the text into chunks
def text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function For Storing the chunks in vector DB (In our case it is FAISS)
def vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=openai_api_key
    )
    vector_store = FAISS.from_texts(chunks,embedding = embeddings)
    vector_store.save_local('faiss_index')


# Function For Conversational - Which will be done using Chain
def get_conversational_chain():
    
    prompt_template = """
    Your role is to answer the user's question based solely on the provided context.

    # Instructions:
    - Provide a complete and detailed answer using only the information from the context.
    - If the answer is not found in the context, respond with: "Sorry, the answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    """

    model = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0.3
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain = load_qa_chain(
        llm=model,
        prompt=prompt,
        chain_type='stuff'
    )
    return chain

# Function For Executing User Input
def user_input(user_question):
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=openai_api_key
    )

    new_db = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {'input_documents':docs,'question':user_question},
        return_only_outputs=True
    )

    print(response)
    streamlit.write(response['output_text'])

# Main Function
def main():
    streamlit.set_page_config("Chat With PDFs")
    streamlit.header('Chat With PDFs')
    streamlit.subheader("Powered By AI")

    user_question = streamlit.text_input("Ask anything from the PDF Files")

    if user_question:
        user_input(user_question)

    with streamlit.sidebar:
        streamlit.title("Menu")
        pdf_docs = streamlit.file_uploader(
            "Upload your PDF Files & Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if streamlit.button('Submit & Process'):
            with streamlit.spinner('Processing..'):
                raw_text = pdf_to_text(pdf_docs)
                chunks = text_to_chunks(raw_text)
                vector_store(chunks)
                streamlit.success('Done')

if __name__ == '__main__':
    main()



    




