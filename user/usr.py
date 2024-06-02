import boto3
import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Set up AWS credentials
boto3.setup_default_session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

## downloading embedding files to local folder.
folder_path="/tmp/"


## defining uuid
def get_unique_id():
    return str(uuid.uuid4())

## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

# Define the LLM with adjustable parameters
def get_llm(max_gen_len=512, temperature=0.7, top_p=0.9):
    llm = Bedrock(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock_client,
        model_kwargs={
            'max_gen_len': max_gen_len,
            'temperature': temperature,
            'top_p': top_p
        }
    )
    return llm



# get_response()
def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """
    [INST]You are an AI assistant with expertise in analyzing and extracting information from PDF documents. You are capable of understanding various topics and providing detailed, accurate, and relevant responses based on the content of the document provided.[/INST]

    Task: Analyze the provided PDF document and answer the user's questions based on its content. Follow these guidelines:

    Carefully read and comprehend the relevant sections of the PDF document to understand the context of the question.
    - Example: "Based on the introduction section, the main focus of this document is..."

    Provide a clear, concise, and accurate answer to the user's question.
    - Example: "The key findings from the research study mentioned are..."

    When appropriate, reference specific sections, headings, or page numbers from the PDF to support your answer.
    - Example: "As mentioned on page 15 under the section 'Results'..."

    If the information is not available in the document, inform the user honestly.
    - Example: "The document does not provide details on this topic."

    Ensure responses are professional, informative, and helpful.
    - Example: "Thank you for your question. Here is the information based on the provided document."

    Question: {question}

    Context from PDF:
    {context}

    Assistant:
    """

    # Example usage
    # question = "What are the main conclusions of the report?"
    # context = "The report concludes that implementing renewable energy sources significantly reduces greenhouse gas emissions and offers a sustainable alternative to fossil fuels. Key points include the reduction of carbon footprint, economic benefits, and long-term sustainability."

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']

    


## main method
def main():
    st.header("This is the frontend client site for users to interact with the app ")

    load_index()

    ## use below to display if the vector stores are downloaded in local folder.
    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

    ## create index (load the saved files and create index of user query and relevant chunk or similarity search)
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path = folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("INDEX IS READY")

    # Querying llm.
    question = st.text_input("Please ask your question.")

     # Adjustable parameters
    st.write("Adjustable Parameters:")

    # Max Generation Length
    max_gen_len = st.number_input("Enter Max Gen Length", min_value=50, max_value=1024, value=512, key='max_gen_len_input')

    # Temperature
    temperature = st.number_input("Enter Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1, key='temperature_input')

    # Top-p
    top_p = st.number_input("Enter Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.1, key='top_p_input')

    if st.button("Ask a Question"):
        with st.spinner("Querying ..."):


            llm = get_llm(max_gen_len, temperature, top_p)  # User-adjusted parameters overwrite defaults
            st.write(get_response(llm, faiss_index, question))
            st.success("Done")


if __name__=="__main__":
    main()