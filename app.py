import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
# from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import xml.etree.ElementTree as ET
import requests
import tensorflow as tf

# Add GPU device and use
device = tf.config.list_physical_devices('GPU')

def load_urls_from_sitemap(sitemap_path):
    """Load URLs from a sitemap.xml file."""
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}  
    urls = [url.text for url in root.findall('ns:url/ns:loc', namespaces)]
    return urls

def load_docs_from_urls(urls):
    docs = []
    for url in urls:
        try:
            response = requests.get(url)
            # Check if the request was successful
            if response.status_code == 200:
                docs.append(response.text)  # Add the content of the web page to the list
            else:
                print(f"Failed to retrieve {url}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}. Exception: {e}")
    return docs

sitemap_path = 'sitemap.xml'  
urls = load_urls_from_sitemap(sitemap_path)

# Load documents from URLs
docs = load_docs_from_urls(urls)

with tf.device('/device:GPU:0'):
    def process_input(question):
        model_local = ChatOllama(model="mistral")

        # Load urls from sitemap.xml
        sitemap_path = "sitemap.xml"
        # Convert string of urls to list of urls
        # urls = urls.split('\n')
        urls = load_urls_from_sitemap(sitemap_path)

        docs = load_docs_from_urls(urls)
        # docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        # 1. Split data into chunks
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 7500, chunk_overlap = 100)
        doc_splits = text_splitter.split_documents(docs_list)

        # 2. Convert documents to embeddings and store them
        vectorstore = Chroma.from_documents(
            documents = doc_splits,
            collection_name = "rag-chroma",
            embedding = embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"),
        )

        retrieval = vectorstore.as_retriever()
        # retrieval = vectorstore.as_retriever(search_kwargs={"k":1})

        # 3. Before RAG
        before_rag_template = "What is {topic}"
        before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
        before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
        before_rag_output = before_rag_chain.invoke({"topic": "Ollama"})

        # 4. After RAG
        after_rag_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """

        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
            {"context": retrieval, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
        )
        after_rag_output = after_rag_chain.invoke({question})

        return before_rag_output, after_rag_output

    # Define gradio interface
    iface = gr.Interface(
        fn=process_input,
        inputs=[
            # gr.Textbox(label="|Enter URLs sepretated by new lines"),
            gr.Textbox(label="Question")
        ],
        outputs="text",
        title="Document Query with Ollama",
        description="Enter the URLs of the documents you want to query and ask a question about the content."
    )
    iface.launch()
