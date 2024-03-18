import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

# model_local = ChatOllama(model="mistral")

def process_input(file_obj, question):
    model_local = ChatOllama(model="mistral", device="cuda")
    
    if file_obj.name.endswith('.pdf'):
        loader = PyPDFLoader(file_obj.name)
    elif file_obj.name.endswith('.txt'):
        loader = TextLoader(file_obj.name)
    else:
        return "Unsupported file format. Please upload a PDF or TXT file."
    
    docs_list = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    after_rag_template = """Answer the question based only on the following context: {context} Question: {question} """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} |
        after_rag_prompt |
        model_local |
        StrOutputParser()
    )
    
    return after_rag_chain.invoke(question)

# Define Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.File(label="Upload Document"), gr.Textbox(label="Question")],
    outputs="text",
    title="Document Query with Ollama",
    description="Upload a document and ask a question to query the document."
)

iface.launch()