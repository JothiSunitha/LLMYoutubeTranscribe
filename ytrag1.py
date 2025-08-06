import whisper

from langchain.tools import tool
#from pytube import YouTube
from pytubefix import YouTube

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda

from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate


from langchain_mistralai.chat_models import ChatMistralAI
import os

@tool
def download_transcribe(video_url:str):
    """This extracts the text from the given video URL"""
    yt = YouTube(video_url)
    audio = yt.streams.filter(only_audio = True).order_by('abr').desc().first()
    audio_path = audio.download()
    print(audio.default_filename)
    
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def embed_store(text:str):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 30)
    chunks = splitter.split_text(text)
    
    embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")    
    db = Chroma.from_texts(chunks, embed_model, collection_name='yt_chunks')
    
    return db

runnable = RunnableLambda(embed_store)

os.environ["MISTRAL_API_KEY"] = "veMgdofLWqX4XlTw3tFduXu4kTTkGwEK"
llm = ChatMistralAI(model="mistral-small")

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

def run_query(query: str, db) -> str:
    # Step 1: Create a retriever from your Chroma vector store
    retriever = db.as_retriever(search_type="similarity", k=5)

    # Step 2: Define the prompt template
    prompt = PromptTemplate.from_template(
        "Use the following context to answer the question:\n{context}\n\nQuestion: {question}"
    )

    # Step 3: Wrap your LLM into a combine_documents_chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Use it in StuffDocumentsChain
    
    combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
    )
#     combine_documents_chain = StuffDocumentsChain(
#     llm_chain=llm_chain
#     )    

    # Step 4: Initialize RetrievalQA correctly
    chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_documents_chain
    )

    # Step 5: Run the query
    return chain.run(query)


import streamlit as st

st.title("YouTube RAG Explorer")

video_url = st.text_input("Enter YouTube URL")
query = st.text_input("Ask a question")

if st.button("Process & Query"):
    st.write("🎯 Button clicked!")

with st.spinner("Downloading and transcribing..."):
    transcript = download_transcribe(video_url)
    st.write("Transcript preview:", transcript[:300])

with st.spinner("Embedding transcript..."):
    db = embed_store(transcript)

with st.spinner("Running query..."):
    response = run_query(query, db)

print(response)

# if st.button("Process & Query"):
#     transcript = download_and_transcribe(video_url)
#     db = embed_and_store(transcript)
#     response = run_query(query, db)
#     st.markdown("### Answer")
#     st.write(response)