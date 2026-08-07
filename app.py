from dotenv import load_dotenv
import os
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
print(os.getenv("GEMINI_API_KEY"))

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

def create_vector_db(pdf_path):
 loader = PyPDFLoader(pdf_path)
 documents = loader.load()

 splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
 )

 chunks = splitter.split_documents(documents)

 embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
 )
 vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="database"
) 
 return vectorstore

def retrieve_context(vectorstore, query):

    results = vectorstore.similarity_search_with_score(
        query,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc, score in results]
    )

    return context, results

def generate_answer(context, query):

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )
    
    print("="*50)
    print("CONTEXT SENT TO GEMINI")
    print(context)
    print("="*50)
    prompt = f"""
You are an AI Document Research Assistant.

Your job is to answer ONLY from the provided context.

Rules:
1. If the answer is explicitly present in the context, answer it.
2. If the context is unrelated to the user's question, do NOT guess.
3. Do NOT use your own knowledge.
4. Do NOT summarize unrelated context.
5. If the context does not directly answer the question, reply with EXACTLY:

INSUFFICIENT_CONTEXT

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    print(response.content)
    return response.content

def search_web(query):

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

    web_results = tavily.search(
        query=query,
        max_results=3
    )

    web_context = "\n\n".join(
        [
            result["content"]
            for result in web_results["results"]
        ]
    )

    web_prompt = f"""
You are a helpful AI assistant.

Answer using ONLY the web search results.

Web Results:

{web_context}

Question:

{query}
"""

    response = llm.invoke(web_prompt)
    print("search_web() was called")
    return response.content