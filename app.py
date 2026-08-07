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
1. If the context contains information relevant to the question — even if it's
   spread across multiple chunks — synthesize it into a clear answer. For broad
   or "what is this about" style questions, summarizing the relevant chunks
   IS a valid, correct answer.
2. If the context has NOTHING relevant to the question at all, reply with
   EXACTLY: INSUFFICIENT_CONTEXT
3. Do not invent facts that aren't present anywhere in the context.
4. Do not answer using your own outside knowledge — only the context provided.

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
def verify_answer(context, query, answer):
    """
    Checks whether the given answer is actually supported by the context.
    Returns True if supported, False if not.
    """
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

    verify_prompt = f"""
You are verifying whether an answer is properly supported by the given context.

Context:
{context}

Question:
{query}

Answer:
{answer}

Is this answer fully and directly supported by the context above?
Reply with EXACTLY one word: YES or NO.
"""

    response = llm.invoke(verify_prompt)
    result = response.content.strip().upper()
    print("Self-correction verification result:", result)
    return "YES" in result


def generate_answer_with_self_correction(vectorstore, query):
    """
    Retrieves context, generates an answer, and verifies it against the context.
    If the answer isn't supported, retries once with a wider retrieval + regeneration.
    Returns: answer, context, results, self_correction_triggered (bool)
    """
    context, results = retrieve_context(vectorstore, query)
    answer = generate_answer(context, query)

    self_correction_triggered = False

    # Only verify if we got a real answer, not an explicit "insufficient context" flag
    if "INSUFFICIENT_CONTEXT" not in answer:
        is_supported = verify_answer(context, query, answer)

        if not is_supported:
            self_correction_triggered = True
            print("Self-correction triggered — retrying with wider retrieval")

            # One additional retrieval attempt with more chunks
            results = vectorstore.similarity_search_with_score(query, k=8)
            context = "\n\n".join([doc.page_content for doc, score in results])
            answer = generate_answer(context, query)

    return answer, context, results, self_correction_triggered


def search_web_with_self_correction(query):
    """
    Performs a web search + answer, verifies it, and retries once with
    more results if the answer isn't well supported.
    Returns: answer, self_correction_triggered (bool)
    """
    answer = search_web(query)
    self_correction_triggered = False

    is_supported = verify_answer(
        context="(web search results used to generate this answer)",
        query=query,
        answer=answer
    )

    if not is_supported:
        self_correction_triggered = True
        print("Self-correction triggered on web answer — retrying")
        answer = search_web(query)  # simplest retry; could widen max_results instead

    return answer, self_correction_triggered