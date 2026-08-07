import streamlit as st
from app import (
    create_vector_db,
    retrieve_context,
    generate_answer,
    search_web
)

st.title("🤖 AI Document Research Assistant")
st.write("Upload a PDF and ask questions!")

# ---- Initialize all session_state keys up front ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "show_web_option" not in st.session_state:
    st.session_state.show_web_option = False
if "saved_query" not in st.session_state:
    st.session_state.saved_query = ""
if "doc_answer" not in st.session_state:
    st.session_state.doc_answer = None
if "retrieved_results" not in st.session_state:
    st.session_state.retrieved_results = None
if "web_answer" not in st.session_state:
    st.session_state.web_answer = None

# ---- PDF upload ----
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None and st.session_state.vectorstore is None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF... Please wait..."):
        st.session_state.vectorstore = create_vector_db(uploaded_file.name)

    st.success("PDF uploaded successfully!")

# ---- Question input ----
query = st.text_input("Ask your question")

if st.button("Ask"):
    if uploaded_file is None:
        st.warning("Please upload a PDF first!")
    elif not query.strip():
        st.warning("Please enter a question!")
    else:
        # Reset any previous web-search state for a new question
        st.session_state.show_web_option = False
        st.session_state.web_answer = None

        with st.spinner("Generating answer..."):
            context, results = retrieve_context(st.session_state.vectorstore, query)
            answer = generate_answer(context, query)

        print("Answer before if:", repr(answer))

        # Save everything we'll need on LATER reruns (when Yes/No get clicked)
        st.session_state.doc_answer = answer
        st.session_state.retrieved_results = results
        st.session_state.saved_query = query

        if "INSUFFICIENT_CONTEXT" in answer:
            st.session_state.show_web_option = True

# ---- Everything below runs on EVERY rerun, not just after "Ask" ----
# This is what makes the Yes/No buttons actually work across reruns.

if st.session_state.doc_answer and not st.session_state.show_web_option:
    st.subheader("Answer")
    st.write(st.session_state.doc_answer)

if st.session_state.retrieved_results:
    st.subheader("Retrieved Chunks")
    for i, (doc, score) in enumerate(st.session_state.retrieved_results):
        with st.expander(f"Chunk {i+1}"):
            st.write(f"Similarity Score: {score:.3f}")
            st.write(doc.page_content)

if st.session_state.show_web_option:
    st.warning("I couldn't find the answer in the document.")
    st.write("Would you like me to search the web?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes"):
            with st.spinner("Searching the web..."):
                st.session_state.web_answer = search_web(st.session_state.saved_query)
            st.session_state.show_web_option = False

    with col2:
        if st.button("No"):
            st.session_state.web_answer = "Okay! I won't search the web."
            st.session_state.show_web_option = False

if st.session_state.web_answer:
    st.subheader("Web Answer")
    st.write(st.session_state.web_answer)