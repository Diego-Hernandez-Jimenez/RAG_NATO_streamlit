
from time import sleep
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# CONSTANTS

EMBEDDING_MODEL = 'models/text-embedding-004'
DB_DIR = './vector_db_alta_v2'
SEARCH_TYPE = 'similarity'
N_DOCS_RETRIEVED = 2
REPHRASE_MODEL = 'llama-3.3-70b-versatile'

USER_AVATAR = './images/Comp_RGB_inv.jpg'
ASSISTANT_AVATAR = './images/Comp_RGB.jpg'
NATO_FAVICON = './images/NATO_favicon.png'
SPEED = 10

# RAG RELATED FUNCTIONS

@st.cache_resource
def build_vector_database():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(collection_name='alta_handbook', persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": N_DOCS_RETRIEVED})

    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(selected_model, retriever):
    base_template = """You are a NATO assistant for question-answering tasks.
    If you don't know the answer, respond with: "Based on my knowledge, I can't provide an answer." Keep your responses concise.
    Use the following context to answer the question:

    Context:
    {context}

    Question:
    {question}
    """
    base_prompt = ChatPromptTemplate.from_template(base_template)
    chat_llm = ChatGroq(model=selected_model)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | base_prompt
        | chat_llm
        | StrOutputParser()
    )

    return rag_chain

@st.cache_resource
def create_rephrase_chain():
    rephrase_template = """You are a query rephraser. Given a chat history and the latest user query, your task is to rephrase the query if it implicitly references topics in the chat history.
    If the query does not reference the chat history, return it as is. Do not provide explanations, just return the rephrased or original query.

    Chat history:
    {chat_history}

    Latest user query:
    {input}
    """
    rephrase_prompt = ChatPromptTemplate.from_template(rephrase_template)
    rephraser_llm = ChatGroq(model=REPHRASE_MODEL)
    rephrase_chain = rephrase_prompt | rephraser_llm | StrOutputParser()

    return rephrase_chain

# STREAMLIT RELATED FUNCTIONS

def typewriter_effect(text, speed, allow_html):
    # https://discuss.streamlit.io/t/st-write-typewritter/43111/3
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text, unsafe_allow_html=allow_html)
        sleep(1 / speed)

def restart_chat():
    st.session_state.pop("messages", None)

# -----------------------------

# STREAMLIT

disclaimer = """This app and its author are not affiliated with NATO and do not represent the organization in any official capacity.
The content provided is based on The NATO Alternative Analysis Handbook
but is intended for general informational purposes only and does not reflect NATO's views or policies.
"""
st.set_page_config(
    page_title="NATO Chatbot",
    page_icon=NATO_FAVICON, # 'https://cdn3.emoji.gg/emojis/5667-nato.png'
    menu_items={'About': disclaimer, 'Report a Bug': 'mailto:diego.her.jimenez@gmail.com'}
)

# Custom CSS for font change
# https://fonts.google.com
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Solway:wght@300;400;500;700;800&display=swap');

    body * {
        font-family: 'Solway', sans-serif !important;
        font-weight: 400;
        font-style: normal;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with description and model selector
with st.sidebar:
    left_col, cent_col, right_col = st.columns(3)
    with cent_col:
        st.image(NATO_FAVICON, use_container_width=True)
    st.title("Nate: Your NATO QA Assistant for Alternative Analysis")
    st.markdown("""
    This AI assistant is designed to help answer questions related to Alternative Analysis, a framework for intelligent decision-making.
    Developed by NATO, this framework offers a set of techniques that can be applied across various domains.
    However, this assistant bases its responses exclusively on [The NATO Alternative Analysis Handbook](https://www.act.nato.int/wp-content/uploads/2023/05/alta-handbook.pdf)
    """)
    selected_model = st.selectbox('Chat model', ['mixtral-8x7b-32768', 'llama3-70b-8192', 'gemma2-9b-it'], on_change=restart_chat)
    st.markdown('#')
    st.markdown('#')
    st.button("Clear conversation", on_click=restart_chat, type='tertiary', icon='🗑️')


st.markdown('#')
st.markdown('#')

# ---------------------------------------------
# build vector database
retriever = build_vector_database()

# instantiate llms
rag_chain = create_rag_chain(selected_model, retriever)
rephrase_chain = create_rephrase_chain()
# ---------------------------------------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested questions (only at the beginning)
# We display the questions with the emojis, but only the question itself is saved as input
# (see format_func below)
suggested_questions = {
    'What is Alternative Analysis?': 'What is Alternative Analysis? 🤔',
    'How can I apply SWOT step by step?': 'How can I apply SWOT step by step? 📝',
    'How can I manage dysfunctional behavior in a AltA session?': 'How can I manage dysfunctional behavior in a AltA session? ⚠️',
    'Point me to some resources to learn about Six Thinking Hats': 'Point me to some resources to learn about Six Thinking Hats 🎩'
}

if len(st.session_state.messages) == 0:
    selected_question = st.pills(
        "Suggested questions:", suggested_questions.keys(),
        selection_mode='single',
        format_func=lambda option: suggested_questions[option]
    )
    if selected_question:
        st.session_state.messages.append({"role": "user", "content": selected_question, "avatar": USER_AVATAR})
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# Ensure the chat input box is always visible
user_input = st.chat_input("Ask anything about AltA", max_chars=500)
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": USER_AVATAR})
    st.rerun()

# Generate assistant response if user input exists
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_input = st.session_state.messages[-1]["content"]
    # Generate assistant response
    if len(st.session_state.messages) == 1:
        final_input = user_input
    # when there are previous interactions (history)...
    else:
        # ... keep only the last two interactions in "memory"
        trimmed_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-2:]])
        # based on the context given by the memorized conversation, rephrase the current query (it might implicitly refer to previous responses or questions)
        final_input = rephrase_chain.invoke({"chat_history": trimmed_conversation, "input": user_input})
    response = rag_chain.invoke(final_input)
    # Display assistant response in chat message container
    with st.chat_message('assistant', avatar=ASSISTANT_AVATAR):
        typewriter_effect(response, speed=SPEED, allow_html=True) # unsafe_allow is True because the model sometimes outputs tables, but maybe I should remove it
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": ASSISTANT_AVATAR})
