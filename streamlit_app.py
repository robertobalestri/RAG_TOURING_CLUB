# CHANGE IF LOCAL OR CLOUD
ON_STREAMLIT_CLOUD = True

if ON_STREAMLIT_CLOUD:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.cohere import CohereEmbedding
from dotenv import load_dotenv

if not ON_STREAMLIT_CLOUD:
    load_dotenv()

def main():
    # Password protection
    def check_password():
        st.session_state["password_correct"] = False
        if ON_STREAMLIT_CLOUD:
            correct_password = st.secrets["APP_PASSWORD"]
        else:
            correct_password = os.getenv("APP_PASSWORD")
        
        if st.session_state.get("password") == correct_password:
            st.session_state["password_correct"] = True
        else:
            st.warning("Incorrect password")

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=check_password, key="password")
        return

    st.title("Il tuo itinerario in Campania")

    # Configure Azure OpenAI
    llm = AzureOpenAI(
        engine=st.secrets["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"] if ON_STREAMLIT_CLOUD else os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME"),
        model="gpt-4o",
        temperature=0.0,
        azure_endpoint=st.secrets["AZURE_OPENAI_API_ENDPOINT"] if ON_STREAMLIT_CLOUD else os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=st.secrets["AZURE_OPENAI_API_KEY"] if ON_STREAMLIT_CLOUD else os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=st.secrets["AZURE_OPENAI_API_VERSION"] if ON_STREAMLIT_CLOUD else os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    cohere_api_key = st.secrets["COHERE_API_KEY"] if ON_STREAMLIT_CLOUD else os.getenv("COHERE_API_KEY")

    # Configure the embedding model
    embed_model = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-multilingual-v3.0",
        input_type="search_query",
    )

    # Set configuration parameters
    Settings.llm = llm
    Settings.embed_model = embed_model

    if not ON_STREAMLIT_CLOUD:
        # Load documents
        parser = PDFReader()
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            "./content/Documents", file_extractor=file_extractor
        ).load_data()

    # Create ChromaDB client and collection
    db = chromadb.PersistentClient(path="./chroma_db")

    # Load or create the vector store index
    try:
        chroma_collection = db.get_collection("quickstart")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
    except Exception as e:
        print(f"Failed to load index from storage: {e}")
        
        st.write("Qualcosa è andato storto :(")

        if not ON_STREAMLIT_CLOUD:
            chroma_collection = db.get_or_create_collection("quickstart")
            st.write("Chroma collection: ", str(chroma_collection))

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Create the index from documents if it doesn't exist
            vector_store_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
            st.write(f"Index created and stored in: ./chroma_db")

    try:
        chat_engine = vector_store_index.as_chat_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=10,
            chat_mode="condense_plus_context",
            verbose=True
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.form(key="itinerary_form"):
            zona = st.text_input("Inserisci la zona:", key="zona")
            giorni = st.text_input("Inserisci il numero di giorni:", key="giorni")
            submit_button = st.form_submit_button(label="Cerca Itinerario")

        if submit_button:
            with st.spinner("Caricamento in corso..."):
                if zona and giorni:
                    query = f"Consigliami un itinerario in {zona} per {giorni} giorni."
                    response = chat_engine.chat(query)
                    st.session_state.chat_history.append(("user", query))
                    st.session_state.chat_history.append(("machine", response.response))
                    st.experimental_rerun()

        st.markdown(
            """
            <style>
            .user-message {
                background-color: #D6EAF8;
                color: black;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .machine-message {
                background-color: #FADBD8;
                color: black;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .markdown h1, .markdown h2, .markdown h3, .markdown h4, .markdown h5, .markdown h6 {
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.chat_history:
            st.write("Cronologia chat:")
            for sender, message in st.session_state.chat_history:
                if sender == "user":
                    st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="machine-message">{message}</div>', unsafe_allow_html=True)

            with st.form(key="chat_form"):
                user_input = st.text_input("Continuare a chattare:", key="chat_input")
                chat_submit_button = st.form_submit_button(label="Invia")

                if chat_submit_button and user_input:
                    with st.spinner("Caricamento in corso..."):
                        response = chat_engine.chat(user_input)
                        st.session_state.chat_history.append(("user", user_input))
                        st.session_state.chat_history.append(("machine", response.response))
                        st.experimental_rerun()

    except KeyError as e:
        st.write(f"KeyError: {e} - The specified was not found in the graph indices.")
    except Exception as e:
        st.write(f"An error occurred while creating the chat engine: {e}")

if __name__ == "__main__":
    main()
