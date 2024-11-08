# app.py

import os
import logging
import tempfile

import streamlit as st

from utils import (
    initialize_langchain_graph,
    initialize_vector_index,
    initialize_cypher_chain,
    initialize_agent_tools,
    initialize_langchain_agent,
    process_files,
    get_all_file_paths,
    chunk_text,
    query_agent
)

# Configure logging for app.py (optional, since utils.py already configures logging)
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="DevOps Knowledge Graph RAG App", layout="wide")
    st.title("DevOps Knowledge Graph RAG Application")

    # Sidebar for file upload or directory selection
    st.sidebar.header("Upload Documents or Select Directory")
    upload_option = st.sidebar.selectbox("Choose an option", ["Upload Files", "Select Directory"])

    file_paths = []
    if upload_option == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdirname:
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(tmpdirname, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_path)
                st.sidebar.success(f"Uploaded {len(uploaded_files)} files.")
    else:
        directory = st.sidebar.text_input("Enter directory path")
        if directory:
            if os.path.isdir(directory):
                file_paths = get_all_file_paths(directory)
                st.sidebar.success(f"Found {len(file_paths)} files in the directory.")
            else:
                st.sidebar.error("Invalid directory path.")

    if file_paths:
        with st.spinner("Processing files..."):
            texts = process_files(file_paths)
            all_chunks = []
            for text in texts:
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
            st.write(f"**Total chunks created:** {len(all_chunks)}")

        # Initialize LangChain's Neo4jGraph
        graph = initialize_langchain_graph(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )

        # Initialize vector index and Cypher chain
        with st.spinner("Initializing vector index and Cypher chain..."):
            vector_index = initialize_vector_index(graph)
            cypher_chain = initialize_cypher_chain(graph)

        # Initialize agent with tools
        with st.spinner("Initializing LangChain agent..."):
            tools = initialize_agent_tools(vector_index, cypher_chain)
            agent = initialize_langchain_agent(tools)

        # User Interaction
        st.header("Ask a Question")
        user_question = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Generating answer..."):
                    answer = query_agent(agent, user_question)
                st.subheader("**Answer:**")
                st.write(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in the application: {e}")
        st.error("An unexpected error occurred. Please check the logs for more details.")
    finally:
        # Ensure that the Neo4j driver is properly closed upon termination
        from utils import driver
        driver.close()
        logger.info("Closed Neo4j driver connection.")
