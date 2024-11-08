# utils.py

import os
import logging
import tempfile
from typing import List, Tuple

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document
from neo4j import GraphDatabase

# Import LangChain components
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, GraphCypherQAChain
from langchain.agents import initialize_agent, Tool, AgentType

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------
# Neo4j Configuration
# ---------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Update as per your Neo4j instance
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")                 # Update as per your Neo4j credentials
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")      # Update as per your Neo4j credentials

def initialize_neo4j_driver(uri: str, user: str, password: str):
    """
    Initialize the Neo4j driver.

    Args:
        uri (str): Neo4j URI.
        user (str): Neo4j username.
        password (str): Neo4j password.

    Returns:
        GraphDatabase.driver: Initialized Neo4j driver.

    Example:
        >>> driver = initialize_neo4j_driver("bolt://localhost:7687", "neo4j", "password")
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise e

# Initialize Neo4j driver
driver = initialize_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Initialize LangChain's Neo4jGraph
def initialize_langchain_graph(uri: str, user: str, password: str) -> Neo4jGraph:
    """
    Initialize LangChain's Neo4jGraph.

    Args:
        uri (str): Neo4j URI.
        user (str): Neo4j username.
        password (str): Neo4j password.

    Returns:
        Neo4jGraph: Initialized Neo4jGraph object.

    Example:
        >>> graph = initialize_langchain_graph("bolt://localhost:7687", "neo4j", "password")
    """
    try:
        graph = Neo4jGraph(
            url=uri,
            user=user,
            password=password
        )
        logger.info("LangChain's Neo4jGraph initialized.")
        return graph
    except Exception as e:
        logger.error(f"Failed to initialize LangChain's Neo4jGraph: {e}")
        raise e

graph = initialize_langchain_graph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# ---------------------------
# Hugging Face Models Setup
# ---------------------------
def initialize_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Initialize the SentenceTransformer for embeddings.

    Args:
        model_name (str, optional): Model name. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer: Initialized embedding model.

    Example:
        >>> embedding_model = initialize_embedding_model()
    """
    try:
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"Initialized SentenceTransformer with model: {model_name}")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformer: {e}")
        raise e

embedding_model = initialize_embedding_model()

def initialize_text_generator(model_name: str = 'gpt2') -> pipeline:
    """
    Initialize the Hugging Face text generation pipeline.

    Args:
        model_name (str, optional): Model name. Defaults to 'gpt2'.

    Returns:
        pipeline: Initialized text generation pipeline.

    Example:
        >>> generator = initialize_text_generator()
    """
    try:
        generator = pipeline('text-generation', model=model_name)
        logger.info(f"Initialized text generation pipeline with model: {model_name}")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize text generation pipeline: {e}")
        raise e

generator = initialize_text_generator()

# ---------------------------
# Helper Functions
# ---------------------------
def load_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content.

    Example:
        >>> load_text_from_pdf("sample.pdf")
        "This is the extracted text from the PDF."
    """
    logger.info(f"Loading PDF file from {file_path}")
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
    return text

def load_text_from_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text content.

    Example:
        >>> load_text_from_docx("sample.docx")
        "This is the extracted text from the DOCX file."
    """
    logger.info(f"Loading DOCX file from {file_path}")
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}")
    return text

def load_text_from_txt(file_path: str) -> str:
    """
    Extract text from a TXT file.

    Args:
        file_path (str): Path to the TXT file.

    Returns:
        str: Extracted text content.

    Example:
        >>> load_text_from_txt("sample.txt")
        "This is the text content from the TXT file."
    """
    logger.info(f"Loading TXT file from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading TXT file {file_path}: {e}")
        text = ""
    return text

def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """
    Split text into chunks of specified maximum length.

    Args:
        text (str): The text to be chunked.
        max_length (int, optional): Maximum number of characters per chunk. Defaults to 500.

    Returns:
        List[str]: A list of text chunks.

    Example:
        >>> chunk_text("This is a long text...", max_length=10)
        ["This is a ", "long text..."]
    """
    logger.info("Starting text chunking")
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def process_files(file_paths: List[str]) -> List[str]:
    """
    Process a list of file paths and extract text content.

    Args:
        file_paths (List[str]): List of file paths.

    Returns:
        List[str]: List of extracted text contents.

    Example:
        >>> process_files(["sample.pdf", "sample.txt"])
        ["Text from PDF", "Text from TXT"]
    """
    logger.info("Processing files to extract text")
    texts = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            text = load_text_from_pdf(file_path)
        elif ext == '.docx':
            text = load_text_from_docx(file_path)
        elif ext == '.txt':
            text = load_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            continue
        if text:
            texts.append(text)
    logger.info(f"Extracted text from {len(texts)} files")
    return texts

def get_all_file_paths(directory: str) -> List[str]:
    """
    Retrieve all file paths within a directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        List[str]: List of file paths.

    Example:
        >>> get_all_file_paths("/path/to/dir")
        ["/path/to/dir/file1.pdf", "/path/to/dir/file2.txt"]
    """
    logger.info(f"Retrieving all file paths from directory: {directory}")
    supported_extensions = ['.pdf', '.docx', '.txt']
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                file_paths.append(os.path.join(root, file))
    logger.info(f"Found {len(file_paths)} supported files")
    return file_paths

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using a specified model.

    Args:
        texts (List[str]): List of text strings.

    Returns:
        List[List[float]]: List of embedding vectors.

    Example:
        >>> generate_embeddings(["Hello world"])
        [[0.1, 0.2, ..., 0.3]]
    """
    logger.info("Generating embeddings for texts")
    embeddings = embedding_model.encode(texts)
    logger.info("Embeddings generation completed")
    return embeddings.tolist()

def retrieve_similar_chunks(query: str, embeddings: List[List[float]], texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Retrieve top_k most similar text chunks to the query based on cosine similarity.

    Args:
        query (str): The query string.
        embeddings (List[List[float]]): List of embedding vectors.
        texts (List[str]): List of text chunks.
        top_k (int, optional): Number of top similar chunks to retrieve. Defaults to 5.

    Returns:
        List[Tuple[str, float]]: List of tuples containing text chunks and their similarity scores.

    Example:
        >>> retrieve_similar_chunks("Hello", embeddings, ["Hello world", "Goodbye"])
        [("Hello world", 0.95)]
    """
    logger.info("Generating embedding for the query")
    query_embedding = embedding_model.encode([query])

    logger.info("Calculating cosine similarities")
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    similar_chunks = [(texts[idx], similarities[idx]) for idx in top_indices]
    logger.info(f"Retrieved top {top_k} similar chunks")
    return similar_chunks

def initialize_vector_index(graph: Neo4jGraph) -> Neo4jVector:
    """
    Initialize the Neo4j Vector Index using Hugging Face embeddings.

    Args:
        graph (Neo4jGraph): LangChain's Neo4jGraph instance.

    Returns:
        Neo4jVector: Initialized vector index.

    Example:
        >>> vector_index = initialize_vector_index(graph)
    """
    logger.info("Initializing Neo4j Vector Index")
    # Initialize HuggingFaceEmbeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vector_index = Neo4jVector.from_existing_graph(
        embedding=hf_embeddings,
        url=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name='tasks',
        node_label="Task",
        text_node_properties=['name', 'description', 'status'],
        embedding_node_property='embedding',
    )
    logger.info("Neo4j Vector Index initialized")
    return vector_index

def initialize_cypher_chain(graph: Neo4jGraph) -> GraphCypherQAChain:
    """
    Initialize the GraphCypherQAChain using Hugging Face models.

    Args:
        graph (Neo4jGraph): LangChain's Neo4jGraph instance.

    Returns:
        GraphCypherQAChain: Initialized Cypher QA chain.

    Example:
        >>> cypher_chain = initialize_cypher_chain(graph)
    """
    logger.info("Initializing GraphCypherQAChain")
    cypher_llm = pipeline('text2text-generation', model='gpt2')  # Replace with a more suitable model if available
    qa_llm = pipeline('text-generation', model='gpt2')            # Replace with a more suitable model if available

    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        graph=graph,
        verbose=True,
    )
    logger.info("GraphCypherQAChain initialized")
    return cypher_chain

def initialize_agent_tools(vector_index: Neo4jVector, cypher_chain: GraphCypherQAChain) -> List[Tool]:
    """
    Initialize the tools for the LangChain agent.

    Args:
        vector_index (Neo4jVector): Vector index for unstructured data.
        cypher_chain (GraphCypherQAChain): Cypher chain for structured queries.

    Returns:
        List[Tool]: List of initialized tools.

    Example:
        >>> tools = initialize_agent_tools(vector_index, cypher_chain)
    """
    logger.info("Initializing agent tools")
    tools = [
        Tool(
            name="Tasks",
            func=vector_index.similarity_search,
            description="""Use this tool when you need to answer questions about task descriptions.
            Input should be a string containing your question.
            """
        ),
        Tool(
            name="Graph",
            func=cypher_chain.run,
            description="""Use this tool when you need to answer questions about microservices,
            their dependencies, teams, or perform aggregations like counting tasks.
            Input should be a string containing your question.
            """
        ),
    ]
    logger.info("Agent tools initialized")
    return tools

def initialize_langchain_agent(tools: List[Tool]) -> any:
    """
    Initialize the LangChain agent with the provided tools.

    Args:
        tools (List[Tool]): List of tools for the agent.

    Returns:
        any: Initialized agent.

    Example:
        >>> agent = initialize_langchain_agent(tools)
    """
    logger.info("Initializing LangChain agent")
    agent = initialize_agent(
        tools=tools,
        llm=generator,  # Using Hugging Face generator
        agent=AgentType.OPENAI_FUNCTIONS,  # Customize based on the agent type compatible with Hugging Face
        verbose=True
    )
    logger.info("LangChain agent initialized")
    return agent

def query_agent(agent, question: str) -> str:
    """
    Query the LangChain agent with a user question.

    Args:
        agent (any): Initialized LangChain agent.
        question (str): User's question.

    Returns:
        str: Agent's response.

    Example:
        >>> response = query_agent(agent, "How many open tickets are there?")
    """
    logger.info(f"Agent received question: {question}")
    try:
        response = agent.run(question)
        logger.info("Agent generated a response.")
    except Exception as e:
        logger.error(f"Error during agent query: {e}")
        response = "I'm sorry, I couldn't process your request."
    return response
