import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") 

client = chromadb.Client()
collection = client.get_or_create_collection(name="interview_knowledge")


def load_knowledge_base(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 75) -> list[str]:
    """
    Split text into overlapping chunks.

    chunk_size: number of words per chunk
    overlap: number of words overlap between chunks
    """

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def index_documents(chunks: list[str]) -> None:
    existing_count = collection.count()

    if existing_count > 0:
        return
    print(f"Indexing {len(chunks)} chunks into ChromaDB")  

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
        )


knowledge_text = load_knowledge_base("data/interview_knowledge.md")
knowledge_chunks = split_into_chunks(knowledge_text)
index_documents(knowledge_chunks)


def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    retrieved_docs = results["documents"][0]
    reranked_docs = rerank_chunks(query, retrieved_docs, top_k=top_k)

    print("===== RAG QUERY =====")
    print(query)

    print("===== RETRIEVED BEFORE RERANK =====")
    for doc in retrieved_docs:
        print(doc)

    print("===== RERANKED CONTEXT =====")
    for doc in reranked_docs:
        print(doc)

    return reranked_docs

def retrieve_from_text(text: str, query: str, top_k: int = 3) -> list[str]:
    """
    Build a temporary vector collection from input text,
    then retrieve the most relevant chunks for the query.
    """

    chunks = split_into_chunks(text)

    temp_client = chromadb.Client()
    temp_collection = temp_client.get_or_create_collection(name="temp_resume_collection")

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        temp_collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
        )

    query_embedding = embedding_model.encode(query).tolist()

    results = temp_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, len(chunks))
    )

    retrieved_docs = results["documents"][0]
    return rerank_chunks(query, retrieved_docs, top_k=top_k)
    

def rerank_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    if not chunks:
        return []

    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker_model.predict(pairs)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored_chunks[:top_k]]