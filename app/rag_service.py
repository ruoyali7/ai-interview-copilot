import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(name="interview_knowledge")


def load_knowledge_base(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chunks(text: str) -> list[str]:
    chunks = text.split("\n\n")
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def index_documents(chunks: list[str]) -> None:
    existing_count = collection.count()

    if existing_count > 0:
        return

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
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]

    print("===== RAG QUERY =====")
    print(query)
    print("===== RETRIEVED CONTEXT =====")
    for doc in retrieved_docs:
        print(doc)

    return retrieved_docs