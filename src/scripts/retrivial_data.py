def retrieve_relevant_docs(query, index, documents, top_k=2):
    """Busca os documentos mais relevantes para uma pergunta usando FAISS."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs
