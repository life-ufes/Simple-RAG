import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Carregar modelo de embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Carregar índice FAISS
index = faiss.read_index("medical_index.faiss")

# Carregar documentos salvos
with open("medical_documents.pkl", "rb") as f:
    documents = pickle.load(f)

def retrieve_relevant_docs(query, index, documents, top_k=2):
    """Busca os documentos mais relevantes para uma pergunta usando FAISS."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def generate_response(user_query):
    """
    Recupera contexto relevante e gera uma resposta.
    """
    retrieved_docs = retrieve_relevant_docs(user_query, index, documents)

    # Construindo prompt formatado corretamente
    context = "\n".join(retrieved_docs)
    prompt = f"""
    Você é um assistente médico especializado em dermatologia.
    Baseado nas informações abaixo, gere uma resposta médica concisa:
    
    Contexto:
    {context}

    Pergunta: {user_query}
    Resposta:
    """
    
    # Carregando modelo de geração de texto
    generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", device="cpu")
    
    response = generator(prompt, max_new_tokens=150, temperature=0.5)
    return response[0]['generated_text']

if __name__=="__main__":
    # Testando o RAG
    query = "Quais são os fatores de risco para melanoma?"
    response=generate_response(query)
    print(response)
