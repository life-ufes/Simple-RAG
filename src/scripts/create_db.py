import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Modelo para embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Exemplo de documentos (pode carregar de um CSV, banco de dados, etc.)
documents = [
    "Paciente masculino, 45 anos, com histórico de melanoma.",
    "Paciente feminino, 60 anos, sem histórico familiar de câncer de pele.",
    "Lesão observada na perna, 3cm de diâmetro, crescimento irregular.",
    "Exposição prolongada ao sol é um fator de risco para melanoma.",
    "Caso a lesão esteja sangrando e e está coçando, a o risco de ser um melanoma"
]

# Gerando os embeddings
doc_embeddings = embedding_model.encode(documents)

# Criando índice FAISS
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Salvando o índice
faiss.write_index(index, "medical_index.faiss")

# Salvando os documentos separadamente
with open("medical_documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Indexação concluída!")
