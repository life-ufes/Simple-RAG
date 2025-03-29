import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Modelo para embeddings
embedding_model = SentenceTransformer("Qwen/Qwen2.5-1.5B")

# Exemplo de documentos (pode carregar de um CSV, banco de dados, etc.)
documents = [
    "The patient is an 8-year-old individual presenting with a lesion on the arm. There are no reported symptoms such as itching, growth, pain, changes in the lesion, bleeding, or elevation. The patient does not have a known history of skin cancer or a family history of cancer. Information regarding lifestyle factors like smoking or alcohol consumption is not available. Environmental exposures and Fitzpatrick skin type classification are also not specified.",
    # "The patient is a 55-year-old female presenting with a lesion on the neck measuring 6.0 x 5.0 mm. The Fitzpatrick skin type is classified as 3.0, indicating medium skin tone. Environmental factors include access to piped water and sewage systems, with no reported pesticide exposure. The patient has a personal history of skin cancer and a family history of cancer. She does not smoke or consume alcohol. Symptoms associated with the lesion include itching, growth, changes in the lesion, bleeding, and elevation, but there is no reported pain.",
    # "The patient is a 77-year-old individual presenting with a lesion on the face. The size of the lesion and Fitzpatrick skin type classification are not specified. There is no available information regarding family medical history, environmental factors, or lifestyle habits such as smoking or alcohol consumption. Medical history does not indicate any known personal or familial cancer history. Symptoms associated with the lesion include itching; however, there are no reports of growth, pain, changes in the lesion, bleeding, or elevation.",
    # "The patient is a 75-year-old individual presenting with a lesion on the hand. The size of the lesion and Fitzpatrick skin type classification are not specified. There is no available information regarding family medical history, environmental factors, or lifestyle habits such as smoking or alcohol consumption. Medical history does not indicate any known personal or familial cancer history. Symptoms associated with the lesion include itching; however, there are no reports of growth, pain, changes in the lesion, bleeding, or elevation.",
    # "The patient is a 79-year-old male presenting with a lesion on the forearm measuring 5.0 x 5.0 mm. The Fitzpatrick skin type classification is 1.0, indicating very fair skin. Family medical history for both parents is listed as 'POMERANIA,' which does not provide relevant information regarding health conditions. Environmental factors indicate that the patient does not have access to piped water or a sewage system and has no reported pesticide exposure. Medical history reveals a personal history of skin cancer, but there is no known family history of cancer. Lifestyle factors include non-smoker status but acknowledges alcohol consumption. Symptoms associated with the lesion are itching, growth, bleeding, and elevation, with no reports of pain or changes in the lesion.",
    # "The patient is a 53-year-old individual presenting with a lesion on the chest. The size of the lesion and Fitzpatrick skin type classification are not specified. There is no available information regarding family medical history, environmental factors, or lifestyle habits such as smoking or alcohol consumption. Medical history does not indicate any known personal or familial cancer history. Symptoms associated with the lesion include itching and elevation; however, there are no reports of growth, pain, changes in the lesion, or bleeding.",
    # "The patient is a 74-year-old female presenting with a lesion on the face measuring 15.0 x 10.0 mm. The Fitzpatrick skin type classification is 1.0, indicating very fair skin. Family medical history for both parents is listed as 'POMERANIA,' which does not provide relevant information regarding health conditions. Environmental factors indicate that the patient does not have access to piped water or a sewage system but has reported pesticide exposure. Medical history shows no personal or family history of cancer. Lifestyle factors include non-smoker status and no alcohol consumption. Symptoms associated with the lesion are itching, growth, pain, bleeding, and elevation; however, there are no reports of changes in the lesion.",
    # "The patient is a 68-year-old individual presenting with a lesion on the forearm. The size and Fitzpatrick skin type classification are not specified. There is no available information regarding family medical history, environmental factors, or lifestyle habits such as smoking or alcohol consumption. Medical history does not indicate any known personal or familial cancer history. Symptoms associated with the lesion include itching and changes in the lesion; however, there are no reports of growth, pain, bleeding, or elevation.",
    # "The patient is a 58-year-old female presenting with a lesion on the forearm measuring 9.0 x 7.0 mm. The Fitzpatrick skin type classification is 1.0, indicating very fair skin. Family medical history indicates that both parents are from Germany, but no specific health conditions are noted. Environmental factors reveal that the patient has access to piped water and a sewage system, with reported pesticide exposure. Medical history shows a personal history of skin cancer and a family history of cancer. Lifestyle factors include non-smoker status and alcohol consumption. Symptoms associated with the lesion are growth; however, there are no reports of itching, pain, changes in the lesion, bleeding, or elevation.",
    # "The patient is a 45-year-old individual presenting with a lesion on the neck. The size and Fitzpatrick skin type classification are not specified. There is no available information regarding family medical history, environmental factors, or lifestyle habits such as smoking or alcohol consumption. Medical history does not indicate any known personal or familial cancer history. Symptoms associated with the lesion include itching and elevation; however, there are no reports of growth, pain, changes in the lesion, or bleeding."
]


# Gerar os embeddings para todos os documentos
document_embeddings = embedding_model.encode(documents)

# Converter os embeddings para um formato que o FAISS pode usar (array numpy)
document_embeddings = np.array(document_embeddings).astype(np.float32)

# Criar o índice FAISS
index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Usando o índice L2 (distância Euclidiana)
index.add(document_embeddings)  # Adicionando os embeddings ao índice

# Salvar o índice FAISS em um arquivo
faiss.write_index(index, "medical_index.faiss")

# Salvar os documentos em um arquivo pickle
with open("medical_documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Índice FAISS e documentos salvos com sucesso!")