import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Modelo para embeddings
embedding_model = SentenceTransformer("Qwen/Qwen2.5-1.5B")

# Exemplo de documentos (pode carregar de um CSV, banco de dados, etc.)
documents = [
   "The patient is an 8-year-old female presenting with a lesion on the arm, but specific details such as the size and Fitzpatrick skin type are not available. There is no documented family medical history or personal medical history of skin cancer. Environmental factors including piped water, sewage systems, and pesticide exposure are not reported. The patient does not have any known lifestyle factors like smoking or alcohol consumption. Currently, there are no symptoms associated with the lesion, including itching, growth, pain, changes in appearance, bleeding, or elevation.",
   "The patient is a 55-year-old female presenting with a lesion on the neck measuring 6.0 x 5.0 mm. Her Fitzpatrick skin type is classified as 3.0, indicating medium skin tone. There is a family history of cancer, including in both her father and mother from Pomerania. The patient has a history of skin cancer and reports no pesticide exposure. Environmental factors include access to piped water and sewage systems. She does not have any lifestyle risk factors such as smoking or alcohol consumption. The lesion shows signs of growth, itching, bleeding, elevation, and changes in appearance, but there is no associated pain.",
   "The patient is a 77-year-old female presenting with an unspecified lesion on the face, but specific details such as the size and Fitzpatrick skin type are not available. There is no documented family medical history or personal history of skin cancer. Environmental factors including piped water, sewage systems, and pesticide exposure are not reported. The patient does not have any known lifestyle risk factors like smoking or alcohol consumption. Currently, she reports itching as a symptom associated with the lesion, but there are no signs of growth, pain, changes in appearance, bleeding, or elevation.",
   "The patient is a 75-year-old female with lesion ID 4061 located on the hand. She has no symptoms such as itching, growth, pain, changes, bleeding, or elevation. Her family medical history, lifestyle factors, and exposure to pesticides are not provided. There is no record of prior skin cancer or general cancer history. An image of the lesion is available under PAT_1989_4061_934.png. Summary: The patient is a 75-year-old female with a lesion on the hand, identified by Lesion ID 4061. There are no symptoms such as itching, growth, pain, changes in appearance, bleeding, or elevation. Her family medical history and lifestyle factors are not provided, and there is no record of prior skin cancer or general cancer history. An image of the lesion is available under PAT_1989_4061_934.png.",
   "The patient is a 79-year-old male with a lesion on his forearm, identified by Lesion ID 1302. The lesion measures 5.0 x 5.0 mm and has a Fitzpatrick Skin Type of 1.0. His family medical history shows that both his father and mother are from Pomerania. There is no piped water or sewage system in the environment, and he has not been exposed to pesticides. The patient has a history of skin cancer but no family history of cancer. He does not smoke but consumes alcohol. The lesion exhibits symptoms of itching, growth, bleeding, and elevation, but there are no changes in appearance noted. An image of the lesion is available under PAT_684_1302_934.png.",
   "The patient is a 53-year-old individual with an unidentified gender, who has a lesion on the chest, identified by Lesion ID 1882. The size and other details of the lesion are not available. An image of the lesion is available under PAT_1549_1882_934.png. The patient does not exhibit significant symptoms, with only itching noted. There is no reported growth, pain, changes in the lesion, or bleeding. Additionally, there is no information on family medical history, environmental factors, smoking status, or alcohol consumption.",
   "The patient is a 52-year-old female with a lesion on her face, identified by Lesion ID 1471. The lesion measures 15.0 x 10.0 mm and has a Fitzpatrick Skin Type of 3.0. Her family medical history indicates that her father is from Germany and her mother is from Italy. The patient has access to piped water and a sewage system, and there is no reported pesticide exposure. She does not have a history of skin cancer but has a family history of cancer. She does not smoke but consumes alcohol. The lesion exhibits several symptoms: it is growing, showing changes in appearance, bleeding, and being elevated. An image of the lesion is available under PAT_778_1471_934.png.",
   "The patient is a 74-year-old female with a lesion on her face, identified by Lesion ID 179. The lesion measures 15.0 x 10.0 mm and has a Fitzpatrick Skin Type of 1.0. Her family medical history indicates that both her father and mother are from Pomerania. There is no access to piped water or a sewage system, and she has been exposed to pesticides. She does not have a history of skin cancer or any family history of cancer. She does not smoke or consume alcohol. The lesion exhibits several symptoms: it is growing, causing pain, bleeding, and being elevated. An image of the lesion is available under PAT_117_179_934.png.",
   "The patient is a 74-year-old individual (PAT_2070) with Lesion ID 4430 located on the arm. This lesion shows no symptoms such as itching, growth, pain, changes, bleeding, or elevation. Additionally, there was no information provided about the patient's family medical history, lifestyle factors, and exposure to pesticides. There is also no record of prior skin cancer history. For comparison, another dataset entry for a 68-year-old individual (PAT_1995) with Lesion ID 4080 on the forearm shows symptoms including itching but no changes in the lesion or other symptoms. An image of PAT_1995's lesion is available under PAT_1995_4080_695.png.",
   "The patient is a 58-year-old female (PAT_705) with Lesion ID 4015 located on the forearm. This lesion has a size of 9.0 x 7.0 mm and shows signs of growth but no itching, pain, changes, bleeding, or elevation. The patient has a history of skin cancer and family cancer. She does not smoke but consumes alcohol. There is also evidence of pesticide exposure in her environmental factors. An image of the lesion can be found under PAT_705_4015"
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