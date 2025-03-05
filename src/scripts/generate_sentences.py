import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
# import torch
import os
import re

# Carregar modelos uma vez
embedding_model = SentenceTransformer("Qwen/Qwen2.5-0.5B")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", device="cpu")

# Carregar índice FAISS
index = faiss.read_index("medical_index.faiss")

# Carregar documentos salvos
with open("medical_documents.pkl", "rb") as f:
    documents = pickle.load(f)

def retrieve_relevant_docs(query, index, documents, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# def generate_response(user_query):
#     """
#     Recupera contexto relevante e gera uma resposta.
#     """
#     retrieved_docs = retrieve_relevant_docs(user_query, index, documents)
    
#     # Construindo prompt formatado corretamente
#     context = "\n".join(retrieved_docs)
#     prompt = f"""
#     You are a health assistant. Analyze the clinical patient data below and generate a concise new sentence summarizing the patient's current condition. Do not simply repeat the input. 

#     Patient Data:
#     {user_query}

#     Additional Context from similar cases:
#     {context}

#     New Sentence:
#     """

#     response = generator(prompt, max_new_tokens=256, temperature=0.4, do_sample=True)
#     return response[0]['generated_text'].replace(prompt, "").strip()

def generate_response(user_query):
    """
    Retrieve relevant context and generate a new clinical sentence.
    """
    retrieved_docs = retrieve_relevant_docs(user_query, index, documents)
    
    context = "\n".join(retrieved_docs)
    prompt = f"""
        You are a health assistant. Analyze the clinical patient data below and generate a completely new and concise sentence summarizing the patient's current condition. Avoid repeating any part of the provided data verbatim.

        Patient Data:
        {user_query}

        Additional Context from similar cases:
        {context}

        New Sentence:
        """    
    response = generator(prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.7, do_sample=True)
    generated_text = response[0]['generated_text'].replace(prompt, "").strip()
    return generated_text


def preprocess_patient_data(row, column_names):
    """
    Processa os dados do paciente e retorna um dicionário com os valores filtrados.
    """
    filtered_data = {col: str(row[col]) for col in column_names if pd.notna(row[col]) and row[col] != ""}
    return filtered_data
  
def load_dataset(file_path: str):
    try:
        df = pd.read_csv(file_path)
        return df.columns.tolist(), df
    except Exception as e:
        print(f"Erro ao ler o arquivo. Erro: {e}")
        return None, None

def mounting_prompt(data):
    try:
        # Monta o prompt utilizando os valores do dicionário
        prompt = f'''
            - Patient ID: {data.get('patient_id', 'N/A')}
            - Lesion ID: {data.get('lesion_id', 'N/A')}
            - Age: {data.get('age', 'N/A')} years old
            - Gender: {data.get('gender', 'N/A')}
            - Lesion Location: {data.get('region', 'N/A')}
            - Lesion Size: {data.get('diameter_1', 'N/A')} x {data.get('diameter_2', 'N/A')} mm
            - Fitspatrick: {data.get('fitspatrick', 'N/A')}            
            - Family Medical History:
                - Father: {data.get('background_father', 'N/A')}
                - Mother: {data.get('background_mother', 'N/A')}
            - Environmental Factors:
                - Has Piped Water: {data.get('has_piped_water', 'N/A')}
                - Has Sewage System: {data.get('has_sewage_system', 'N/A')}
                - Pesticide Exposure: {data.get('pesticide', 'N/A')}
            - Medical History:
                - Skin Cancer History: {data.get('skin_cancer_history', 'N/A')}
                - Family Cancer History: {data.get('cancer_history', 'N/A')}
            - Lifestyle:
                - Smoker: {data.get('smoke', 'N/A')}
                - Alcohol Consumption: {data.get('drink', 'N/A')}
            - Symptoms:
                - Itching: {data.get('itch', 'N/A')}
                - Growth: {data.get('grew', 'N/A')}
                - Pain: {data.get('hurt', 'N/A')}
                - Changes in Lesion: {data.get('changed', 'N/A')}
                - Bleeding: {data.get('bleed', 'N/A')}
                - Elevation: {data.get('elevation', 'N/A')}'''
        return prompt
    except Exception as e:
        print(f"Erro ao processar os dados. Erro:{e}\n")
        return None
  
def write_dataset_with_sentences(file_folder_path, dataframe):
    """Salva o dataframe contendo os prompts processados."""
    file_path = os.path.join(file_folder_path, "metadata_with_sentences.csv")
    dataframe.to_csv(file_path, index=False, encoding="utf-8", quotechar='"', sep=",")

if __name__ == "__main__":
    file_folder_path = "/home/wytcor/PROJECTs/SIMPLE-RAG/data"
    
    # Carregar dataset original
    columns_names, file_content = load_dataset(os.path.join(file_folder_path, "metadata.csv"))

    if file_content is not None:
        processed_data = []  # Lista para armazenar os dados antes de criar o dataframe

        for _, row in file_content.iterrows():
            data = preprocess_patient_data(row, columns_names)
            query = mounting_prompt(data)

            patient_id = data.get("patient_id")
            img_id = data.get("img_id")

            query_single_line = re.sub(r'\s+', ' ', query.strip())

            # Adiciona os dados processados à lista
            processed_data.append({
                "patient_id": patient_id,
                "img_id": img_id,
                "sentence": query
            })

        # Criar DataFrame final e salvar
        dataframe = pd.DataFrame(processed_data)
        write_dataset_with_sentences(file_folder_path, dataframe)
        
        print(f"Arquivo salvo em: {file_folder_path}/metadata_with_sentences.csv")
