import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import request_to_llm, request_to_llm_image_description
import base64
from itertools import dropwhile
import re

def load_image_base64(image_path):
    try:
        with open(image_path, 'rb') as image_buffer:
            buffer = image_buffer.read() 
            img_base64 = base64.b64encode(buffer).decode("utf-8")
        return img_base64
    except Exception as e:
        raise ValueError(f"Erro ao tentar carregar o buffer da imagem! Error: {e}")
    
def load_dataset(file_path: str):
    try:
        df = pd.read_csv(file_path)
        return df.columns.tolist(), df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    
def generate_response_function(image_base64_content: str, model_name: str = "qwen2.5:14b") -> str:
    response = request_to_llm_image_description.request_to_ollama(model_name=model_name, image_base64_content=image_base64_content)
    return response.strip()


def clean_response_header(text):
    lines = text.splitlines()

    # Typical AI-generated header patterns
    header_patterns = [
        r'^\s*here is.*$',
        r'^\s*here\´s.*$',
        r'^\s*here\’s.*$',
        r"^\s*here's.*$",       
        r'^\s*below is.*$',      
        r'^\s*this is.*$',       
        r'^\s*okay.*$',   
    ]

    # Check only if the first line (line 0) has response headers
    first_line = lines[0].strip() if lines else ""
    is_header = any(re.match(pattern, first_line, flags=re.IGNORECASE) for pattern in header_patterns)

    if not is_header:
        # If the first line is not a header, return the original text
        return text

    # Take everything after the first line and remove any following blank lines
    lines_after_header = lines[1:]
    cleaned_lines = list(dropwhile(lambda x: x.strip() == '', lines_after_header))

    return "".join(cleaned_lines)

def main():
    dataset_name = "PAD-UFES-20"
    model_name = "gemma3:4b"
    data_folder = f"./data/{dataset_name}"
    
    # Assume que a função `load_dataset` agora só precisa do caminho do arquivo
    columns, dataset = load_dataset(os.path.join(data_folder, "metadata.csv"))
    
    # Cria o diretório de resultados se ele não existir
    output_dir = os.path.join(data_folder, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"metadata_with_sentences_of_image-description_{model_name}.csv")

    results = []
    processed_ids = set()

    # 1. Lógica de Sincronização: Carregar resultados existentes
    if os.path.exists(output_file):
        try:
            print(f"Arquivo de resultados encontrado. Carregando registros existentes...")
            df_existing = pd.read_csv(output_file)
            # Mantém o progresso adicionando os dicionários existentes à lista
            results = df_existing.to_dict('records')
            # Adiciona os IDs já processados ao set para verificação rápida
            for item in results:
                processed_ids.add(str(item['img_id']))
            print(f"✅ {len(processed_ids)} imagens já processadas. Continuando de onde parou.")
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível ler o arquivo de resultados existente. Começando do zero. Erro: {e}")
            results = []

    # 2. Loop principal sobre o dataset
    for idx, row in dataset.iterrows():
        img_id = row['img_id']

        # ✅ CRÍTICO: Pula a imagem se o ID já foi processado
        if img_id in processed_ids:
            continue

        print(f"\n🔄 Processando imagem {len(processed_ids) + 1}/{len(dataset)}: {img_id}")
        
        image_path = os.path.join(data_folder, "images", img_id)

        # As funções a seguir são assumidas como existentes e funcionais
        # --------------------------------------------------------------------
        img_buffer_content = load_image_base64(image_path)
        if not img_buffer_content:
            print(f"❌ Erro ao carregar a imagem: {image_path}")
            continue

        response = generate_response_function(image_base64_content=img_buffer_content, model_name=model_name)
        # --------------------------------------------------------------------

        if "</think>" in response:
            after_think = response.split("</think>", 1)[1].strip()
        else:
            after_think = response
        
        filtered_after_think = clean_response_header(after_think)
        print(f"💬 Descrição Gerada: {filtered_after_think}")
                   
        results.append({
            "patient_id": row.get("patient_id"),
            "img_id": row.get("img_id"),
            "diagnostic": row.get("diagnostic"),
            "sentence": filtered_after_think,
        })

        # Salva o progresso a cada imagem processada
        pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
        print(f"💾 Progresso salvo em {output_file}")

    print("\n🎉 Processo concluído com sucesso!")

if __name__ == "__main__":
    main()