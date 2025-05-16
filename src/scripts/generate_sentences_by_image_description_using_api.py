import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import request_to_llm, request_to_llm_image_description
import base64

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


def main():
    dataset_name = "PAD-UFES-20"
    model_name = "qwen2.5vl:32b" # "gemma3:27b" # "llava:34b" # "deepseek-r1:70b" # "qwen2.5:72b" # "phi4" # "deepseek-r1:70b" # "gemma3:27b" # "qwq" # "qwen2.5:14b" # "deepseek-r1:70b" # "qwen2.5:0.5b"
    data_folder = f"./data/{dataset_name}"
    columns, dataset = load_dataset(os.path.join(data_folder, "metadata.csv"))
    output_file = os.path.join(data_folder, f"metadata_with_sentences_of_image-description_{model_name}.csv")
    
    # Add coluna com a descrição da imagem
    results = []
    for idx, row in dataset.iterrows():
        img_id = row['img_id']
        image_path = os.path.join(data_folder, "images")
        image_folder_path = os.path.join(image_path, img_id)

        img_buffer_content = load_image_base64(image_folder_path)
        generate_response = generate_response_function(image_base64_content=img_buffer_content, model_name=model_name)
        # Filter text sentences in the sentence after "</think>\n"
        if "</think>" in generate_response:
            after_think = generate_response.split("</think>", 1)[1].strip()
            print("✅ Extracted text after </think>:\n")
        else:
            print("❌ No </think> found in text.")
            after_think = generate_response
        print(f"Generated Sentence after filtered: {after_think}\n")
                   
        results.append({
            "patient_id": row.get("patient_id"),
            "img_id": row.get("img_id"),
            "diagnostic": row.get("diagnostic"),
            "sentence": after_think,
        })

        pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()