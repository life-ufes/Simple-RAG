import os
import re
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import request_to_llm  

def generate_response(user_query, model_name="qwen2.5:14b"):
    """
    Generates a new, concise sentence summarizing the patient's condition.
    The prompt is designed to force the model to create a completely new sentence,
    avoiding any verbatim repetition from the provided data.
    """
    prompt = (
    "## Instruction:\n"
    "You are a health assistant. Based on the provided patient data, write a natural, concise summary of the patient's anamnesis (in English). "
    "Your output must incorporate the following placeholders exactly as given:\n"
    "  - [PATIENT_ID]: the current patient's identifier.\n"
    "  - [PATIENT_AGE]: the patient's age.\n"
    "  - [PATIENT_GENDER]: the patient's gender.\n"
    "  - [LESION_LOCATION]: the location of the lesion.\n"
    "  - [LESION_SIZE]: the lesion size in mm (formatted as 'width x height').\n"
    "  - [FITZPATRICK_VALUE]: the Fitzpatrick skin type.\n"
    "  - [FAMILY_HISTORY]: family medical history details.\n"
    "  - [ENVIRONMENTAL_FACTORS]: environmental factors such as piped water, sewage system, and pesticide exposure.\n"
    "  - [MEDICAL_HISTORY]: the patient's medical history.\n"
    "  - [LIFESTYLE]: lifestyle details such as smoking and alcohol consumption.\n"
    "  - [SYMPTOMS]: reported symptoms (e.g., itching, pain, changes in lesion, bleeding, elevation).\n\n"
    "Write a natural sentence summarizing the patient's condition using the above placeholders where appropriate. "
    "Do not include any comparisons with other patients.\n\n"
    "Example of desired output:\n"
    "    'Patient [PATIENT_ID] is an [PATIENT_AGE]-year-old [PATIENT_GENDER] presenting with a lesion on the [LESION_LOCATION] "
    "measuring [LESION_SIZE]. The patient's Fitzpatrick skin type is [FITZPATRICK_VALUE]. Their family medical history is [FAMILY_HISTORY], "
    "environmental factors include [ENVIRONMENTAL_FACTORS], and their medical history is [MEDICAL_HISTORY]. They have a history of "
    "[LIFESTYLE] habits and report symptoms such as [SYMPTOMS].'\n\n"
    "Make a natural sentence full-filling the targets, if there is not a specific information, don't cite it!!!"
    "## Patient data:\n"
    f"{user_query}\n\n"
    "New sentence:\n"
    )

    ## Faz a requisição dos dados
    new_sentence = request_to_llm.request_to_ollama(model_name=model_name, prompt_message=prompt)

    return new_sentence
def preprocess_patient_data(row, column_names):
    """
    Process patient row data and return a dictionary of non-empty values.
    """
    return {col: str(row[col]) for col in column_names if pd.notna(row[col]) and row[col] != ""}

def load_dataset(file_path: str):
    try:
        df = pd.read_csv(file_path)
        return df.columns.tolist(), df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def mounting_prompt(data):
    """
    Create a human-readable prompt from patient data.
    """
    prompt = f'''\n
- Patient ID: {data.get('patient_id', 'N/A')}
- Lesion ID: {data.get('lesion_id', 'N/A')}
- Age: {data.get('age', 'N/A')} years old
- Gender: {data.get('gender', 'N/A')}
- Lesion Location: {data.get('region', 'N/A')}
- Lesion Size: {data.get('diameter_1', 'N/A')} x {data.get('diameter_2', 'N/A')} mm
- Fitzpatrick: {data.get('fitspatrick', 'N/A')}
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
    - Elevation: {data.get('elevation', 'N/A')}
    '''.strip()
    return prompt

def write_dataset_with_sentences(file_folder_path, dataframe, model_name, number_of_samples):
    """
    Save the dataframe with generated sentences to a CSV file.
    """
    file_path = os.path.join(file_folder_path, f"metadata_with_sentences_{model_name}_{number_of_samples}.csv")
    dataframe.to_csv(file_path, index=False, encoding="utf-8", quotechar='"', sep=",")
    print(f"File saved at: {file_path}")

if __name__ == "__main__":
    dataset_name="PAD-UFES-20"
    file_folder_path = f"./data/{dataset_name}"
    number_of_samples = 1
    model_name = "qwen2.5:14b" # "deepseek:70b"

    # Load original dataset.
    columns_names, file_content = load_dataset(os.path.join(file_folder_path, "metadata.csv"))
    if file_content is not None:
        processed_data = []

        for _, row in file_content.iterrows():
            data = preprocess_patient_data(row, columns_names)
            # Create a multi-line prompt from patient data.
            prompt_template = mounting_prompt(data)
            
            # # Clean the prompt to a single string if necessary.
            query_single_line = re.sub(r'\s+', ' ', prompt_template.strip())
            generated_sentence = generate_response(query_single_line, model_name=model_name)
            
            print(f"Generated Sentence: {generated_sentence}\n")
            
            processed_data.append({
                "patient_id": data.get("patient_id"),
                "img_id": data.get("img_id"),
                "diagnostic": data.get("diagnostic"),
                "template": prompt_template,
                "sentence": generated_sentence
            })

            # Create final DataFrame and save to CSV.
            dataframe = pd.DataFrame(processed_data)
            write_dataset_with_sentences(file_folder_path, dataframe, model_name, number_of_samples)