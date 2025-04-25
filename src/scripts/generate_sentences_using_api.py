import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import request_to_llm


def load_dataset(file_path: str):
    try:
        df = pd.read_csv(file_path)
        return df.columns.tolist(), df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None


def preprocess_patient_data(row, column_names):
    """
    Process a DataFrame row into a dict of non-empty values.
    """
    data = {}
    for col in column_names:
        val = row[col]
        if pd.isna(val) or val == "":
            continue
        data[col] = val
    return data


def to_inline(data: dict) -> str:
    """
    Convert patient data dict into a semicolon-separated inline string,
    omitting missing or false/N/A values.
    """
    parts = []
    # Basic demographics
    if data.get("patient_id"):
        parts.append(f"ID={data['patient_id']}")
    if data.get("age"):
        parts.append(f"Age={data['age']} years")
    if data.get("gender"):
        parts.append(f"Gender={str(data['gender']).capitalize()}")
    if data.get("fitspatrick"):
        parts.append(f"Fitzpatrick Skin Type={data['fitspatrick']}")
    # Lesion details
    if data.get("region"):
        parts.append(f"Lesion Location={data['region'].lower()}")
    if data.get("diameter_1") and data.get("diameter_2"):
        parts.append(f"Lesion Size={data['diameter_1']}×{data['diameter_2']} mm")
    # Symptoms
    symptoms_map = {
        "itch": "Itching",
        "grew": "Growth",
        "hurt": "Pain",
        "changed": "Changes in lesion",
        "bleed": "Bleeding",
        "elevation": "Elevation",
    }
    symptoms = [label for key, label in symptoms_map.items()
                if data.get(key) and str(data[key]).lower() not in ("false", "n/a", "0")]
    if symptoms:
        parts.append(f"Symptoms={', '.join(symptoms)}")
    # Lifestyle
    lifestyle_map = {"smoke": "Smoking", "drink": "Alcohol consumption"}
    lifestyle = [label for key, label in lifestyle_map.items()
                 if data.get(key) and str(data[key]).lower() not in ("false", "n/a", "0")]
    if lifestyle:
        parts.append(f"Lifestyle={', '.join(lifestyle)}")
    # Environmental factors
    env_map = {
        "has_piped_water": "Piped water",
        "has_sewage_system": "Sewage system",
        "pesticide": "Pesticide exposure",
    }
    env = [label for key, label in env_map.items()
           if data.get(key) and str(data[key]).lower() not in ("false", "n/a", "0")]
    if env:
        parts.append(f"Environmental Factors={', '.join(env)}")
    # Family and medical history
    family = []
    if data.get("background_father") and str(data["background_father"]).lower() not in ("false", "n/a"):
        family.append(f"Father: {data['background_father']}")
    if data.get("background_mother") and str(data["background_mother"]).lower() not in ("false", "n/a"):
        family.append(f"Mother: {data['background_mother']}")
    if data.get("cancer_history") and str(data["cancer_history"]).lower() not in ("false", "n/a"):
        family.append("Family cancer history")
    if family:
        parts.append(f"Family History={'; '.join(family)}")
    if data.get("skin_cancer_history") and str(data["skin_cancer_history"]).lower() not in ("false", "n/a"):
        parts.append("Medical History=Skin cancer history")

    return "; ".join(parts)


def generate_response(raw_data: str, model_name: str = "qwen2.5:14b") -> str:
    """
    Send the inline patient data to the LLM and request exactly one natural sentence.
    """
    prompt = f"""
        System:
        You are a clinical documentation assistant. Transform the following semicolon-separated data into exactly one fluent, professional English sentence suitable for a dermatologist’s note. Omit any missing or false values. Do not invent information.

        User:
        {raw_data}

        Output sentence:
        """
    response = request_to_llm.request_to_ollama(model_name=model_name, prompt_message=prompt)
    return response.strip()


def main():
    dataset_name = "PAD-UFES-20"
    model_name =  "deepseek-r1:1.5b" # "gemma3:27b" # "qwq" # "qwen2.5:14b" # "deepseek-r1:70b" # "qwen2.5:0.5b"
    data_folder = os.path.join("data", dataset_name)
    columns, df = load_dataset(os.path.join(data_folder, "metadata.csv"))
    output_file = os.path.join(data_folder, f"metadata_with_sentences_new-prompt-{model_name}.csv")
    if df is None:
        return

    results = []
    for _, row in df.iterrows():
        data = preprocess_patient_data(row, columns)
        raw = to_inline(data)
        if not raw:
            continue
        # COntinua a geração dos dados
        generated_sentence = generate_response(raw, model_name=model_name)
        # Filter text sentences in the sentence after "</think>\n"
        if "</think>" in generated_sentence:
            after_think = generated_sentence.split("</think>", 1)[1].strip()
            print("✅ Extracted text after </think>:\n")
        else:
            print("❌ No </think> found in text.")
            after_think = generated_sentence
        print(f"Generated Sentence after filtered: {after_think}\n")
                   
        results.append({
            "patient_id": data.get("patient_id"),
            "img_id": data.get("img_id"),
            "diagnostic": data.get("diagnostic"),
            "raw_data": raw,
            "sentence": after_think,
        })

        pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()