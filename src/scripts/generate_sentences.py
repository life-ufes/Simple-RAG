import os
import re
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Uncomment and configure these if you want to enable retrieval-based context.
# embedding_model = SentenceTransformer("Qwen/Qwen2.5-1.5B", device="cuda")
# index = faiss.read_index("medical_index.faiss")
# with open("medical_documents.pkl", "rb") as f:
#     documents = pickle.load(f)

# Load the generation pipeline. Adjust parameters like repetition_penalty if supported.
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B",
    device="cuda",
)

def retrieve_relevant_docs(query, index, documents, top_k=3):
    """
    Retrieve top_k similar documents based on the query.
    Uncomment and use this function when your FAISS index is ready.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def generate_response(user_query):
    """
    Generates a new, concise sentence summarizing the patient's condition.
    The prompt is designed to force the model to create a completely new sentence,
    avoiding any verbatim repetition from the provided data.
    """
    # Optional: Retrieve similar cases to provide context (can be omitted if causing replication)
    # retrieved_docs = retrieve_relevant_docs(user_query, index, documents)
    # For now, we provide a single sample context as a reference.
    # retrieved_docs = [
    #     "The patient is an 8-year-old female with a lesion on her arm and no significant symptomatic details."
    # ]

    # retrieved_docs = [
    #     "The patient is an 8-year-old female presenting with a lesion on the arm, but specific details such as the size and Fitzpatrick skin type are not available. There is no documented family medical history or personal medical history of skin cancer. Environmental factors including piped water, sewage systems, and pesticide exposure are not reported. The patient does not have any known lifestyle factors like smoking or alcohol consumption. Currently, there are no symptoms associated with the lesion, including itching, growth, pain, changes in appearance, bleeding, or elevation.",
    #     "The patient is a 55-year-old female presenting with a lesion on the neck measuring 6.0 x 5.0 mm. Her Fitzpatrick skin type is classified as 3.0, indicating medium skin tone. There is a family history of cancer, including in both her father and mother from Pomerania. The patient has a history of skin cancer and reports no pesticide exposure. Environmental factors include access to piped water and sewage systems. She does not have any lifestyle risk factors such as smoking or alcohol consumption. The lesion shows signs of growth, itching, bleeding, elevation, and changes in appearance, but there is no associated pain.",
    #     "The patient is a 77-year-old female presenting with an unspecified lesion on the face, but specific details such as the size and Fitzpatrick skin type are not available. There is no documented family medical history or personal history of skin cancer. Environmental factors including piped water, sewage systems, and pesticide exposure are not reported. The patient does not have any known lifestyle risk factors like smoking or alcohol consumption. Currently, she reports itching as a symptom associated with the lesion, but there are no signs of growth, pain, changes in appearance, bleeding, or elevation.",
    #     "The patient is a 75-year-old female with lesion ID 4061 located on the hand. She has no symptoms such as itching, growth, pain, changes, bleeding, or elevation. Her family medical history, lifestyle factors, and exposure to pesticides are not provided. There is no record of prior skin cancer or general cancer history. An image of the lesion is available under PAT_1989_4061_934.png. Summary: The patient is a 75-year-old female with a lesion on the hand, identified by Lesion ID 4061. There are no symptoms such as itching, growth, pain, changes in appearance, bleeding, or elevation. Her family medical history and lifestyle factors are not provided, and there is no record of prior skin cancer or general cancer history. An image of the lesion is available under PAT_1989_4061_934.png.",
    #     "The patient is a 79-year-old male with a lesion on his forearm, identified by Lesion ID 1302. The lesion measures 5.0 x 5.0 mm and has a Fitzpatrick Skin Type of 1.0. His family medical history shows that both his father and mother are from Pomerania. There is no piped water or sewage system in the environment, and he has not been exposed to pesticides. The patient has a history of skin cancer but no family history of cancer. He does not smoke but consumes alcohol. The lesion exhibits symptoms of itching, growth, bleeding, and elevation, but there are no changes in appearance noted. An image of the lesion is available under PAT_684_1302_934.png.",
    #     "The patient is a 53-year-old individual with an unidentified gender, who has a lesion on the chest, identified by Lesion ID 1882. The size and other details of the lesion are not available. An image of the lesion is available under PAT_1549_1882_934.png. The patient does not exhibit significant symptoms, with only itching noted. There is no reported growth, pain, changes in the lesion, or bleeding. Additionally, there is no information on family medical history, environmental factors, smoking status, or alcohol consumption.",
    #     "The patient is a 52-year-old female with a lesion on her face, identified by Lesion ID 1471. The lesion measures 15.0 x 10.0 mm and has a Fitzpatrick Skin Type of 3.0. Her family medical history indicates that her father is from Germany and her mother is from Italy. The patient has access to piped water and a sewage system, and there is no reported pesticide exposure. She does not have a history of skin cancer but has a family history of cancer. She does not smoke but consumes alcohol. The lesion exhibits several symptoms: it is growing, showing changes in appearance, bleeding, and being elevated. An image of the lesion is available under PAT_778_1471_934.png.",
    #     "The patient is a 74-year-old female with a lesion on her face, identified by Lesion ID 179. The lesion measures 15.0 x 10.0 mm and has a Fitzpatrick Skin Type of 1.0. Her family medical history indicates that both her father and mother are from Pomerania. There is no access to piped water or a sewage system, and she has been exposed to pesticides. She does not have a history of skin cancer or any family history of cancer. She does not smoke or consume alcohol. The lesion exhibits several symptoms: it is growing, causing pain, bleeding, and being elevated. An image of the lesion is available under PAT_117_179_934.png.",
    #     "The patient is a 74-year-old individual (PAT_2070) with Lesion ID 4430 located on the arm. This lesion shows no symptoms such as itching, growth, pain, changes, bleeding, or elevation. Additionally, there was no information provided about the patient's family medical history, lifestyle factors, and exposure to pesticides. There is also no record of prior skin cancer history. For comparison, another dataset entry for a 68-year-old individual (PAT_1995) with Lesion ID 4080 on the forearm shows symptoms including itching but no changes in the lesion or other symptoms. An image of PAT_1995's lesion is available under PAT_1995_4080_695.png.",
    #     "The patient is a 58-year-old female (PAT_705) with Lesion ID 4015 located on the forearm. This lesion has a size of 9.0 x 7.0 mm and shows signs of growth but no itching, pain, changes, bleeding, or elevation. The patient has a history of skin cancer and family cancer. She does not smoke but consumes alcohol. There is also evidence of pesticide exposure in her environmental factors. An image of the lesion can be found under PAT_705_4015"
    # # ]   
    # ]
    # # Combine additional context. You may remove this if it triggers too much copying.
    # context = "\n".join(retrieved_docs)
    
    # # Build an explicit prompt with clear instructions and boundaries.
    # prompt = (
    #     "### Instruction:\n"
    #     "You are a medical expert. Based on the patient data below, generate a completely new, concise sentence summarizing "
    #     "the patient's current condition. DO NOT repeat any phrases or details verbatim from the input. Synthesize the information in your own words.\n\n"
    #     "### Patient Data:\n"
    #     f"{user_query}\n\n"
    #     "### Additional Context (for reference only, do not copy):\n"
    #     f"{context}\n\n"
    #     "### New Sentence:"
    # )

    prompt = (
    "## Instruction:\n"
    "You are a medical assistant tasked with generating a concise, natural-language anamnesis summary based on structured patient data. Use only the information explicitly provided — **do not infer or fabricate** missing data. Omit any field that is marked as 'N/A', 'False', or is otherwise not specified."
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
    "Example:\n"
    "Patient [PATIENT_ID] is a [PATIENT_AGE]-year-old [PATIENT_GENDER] presenting with a lesion on the [LESION_LOCATION] measuring [LESION_SIZE]. The patient's Fitzpatrick skin type is [FITZPATRICK_VALUE]. Family history includes [FAMILY_HISTORY]; environmental context includes [ENVIRONMENTAL_FACTORS]. Medical background includes [MEDICAL_HISTORY], with lifestyle factors such as [LIFESTYLE]. Reported symptoms include [SYMPTOMS]."
    "Important Notes:\n"
    "- Exclude any field where the value is missing, 'N/A', or marked 'False'.\n"
    "- If gender is unknown, use 'patient' instead of 'male/female'.\n"
    "- Maintain natural language and avoid rigid template wording.\n"
    "- Do not compare this patient to others."
    "## Patient data:\n"
    f"{user_query}\n\n"
    "New sentence:\n"
    )
    # Generate new sentence with adjusted parameters.
    response = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.8,
        num_return_sequences=1,
        do_sample=True
        #repetition_penalty=2.0  # Adjust or remove if not supported by your model.
    )

    # Post-process: Remove the prompt part and extract only the new sentence.
    generated_text = response[0]['generated_text'][len(prompt):].strip()

    # Optionally, remove any accidental repetition of patient data.
    # This splits on periods or newlines and selects the first complete sentence.
    sentences = re.split(r'\.\s+|\n', generated_text)
    new_sentence = sentences[0].strip() if sentences else generated_text
    if new_sentence and not new_sentence.endswith('.'):
        new_sentence += '.'
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
    prompt = f'''
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
    file_folder_path = "./data"
    number_of_samples = 1
    model_name = "deepseek:70b"

    # Load original dataset.
    columns_names, file_content = load_dataset(os.path.join(file_folder_path, "metadata.csv"))
    if file_content is not None:
        processed_data = []

        for _, row in file_content.iterrows():
            data = preprocess_patient_data(row, columns_names)
            # Create a multi-line prompt from patient data.
            prompt_template = mounting_prompt(data)
            
            # Clean the prompt to a single string if necessary.
            query_single_line = re.sub(r'\s+', ' ', prompt_template.strip())
            generated_sentence = generate_response(query_single_line)
            
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