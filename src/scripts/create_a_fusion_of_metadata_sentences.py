import pandas as pd
import os

def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Erro ao carregar os dados ({path}): {e}")

if __name__ == "__main__":
    dataset_name = "PAD-UFES-20"
    folder = f"./data/{dataset_name}"
    #llm_model_name="llm-deepseek-r1:70b"
    #vllm_model_name="llava:34b"
    for llm_model_name in ["qwen2.5:72b", "phi4", "deepseek-r1:70b" , "gemma3:27b", "qwq"]:
        for vllm_model_name in ["llava:34b","gemma3:27b","qwen2.5:72b"]:
            output_file = os.path.join(folder, f"metadata_with_sentences_of_patient_description_and_image-description_llm-{llm_model_name}_vllm-{vllm_model_name}.csv")

            # Nomes dos arquivos de entrada
            file_history = "metadata_with_sentences_new-prompt-deepseek-r1:70b.csv"
            file_description = "metadata_with_sentences_of_image-description_llava:34b.csv"

            # Carrega ambos os datasets
            df_history = load_dataset(os.path.join(folder, file_history))
            df_desc    = load_dataset(os.path.join(folder, file_description))

            # Certifique-se de que a coluna de join existe em ambos
            assert "img_id" in df_history.columns,  "Coluna 'img_id' não encontrada em histórico de pacientes"
            assert "img_id" in df_desc.columns,     "Coluna 'img_id' não encontrada em descrições de lesão"

            # Faz merge pelos img_id comuns
            df_merged = df_history.merge(
                df_desc[["img_id", "sentence"]],
                on="img_id",
                how="inner",
                suffixes=("_history", "_desc")
            )

            # Concatena as sentenças em uma nova coluna
            df_merged["sentence"] = (
                "Patient data:\n" + df_merged["sentence_history"].str.strip()
                + "\nSkin lesion description:\n" + df_merged["sentence_desc"].str.strip()
            )

            # Seleciona apenas as colunas finais de interesse
            df_final = df_merged[[
                "patient_id",
                "img_id",
                "diagnostic",
                "sentence"
            ]]

            # Salva o CSV resultante
            df_final.to_csv(output_file, index=False, encoding="utf-8")
            print(f"Arquivo salvo em: {output_file}")