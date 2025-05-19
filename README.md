Crie o ambiente virtual:

`conda create -n simple-rag`

Ativação do ambiente virtual:
`conda activate simple-rag`

Instalação das libs:
`pip3 install -r requirements.txt`

Embedding dos dados:
`python3 src/scripts/create_db.py`

Geração de resposta:
`python3 src/scripts/generate_sentences.py`

# Dados
Crie uma pasta chamada 'data' na partição principal, onde o arquivo 'metadata.csv' deve estar.

## Simple-RAG
# Para gerar as senteças via API (requisição ao ollama), é necessário rodar o serviço de LLM .

`ollama run <llm-model-name>`

Entre os modelos que podem ser usados, por exemplo: `qwen2.5:0.5b`, `qwen2.5:1.5b`, `gemma3:27b`, `deepseek-r1:7b`.
Após isso, mude o nome do modelo a ser usado dentro do script 'generate_sentences_using_api.py' e rode-o: 

`python3 src/scripts/generate_sentences_using_api.py`

# Para descrição do conteúdo das imagens: 
Altere o caminho da pasta das images. Exemplo: "data_folder = f"./data/{dataset_name}" no mesmo script, onde 'dataset_name' é o nome da pasta do folder usado.

Além disso, adicione o(s) nome(s) dos modelos desejados (VLM). 
Exemplo: model_name = `qwen2.5vl:32b`, `gemma3:27b`, `llava:34b` ou `qwen2.5:72b` 

Após isso, e com o modelo LLM rodando no serviço do ollama, digite o comando:
`python3 src/scripts/generate_sentences_by_image_description_using_api.py`

