import requests

def request_to_ollama(llm_api_server_host:str="localhost", llm_api_server_port:str=11434, prompt_message:str="You are an experienced dermatology research assistant. Given the following base64‑encoded image of a skin lesion, write one natural, flowing paragraph that describes only its visual features—covering overall shape and symmetry, border definition, main color and any variations, approximate size, surface texture, and any surrounding skin changes such as erythema or scaling. Do not offer a diagnosis or medical opinion. Just describe the image, In a natural description paragraph.", image_base64_content:str="", model_name=""):
    url = f"http://{llm_api_server_host}:{llm_api_server_port}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt_message,
        "images": [image_base64_content],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "[Sem resposta do modelo]")
    except Exception as e:
        return f"Erro ao consultar o modelo: {str(e)}"