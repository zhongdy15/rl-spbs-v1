from openai import OpenAI
import configparser


def llm_chat(api_key: str, model: str, url: str, messages: list) -> str:
    try:
        client = OpenAI(api_key=api_key, base_url=url)
        response = client.chat.completions.create(
            model=model,
            messages=[*messages],
            stream=False,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        message_content = response.choices[0].message.content
        return message_content
    except Exception as e:
        print(f"Error occurred during LLM API call: {str(e)}")
        return ""

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/config.ini', encoding='utf-8')

    llm_api_key = config['llm_api']['api_key']
    llm_base_url = config['llm_api']['url']
    llm_model = config['llm_api']['model']

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "1+1等于多少"}
    ]

    response = llm_chat(llm_api_key, llm_model, llm_base_url, messages)
    print(response)
