from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama ignora este valor, pero el SDK de OpenAI exige que exista
)

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Responde con una sola palabra: ¿sabes responder sí?"}],
)

print(response.choices[0].message.content)