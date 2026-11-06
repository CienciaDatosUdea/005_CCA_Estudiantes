from openai import OpenAI      # SDK oficial de OpenAI; lo reciclamos para hablar con Ollama, que expone un endpoint compatible
import json                    # para convertir el string de argumentos que genera el modelo en un diccionario real de Python
from src.research_tools import wikipedia_search_tool, pubmed_search_tool # la función REAL que ejecuta la búsqueda

tools_by_name = {
    "wikipedia_search_tool": wikipedia_search_tool,
    "pubmed_search_tool": pubmed_search_tool,
}

client = OpenAI(
    base_url="http://localhost:11434/v1",  # apunta al servidor de Ollama en la máquina, no a la nube de OpenAI
    api_key="ollama",                       # Ollama ignora este valor, pero el SDK exige que exista algo
)

# Esto es lo que el modelo realmente entiende. No el código en Python.
# Es el "menú" que describe la herramienta disponible: nombre, qué hace, y qué
# parámetros espera (query obligatorio, sentences opcional con default=5). El
# modelo lee esta descripción como texto, junto con la pregunta, para decidir
# si le conviene pedirla o no.
wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",  # debe coincidir EXACTO con el nombre real de la función Python
        "description": "Busca en Wikipedia y devuelve un resumen corto del artículo más relevante para la consulta.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Palabras clave de búsqueda para Wikipedia.",
                },
                "sentences": {
                    "type": "integer",
                    "description": "Número de oraciones que debe tener el resumen.",
                    "default": 5,
                },
            },
            "required": ["query"],  # sentences es opcional (tiene default); query no
        },
    },
}

pubmed_tool_def = {
    "type": "function",
    "function": {
        "name": "pubmed_search_tool",
        "description": "Busca artículos científicos en PubMed (NCBI) sobre literatura biomédica o clínica — diagnóstico, biosensores, estudios médicos — y devuelve título, autores, revista, año y resumen. Úsala en vez de Wikipedia cuando se pida literatura científica o papers, no una definición general.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Palabras clave de búsqueda para PubMed.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Número máximo de artículos a devolver.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

response = client.chat.completions.create(
    #model="llama3.2:1b",   # probado antes: pidió la herramienta, pero con argumentos mal formados (no confiable)
    model="qwen3:1.7b",     # probado antes: pidió la herramienta con argumentos limpios y bien formados
    messages=[{"role": "user", "content": "Busca papers científicos recientes sobre biosensores para diagnóstico de malaria."}], # ESTA ES MI PETICIÓN COMO USUARIO. ESTO ES LO QUE YO PUEDO CAMBIAR. ASI, el modelo lee MI pregunta, y la compara contra las descripciones disponibles del MENÚ, para decidir cuál herramienta conviene usar — sin que tú le digas directamente cuál usar.
    tools=[wikipedia_tool_def, pubmed_tool_def],  # Se le muestra al modelo el menú de arriba; sin esto, no sabría que esta herramienta existe
)

message = response.choices[0].message
print("¿Pidió usar una herramienta?:", bool(message.tool_calls))  # tool_calls viene vacío/None si el modelo respondió con texto normal en vez de pedir la herramienta
print(message)  # objeto completo: content, role, tool_calls, y (solo en Qwen3) reasoning

# --- A partir de aquí cerramos el ciclo: se toma lo que el modelo PIDIÓ y se ejecuta de verdad ---

#Esto asume que el modelo sí pidió una herramienta. Si algún día tool_calls
# viniera vacío (pregunta que no necesita Wikipedia), esta línea fallaría con


tool_call = message.tool_calls[0]
args = json.loads(tool_call.function.arguments)

nombre_funcion = tool_call.function.name      # ej: "pubmed_search_tool"
funcion_real = tools_by_name[nombre_funcion]  # busca la función de verdad en el diccionario
resultado_real = funcion_real(**args)         # la ejecuta con los argumentos que pidió el modelo

print("Resultado real de la herramienta:")
print(resultado_real)