from openai import OpenAI      # SDK oficial de OpenAI;  para hablar con Ollama
from src.research_tools import wikipedia_search_tool, pubmed_search_tool, save_results_to_csv
import json
import os                     # para leer variables de entorno

client = OpenAI(
    base_url=os.environ.get("OLLAMA_API_URL", "http://localhost:11434/v1"),   # usa la variable si existe; si no, localhost por defecto
    api_key="ollama",
)


tools_by_name = {
    "wikipedia_search_tool": wikipedia_search_tool,
    "pubmed_search_tool": pubmed_search_tool,
}


# Esto es lo que el modelo realmente entiende.
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

# =================================== SE DEFINEN LOS ROLES DE LOS AGENTES ==========================================================


def research_agent(request, model="qwen3:1.7b"):
    messages = [{"role": "user", "content": request}]   # la pregunta en lenguaje natural, recibida como parámetro. Sirve ahora para cualquier pregunta. 

    tool_defs = [wikipedia_tool_def, pubmed_tool_def]   # lista con las dos "fichas de catálogo" (la API espera una lista, no variables sueltas)

    response = client.chat.completions.create(
        model=model,          # el modelo recibido como parámetro
        messages=messages,
        tools=tool_defs,      # el modelo ve estas dos opciones y decide cuál (si acaso) conviene usar
    )

    message = response.choices[0].message   # el mensaje de respuesta, para revisar si pidió una herramienta
    
    if message.tool_calls:                                    # el modelo pidió usar una herramienta
        tool_call = message.tool_calls[0]                     # la primera herramienta pedida
        args = json.loads(tool_call.function.arguments)       # arguments llega como texto; json.loads lo convierte en diccionario real
        nombre_funcion = tool_call.function.name              # ej: "pubmed_search_tool"
        funcion_real = tools_by_name[nombre_funcion]          # busca la función real en el diccionario
        resultado_real = funcion_real(**args)                 # ejecuta la función con los argumentos que pidió el modelo
        fuente = nombre_funcion.replace("_search_tool", "")    # "wikipedia_search_tool" -> "wikipedia", "pubmed_search_tool" -> "pubmed"
        save_results_to_csv(resultado_real, source=fuente)
        return resultado_real                                 # devuelve el resultado de la herramienta
    else:                                                     # el modelo respondió con texto normal, sin pedir herramienta
        return message.content                                # devuelve esa respuesta de texto directamente
    

WRITER_SYSTEM_PROMPT = """
Eres un asistente que sintetiza literatura científica.
Recibes un tema de investigación y una lista de hallazgos (resultados de
búsquedas en Wikipedia y/o PubMed).
Tu trabajo es escribir un resumen breve y claro que sintetice esos hallazgos,
basándote ÚNICAMENTE en la información que te dan, sin inventar datos que no
estén ahí.
Si los hallazgos están vacíos o no tienen información útil, dilo explícitamente
en vez de inventar contenido.

IMPORTANTE sobre cifras numéricas: nunca modifiques, "corrijas" ni reemplaces
una cifra (porcentajes, sensibilidad, especificidad, etc.) por otro número
inventado, ni siquiera si un editor cuestiona su veracidad. Si una cifra que
diste es cuestionada, tienes dos opciones válidas: (1) cítala tal cual aparece
en los hallazgos originales, sin cambiarla, o (2) elimínala y dilo
explícitamente ("no se puede confirmar esta cifra con los hallazgos
disponibles"). Inventar un número distinto para "corregir" uno real es un
error grave, incluso si parece más razonable.
"""


def writer_agent(tema, hallazgos, feedback=None, model="qwen3:1.7b"):
    contenido = f"Tema: {tema}\nHallazgos: {hallazgos}"

    if feedback:   # si hay retroalimentación por el revisor, se hace una corrección, no la primera versión
        contenido += f"\n\nEsta versión fue revisada por un editor, que dio esta crítica:\n{feedback}\nCorrige el resumen teniendo en cuenta esta crítica."

    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": contenido},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content # el texto que escribió, como string

      

EDITOR_SYSTEM_PROMPT = """
Eres un editor crítico que revisa resúmenes de literatura científica.
Recibes un tema de investigación y un resumen ya escrito sobre ese tema.
Revisa el resumen y evalúa dos cosas:
1. Relevancia: ¿cada punto del resumen está realmente relacionado con el tema,
   o hay información que se coló sin ser relevante?
2. Fidelidad: ¿el resumen parece inventar datos, o se mantiene fiel a lo que
   razonablemente se esperaría de los hallazgos?

Responde ÚNICAMENTE con un JSON con esta forma exacta, sin texto adicional:
{"aprobada": true, "comentarios": "tu evaluación, y qué corregir si aplica"}

El campo "aprobada" debe ser el booleano true o false (sin comillas alrededor,
no como texto), nunca una palabra como "sí" o "no".
"""

#def editor_agent(tema, resumen, model="qwen3:1.7b"):
#    messages = [
#        {"role": "system", "content": EDITOR_SYSTEM_PROMPT},                         # el ROL de editor crítico
#        {"role": "user", "content": f"Tema: {tema}\nResumen a revisar: {resumen}"},  # el tema + lo que escribió writer_agent
#    ]
#
#    response = client.chat.completions.create(
#        model=model,
#        messages=messages,
#        # sin tools= aquí tampoco: el editor no busca nada, solo evalúa texto
#    )
#
#    return response.choices[0].message.content   # la evaluación completa, con la línea "aprobada: sí/no" incluida

#def editor_agent(tema, resumen, model="qwen3:1.7b"):
#    messages = [
#        {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
#        {"role": "user", "content": f"Tema: {tema}\nResumen a revisar: {resumen}"},
#    ]
#
#    response = client.chat.completions.create(
#        model=model,
#        messages=messages,
#    )
#
#    texto = response.choices[0].message.content   # debería ser un string con forma de JSON
#    revision = json.loads(texto)                   # lo convierte en diccionario real: {"aprobada": True/False, "comentarios": "..."}
#
#    return revision   # ahora devuelve un diccionario, no un texto plano

def editor_agent(tema, resumen, model="qwen3:1.7b"):
    messages = [
        {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Tema: {tema}\nResumen a revisar: {resumen}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    texto = response.choices[0].message.content

    try:
        revision = json.loads(texto)                 # intenta convertir a diccionario
    except json.JSONDecodeError:                      # si el JSON viene mal formado o cortado...
        revision = {"aprobada": False, "comentarios": texto}   # ...se trata como "no aprobado", usando el texto crudo como crítica

    return revision


# =================================== CICLO DE REFLEXIÓN (escribir, revisar, corregir) ==========================================================
#def investigar_y_escribir(tema, max_intentos=3):
#    hallazgos = research_agent(f"Busca papers científicos sobre {tema}.")
#    print("HALLAZGOS:", hallazgos)   # ver qué trajo research_agent
#
#    resumen = writer_agent(tema, hallazgos)
#
#    for intento in range(max_intentos):
#        revision = editor_agent(tema, resumen)
#        print(f"--- Intento {intento + 1} ---")        # nuevo: saber en qué vuelta vamos
#        print("REVISIÓN:", revision)                   # nuevo: ver la crítica completa
#
#        if "aprobada: sí" in revision:
#            print("¡Aprobado!")                          # nuevo
#            return resumen
#
#        resumen = writer_agent(tema, hallazgos, feedback=revision)
#
#    print("Se agotaron los intentos sin aprobación.")    # nuevo
#    return resumen

def investigar_y_escribir(tema, max_intentos=3):
    hallazgos = research_agent(f"Busca papers científicos sobre {tema}.")
    print("HALLAZGOS:", hallazgos)

    resumen = writer_agent(tema, hallazgos)

    for intento in range(max_intentos):
        revision = editor_agent(tema, resumen)                   
        print(f"--- Intento {intento + 1} ---")
        print("REVISIÓN:", revision)

        if revision["aprobada"]:                                   
            print("¡Aprobado!")
            return resumen

        resumen = writer_agent(tema, hallazgos, feedback=revision["comentarios"])   # se usa solo el texto de la crítica

    print("Se agotaron los intentos sin aprobación.")
    return resumen


# =================================== Planificador ==========================================================
#PLANNER_SYSTEM_PROMPT = """
#Eres un planificador de investigación.
#Recibes una petición amplia o general sobre un tema de investigación.
#Tu único trabajo es descomponerla en una lista de 2 a 4 sub-temas concretos
#y específicos, que se puedan investigar por separado.
#No investigues nada tú mismo, no expliques nada, no agregues texto adicional.
#
#Responde ÚNICAMENTE con una lista en formato JSON de strings, así:
#["sub-tema 1", "sub-tema 2", "sub-tema 3"]
#"""

PLANNER_SYSTEM_PROMPT = """
Eres un planificador de investigación.
Recibes una petición amplia o general sobre un tema de investigación.
Tu único trabajo es descomponerla en una lista de 2 a 4 sub-temas concretos
y específicos, que se puedan investigar por separado.

IMPORTANTE: identifica el tema central de la petición (por ejemplo, una
enfermedad, tecnología o fenómeno específico) y asegúrate de que CADA
sub-tema mencione explícitamente ese tema central, no solo el campo general.
Por ejemplo, si la petición es sobre "biosensores ópticos para diagnóstico de
malaria", cada sub-tema debe incluir la palabra "malaria" o un término
directamente relacionado (ej. "Plasmodium", "hemozoína"), y nunca ser un
sub-tema genérico como "mecanismos de reconocimiento molecular en biosensores
ópticos", que podría aplicarse a cualquier enfermedad y no solo a malaria.

No investigues nada tú mismo, no expliques nada, no agregues texto adicional.

Responde ÚNICAMENTE con una lista en formato JSON de strings, así:
["sub-tema 1 sobre el tema central", "sub-tema 2 sobre el tema central"]
"""

def planner_agent(peticion, model="qwen3:1.7b"):
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},   # el rol de planificador
        {"role": "user", "content": peticion},                   # la petición amplia del usuario
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # sin tools= aquí: el planificador no busca nada, solo descompone en texto
    )

    texto = response.choices[0].message.content   # debería ser un string con forma de lista JSON
    sub_temas = json.loads(texto)                  # convierte ese texto en una lista real de Python

    return sub_temas