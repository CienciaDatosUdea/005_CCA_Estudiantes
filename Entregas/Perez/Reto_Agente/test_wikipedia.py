from src.research_tools import wikipedia_search_tool
import json

resultados = wikipedia_search_tool("malaria", sentences=3)
print(json.dumps(resultados, indent=2, ensure_ascii=False)) # devuelve un resumen de 3 oraciones del artículo más relevante sobre malaria, en formato JSON.