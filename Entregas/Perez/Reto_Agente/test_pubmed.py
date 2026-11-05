from src.research_tools import pubmed_search_tool

resultados = pubmed_search_tool("malaria rapid diagnostic biosensor", max_results=3)

for r in resultados:
    print(r["title"], "-", r["published"])
    print("Autores:", ", ".join(r["authors"]))
    print(r["url"])
    print()