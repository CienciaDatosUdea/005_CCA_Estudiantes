from src.research_tools import wikipedia_search_tool, pubmed_search_tool, save_results_to_csv

wiki = wikipedia_search_tool("malaria", sentences=3)
save_results_to_csv(wiki, source="wikipedia")

pubmed = pubmed_search_tool("malaria rapid diagnostic biosensor", max_results=2)
save_results_to_csv(pubmed, source="pubmed")