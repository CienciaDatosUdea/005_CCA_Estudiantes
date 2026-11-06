import requests
import xml.etree.ElementTree as ET

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "AgenteDicroismoMalaria/1.0 (proyecto de curso, contacto: tu_correo@ejemplo.com)"}


def wikipedia_search_tool(query: str, sentences: int = 5):
    """
    Busca en Wikipedia y devuelve un resumen corto del artículo más relevante.
    """
    # 1) Buscar el título del artículo más relevante 
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }
    search_resp = requests.get(WIKIPEDIA_API_URL, params=search_params, headers=HEADERS)
    search_resp.raise_for_status()
    results = search_resp.json()["query"]["search"]

    if not results:
        return []

    title = results[0]["title"]

    # 2) Pedirle a Wikipedia el resumen (la introducción) de ese artículo específico
    summary_params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    summary_resp = requests.get(WIKIPEDIA_API_URL, params=summary_params, headers=HEADERS)
    summary_resp.raise_for_status()
    pages = summary_resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "")

    # recortar el resumen a las primeras oraciones
    short_summary = ". ".join(extract.split(". ")[:sentences]).strip()
    if short_summary and not short_summary.endswith("."):
        short_summary += "."

    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    return [{"title": title, "summary": short_summary, "url": url}]

# *************************************************************************************************************************


PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi" #base de los dos servicios que se usan de la API de NCBI
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def pubmed_search_tool(query: str, max_results: int = 3):                         
    """
    Busca en PubMed (NCBI) y devuelve título, autores, revista, año,
    resumen y enlace de los artículos más relevantes. Pensada para
    literatura biomédica/clínica (diagnóstico de malaria, biosensores...)
    que Wikipedia y arXiv no cubren bien.
    """
    # 1) Buscar los IDs (PMIDs) de los artículos que coinciden con la búsqueda
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    search_resp = requests.get(PUBMED_ESEARCH_URL, params=search_params, headers=HEADERS)
    search_resp.raise_for_status()
    pmids = search_resp.json()["esearchresult"]["idlist"]

    if not pmids:
        return []

    # 2) Traer los detalles completos (título, autores, abstract...) de esos IDs
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    fetch_resp = requests.get(PUBMED_EFETCH_URL, params=fetch_params, headers=HEADERS)
    fetch_resp.raise_for_status()

    root = ET.fromstring(fetch_resp.content)
    results = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = (article.findtext(".//ArticleTitle", default="") or "").strip()

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName", default="")
            first = author.findtext("ForeName", default="")
            name = f"{first} {last}".strip()
            if name:
                authors.append(name)

        journal = article.findtext(".//Journal/Title", default="")
        year = article.findtext(".//JournalIssue/PubDate/Year", default="")

        abstract_parts = [el.text or "" for el in article.findall(".//Abstract/AbstractText")]
        abstract = " ".join(abstract_parts).strip()

        results.append({
            "title": title,
            "authors": authors,
            "published": year,
            "journal": journal,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "summary": abstract,
        })

    return results


# *************************************************************************************************************************
# PARA GUARDAR LOS RESULTADOS DE LAS BÚSQUEDAS REALIZADAS POR CADA SITIO (Wikipedia, PubMed, arXiv...) EN UN ARCHIVO CSV, PARA PODER CONSULTARLO DESPUÉS SIN VOLVER A BUSCAR EN INTERNET.
import csv
import os


def save_results_to_csv(results, source, filename="referencias_malaria.csv"):
    """
    Guarda una lista de resultados (de cualquiera de tus *_search_tool)
    en un archivo CSV, agregando filas nuevas sin borrar lo que ya había
    y evitando duplicar el mismo artículo si ya estaba guardado (por URL).
    """
    columnas = ["fuente", "title", "authors", "published", "journal", "url", "summary"]

    # 1) Si el archivo ya existe, leer qué URLs ya tenemos guardadas
    archivo_existe = os.path.exists(filename)
    urls_existentes = set()
    if archivo_existe:
        with open(filename, "r", newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                urls_existentes.add(fila["url"])

    # 2) Abrir en modo "a" (append = agregar al final, sin borrar lo anterior)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)

        if not archivo_existe:
            escritor.writeheader()

        nuevos = 0
        for r in results:
            if "error" in r:
                continue  # no se guardan resultados con error
            if r.get("url") in urls_existentes:
                continue  

            escritor.writerow({
                "fuente": source,
                "title": r.get("title", ""),
                "authors": ", ".join(r.get("authors", [])) if r.get("authors") else "",
                "published": r.get("published", ""),
                "journal": r.get("journal", ""),
                "url": r.get("url", ""),
                "summary": r.get("summary", ""),
            })
            nuevos += 1

    print(f"Guardados {nuevos} artículos nuevos en '{filename}' (de {len(results)} encontrados).")