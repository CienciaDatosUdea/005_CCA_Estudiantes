import os
import re
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Optional

import requests
import wikipedia
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tavily import TavilyClient
from urllib3.util.retry import Retry

load_dotenv()  # Loads environment variables from a .env file


# ----- Session with retries & headers -----
def _build_session(
    user_agent: str = "LF-ADP-Agent/1.0 (mailto:your.email@example.com)",
) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_redirect=False,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


session = _build_session()


# ----- Utilities -----
def ensure_pdf_url(abs_or_pdf_url: str) -> str:
    url = abs_or_pdf_url.strip().replace("http://", "https://")
    if "/pdf/" in url and url.endswith(".pdf"):
        return url
    url = url.replace("/abs/", "/pdf/")
    if not url.endswith(".pdf"):
        url += ".pdf"
    return url


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def clean_text(s: str) -> str:
    s = re.sub(r"-\n", "", s)  # "transfor-\nmers" -> "transformers"
    s = re.sub(r"\r\n|\r", "\n", s)  # normaliza saltos
    s = re.sub(r"[ \t]+", " ", s)  # colapsa espacios
    s = re.sub(r"\n{3,}", "\n\n", s)  # no más de 1 línea en blanco seguida
    return s.strip()


def fetch_pdf_bytes(pdf_url: str, timeout: int = 90) -> bytes:
    r = session.get(pdf_url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content


def pdf_bytes_to_text(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    # 1) PyMuPDF
    try:
        import fitz  # PyMuPDF

        out = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            n = len(doc)
            limit = n if max_pages is None else min(max_pages, n)
            for i in range(limit):
                out.append(doc.load_page(i).get_text("text"))
        return "\n".join(out)
    except Exception:
        pass

    # 2) pdfminer.six
    try:
        from pdfminer.high_level import extract_text_to_fp

        buf_in = BytesIO(pdf_bytes)
        buf_out = BytesIO()
        extract_text_to_fp(buf_in, buf_out)
        return buf_out.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")


def maybe_save_pdf(pdf_bytes: bytes, dest_dir: str, filename: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, _safe_filename(filename))
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path


# ----- arXiv search -----
# Palabras que no ayudan a encontrar artículos y que los modelos suelen poner en las consultas.
_PALABRAS_VACIAS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with", "using",
    "via", "from", "by", "about", "recent", "advances", "approach", "approaches",
    "method", "methods", "review", "survey", "overview", "new", "latest", "study",
}


def _consultar_arxiv(terminos: List[str], max_results: int, conjuncion: bool = True):
    """Entradas de arXiv para los términos dados, unidos con AND (o con OR si conjuncion=False)."""
    union = "+AND+" if conjuncion else "+OR+"
    search_query = union.join(f"all:{requests.utils.quote(t)}" for t in terminos)
    resp = session.get(
        "https://export.arxiv.org/api/query"
        f"?search_query={search_query}&start=0&max_results={max_results}",
        timeout=60,
    )
    resp.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    return ET.fromstring(resp.content).findall("atom:entry", ns), ns


def arxiv_search_tool(
    query: str,
    max_results: int = 3,
) -> List[Dict]:
    """
    Searches arXiv for scientific papers and returns title, authors, date, URL and abstract.

    Args:
        query (str): A few keywords, e.g. "autoregressive neural network Ising".
        max_results (int): Number of papers to return (default 3).
    """
    # ===== FLAGS INTERNOS =====
    # Con modelos locales pequeños, 6 páginas de PDF por resultado llenan el contexto
    # y el modelo pierde las instrucciones. Por defecto se usa solo el resumen;
    # ARXIV_PDF_TEXT=1 en el .env vuelve al comportamiento original.
    _INCLUDE_PDF = os.getenv("ARXIV_PDF_TEXT", "0") == "1"
    _EXTRACT_TEXT = _INCLUDE_PDF
    _MAX_PAGES = 6
    _TEXT_CHARS = 5000
    _SAVE_FULL_TEXT = False
    _SLEEP_SECONDS = 1.0
    # ==========================

    # La API de arXiv une las palabras sueltas con OR y devuelve resultados poco
    # relacionados. Se exige que aparezcan todos los términos y, si no hay
    # resultados, se van quitando los últimos (hasta tres) antes de volver a OR.
    terminos = [t for t in re.findall(r"[\w\-]+", query) if t.lower() not in _PALABRAS_VACIAS]
    terminos = terminos or query.split()

    out: List[Dict] = []
    try:
        entries, ns = [], {}
        for n in range(len(terminos), max(len(terminos) - 4, 0), -1):
            entries, ns = _consultar_arxiv(terminos[:n], max_results)
            if entries:
                break
            time.sleep(3)  # la API pide unos segundos entre consultas
        if not entries:
            entries, ns = _consultar_arxiv(terminos, max_results, conjuncion=False)
    except requests.exceptions.RequestException as e:
        return [{"error": f"arXiv API request failed: {e}"}]
    except ET.ParseError as e:
        return [{"error": f"arXiv API XML parse failed: {e}"}]

    try:
        for entry in entries:
            title = (
                entry.findtext("atom:title", default="", namespaces=ns) or ""
            ).strip()
            published = (
                entry.findtext("atom:published", default="", namespaces=ns) or ""
            )[:10]
            # Categoría principal (p. ej. cond-mat.stat-mech), para el primaryClass del BibTeX
            primaria = entry.find("{http://arxiv.org/schemas/atom}primary_category")
            category = primaria.get("term", "") if primaria is not None else ""
            url_abs = entry.findtext("atom:id", default="", namespaces=ns) or ""
            # original abstract
            abstract_summary = (
                entry.findtext("atom:summary", default="", namespaces=ns) or ""
            ).strip()

            authors = []
            for a in entry.findall("atom:author", ns):
                nm = a.findtext("atom:name", default="", namespaces=ns)
                if nm:
                    authors.append(nm)

            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break
            if not link_pdf and url_abs:
                link_pdf = ensure_pdf_url(url_abs)

            item = {
                "title": title,
                "authors": authors,
                "published": published,
                "category": category,
                "url": url_abs,
                "summary": abstract_summary,
                "link_pdf": link_pdf,
            }

            pdf_bytes = None
            if (_INCLUDE_PDF or _EXTRACT_TEXT) and link_pdf:
                try:
                    pdf_bytes = fetch_pdf_bytes(link_pdf, timeout=90)
                    time.sleep(_SLEEP_SECONDS)
                except Exception as e:
                    item["pdf_error"] = f"PDF fetch failed: {e}"

            if _EXTRACT_TEXT and pdf_bytes:
                try:
                    text = pdf_bytes_to_text(pdf_bytes, max_pages=_MAX_PAGES)
                    text = clean_text(text) if text else ""
                    if text:
                        if _SAVE_FULL_TEXT:
                            item["summary"] = text
                        else:
                            item["summary"] = text[:_TEXT_CHARS]
                except Exception as e:
                    item["text_error"] = f"Text extraction failed: {e}"

            out.append(item)
        return out
    except Exception as e:
        return [{"error": f"Unexpected error: {e}"}]


# ---- Tool def ----
arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches arXiv and returns title, authors, date, URL and abstract of each result.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords."},
                "max_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
}


## Tavily search tool


def tavily_search_tool(
    query: str, max_results: int = 5, include_images: bool = False
) -> list[dict]:
    """
    Perform a search using the Tavily API.

    Args:
        query (str): The search query.
        max_results (int): Number of results to return (default 5).
        include_images (bool): Whether to include image results.

    Returns:
        List[dict]: A list of dictionaries with keys like 'title', 'content', and 'url'.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # Si se lanza una excepción, aisuite aborta todo el paso de investigación.
        # Devolviendo el error, el modelo puede seguir con arXiv y Wikipedia.
        return [{"error": "TAVILY_API_KEY not set; use arxiv_search_tool or wikipedia_search_tool instead."}]

    client = TavilyClient(api_key, api_base_url=os.getenv("DLAI_TAVILY_BASE_URL"))

    try:
        response = client.search(
            query=query, max_results=max_results, include_images=include_images
        )

        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                }
            )

        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        return [{"error": str(e)}]  # For LLM-friendly agents


tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}

## Wikipedia search tool


def wikipedia_search_tool(query: str, sentences: int = 5) -> List[Dict]:
    """
    Searches Wikipedia for a summary of the given query.

    Args:
        query (str): Search query for Wikipedia.
        sentences (int): Number of sentences to include in the summary.

    Returns:
        List[Dict]: A list with a single dictionary containing title, summary, and URL.
    """
    try:
        page_title = wikipedia.search(query)[0]
        # Sin auto_suggest la librería no "corrige" títulos que ya existen
        # ("Ising model" pasaba a "isling model" y fallaba).
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0], auto_suggest=False)
        summary = wikipedia.summary(page.title, sentences=sentences, auto_suggest=False)

        return [{"title": page.title, "summary": summary, "url": page.url}]
    except Exception as e:
        return [{"error": str(e)}]


# Tool definition
wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article.",
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# Tool mapping
tool_mapping = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}
