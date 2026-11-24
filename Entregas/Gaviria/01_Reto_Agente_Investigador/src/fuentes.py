"""Fuentes que devuelven las herramientas, verificación de las URLs citadas en el informe
y exportación de esas fuentes a BibTeX.

Un modelo local pequeño inventa referencias con facilidad. El agente de investigación
agrega a su salida la lista de resultados que de verdad devolvieron las herramientas,
al final se señalan las URLs del informe que no aparecen en esa lista, y con los mismos
resultados se arma un archivo BibTeX sin pasar por el modelo.
"""

import html
import json
import re
import unicodedata
from datetime import date

_URL = re.compile(r"https?://[^\s<>\"'\)\]]+")
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:v\d+)?(?:\.pdf)?$")


def normalizar_url(url: str) -> str:
    """Forma comparable de una URL: sin esquema, sin www ni barra final; en arXiv, sin versión."""
    u = url.strip().rstrip(".,;:").lower()
    u = re.sub(r"^https?://(www\.)?", "", u).rstrip("/")
    u = u.replace("arxiv.org/pdf/", "arxiv.org/abs/").removesuffix(".pdf")
    return re.sub(r"(arxiv\.org/abs/.+?)v\d+$", r"\1", u)


def fuentes_de(mensajes) -> list:
    """Resultados que devolvieron las herramientas (título, URL, autores, fecha, categoría), sin repetir."""
    fuentes = {}
    for m in mensajes:
        if not (isinstance(m, dict) and m.get("role") == "tool"):
            continue
        try:
            resultados = json.loads(m["content"])
        except (KeyError, TypeError, ValueError):
            continue
        for r in resultados if isinstance(resultados, list) else []:
            if isinstance(r, dict) and r.get("url"):
                url = r["url"].replace("http://arxiv.org", "https://arxiv.org")
                fuentes.setdefault(normalizar_url(url), {
                    "url": url,
                    "title": r.get("title") or url,
                    "authors": r.get("authors") or [],
                    "published": r.get("published") or "",
                    "category": r.get("category") or "",
                })
    return list(fuentes.values())


def bloque_fuentes(fuentes: list) -> str:
    """Lista HTML con el título y la URL de cada fuente."""
    if not fuentes:
        return ""
    items = "".join(
        f"<li><a href='{html.escape(f['url'])}' target='_blank'>{html.escape(f['title'])}</a></li>"
        for f in fuentes
    )
    return (
        "\n\n<h2 style='font-size:1.5em; color:#2563eb;'>📚 Sources retrieved</h2>"
        f"<ul class='fuentes'>{items}</ul>"
    )


def urls_sin_respaldo(informe: str, fuentes: set) -> list:
    """URLs citadas en el informe que no aparecen entre las fuentes recuperadas (normalizadas)."""
    faltantes = {}
    for u in _URL.findall(informe):
        u = u.rstrip(".,;:")
        if normalizar_url(u) not in fuentes:
            faltantes.setdefault(normalizar_url(u), u)
    return list(faltantes.values())


# ----- BibTeX -----
_PALABRAS_VACIAS = {"a", "an", "the", "on", "of", "in", "for", "and", "to", "with", "from", "via"}
_SIN_DESCOMPOSICION = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "ß": "ss", "æ": "ae"})


def _texto_bib(s: str) -> str:
    """Una sola línea (los títulos de arXiv traen saltos) y con &, % y # escapados."""
    return re.sub(r"(?<!\\)([&%#])", r"\\\1", " ".join(s.split()))


def _ascii(s: str) -> str:
    """Minúsculas sin tildes ni signos, para las claves de las entradas."""
    s = unicodedata.normalize("NFKD", s.translate(_SIN_DESCOMPOSICION))
    return re.sub(r"[^a-z0-9]", "", s.encode("ascii", "ignore").decode().lower())


def bibtex(fuentes: list, hoy: str | None = None) -> str:
    """Entradas @misc para las fuentes recuperadas, con los datos de las herramientas y no del modelo."""
    hoy = hoy or date.today().isoformat()
    entradas, claves, vistas = [], set(), set()
    for f in fuentes:
        if normalizar_url(f["url"]) in vistas:
            continue
        vistas.add(normalizar_url(f["url"]))

        url, autores = f["url"], f.get("authors") or []
        sitio = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        anio = (f.get("published") or "")[:4] or hoy[:4]
        arxiv = _ARXIV_ID.search(url)

        # Clave al estilo de Google Scholar: apellido (o sitio), año y primera palabra del título.
        base = autores[0].split()[-1] if autores else ("wikipedia" if "wikipedia.org" in sitio else sitio.split(".")[0])
        palabra = next((w for w in re.findall(r"[^\W_]+", f["title"]) if w.lower() not in _PALABRAS_VACIAS), "")
        clave = _ascii(base) + anio + _ascii(palabra)
        sufijo = ""
        while clave + sufijo in claves:
            sufijo = chr(ord(sufijo or "a") + 1)
        clave += sufijo
        claves.add(clave)

        campos = [("title", _texto_bib(f["title"]))]
        if autores:
            campos.append(("author", " and ".join(_texto_bib(a) for a in autores)))
        campos.append(("year", anio))
        if arxiv:
            campos += [("eprint", arxiv.group(1)), ("archivePrefix", "arXiv")]
            if f.get("category"):
                campos.append(("primaryClass", f["category"]))
            url = f"https://arxiv.org/abs/{arxiv.group(1)}"
        else:
            fuente = "Wikipedia" if "wikipedia.org" in sitio else sitio
            # Sin autores, BibTeX necesita "key" para ordenar la entrada.
            campos += [("howpublished", fuente), ("note", f"Accessed {hoy}"), ("key", fuente)]
        campos.append(("url", url))

        cuerpo = ",\n".join(f"  {k:<13} = {{{v}}}" for k, v in campos)
        entradas.append(f"@misc{{{clave},\n{cuerpo}\n}}")
    return "\n\n".join(entradas) + "\n" if entradas else ""
