import os
import json
import re
from typing import List
from datetime import datetime
from aisuite import Client
from src.agents import (
    research_agent,
    writer_agent,
    editor_agent,
    DEFAULT_MODEL,
    OLLAMA_API_URL,
    OLLAMA_TIMEOUT,
)

client = Client(
    provider_configs={
        "openai": {
            "api_key": "ollama",
            "base_url": OLLAMA_API_URL.rstrip("/") + "/v1",
            "timeout": OLLAMA_TIMEOUT,
        }
    }
)


def clean_json_block(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip("` \n")


from typing import List
import json, ast


def planner_agent(topic: str, model: str = DEFAULT_MODEL) -> List[str]:
    # Sin clave de Tavily el primer paso obligatorio no se puede cumplir, así que
    # en ese caso la búsqueda inicial se hace con Wikipedia y arXiv.
    if os.getenv("TAVILY_API_KEY"):
        required_first = "Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
    else:
        required_first = "Research agent: Use Wikipedia and arXiv to collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
    required_second = "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."
    final_required = "Writer agent: Generate the final comprehensive Markdown report with inline citations and a complete References section with clickable links."

    prompt = f"""
You are a planning agent responsible for organizing a research workflow using multiple intelligent agents.

🧠 Available agents:
- Research agent: MUST begin with a broad **web search using Tavily** to identify only **relevant** and **authoritative** items (e.g., high-impact venues, seminal works, surveys, or recent comprehensive sources). The output of this step MUST capture for each candidate: title, authors, year, venue/source, URL, and (if available) DOI.
- Research agent: AFTER the Tavily step, perform a **targeted arXiv search** ONLY for the candidates discovered in the web step (match by title/author/DOI). If an arXiv preprint/version exists, record its arXiv URL and version info. Do NOT run a generic arXiv search detached from the Tavily results.
- Writer agent: drafts based on research findings.
- Editor agent: reviews, reflects on, and improves drafts.

🎯 Produce a clear step-by-step research plan **as a valid Python list of strings** (no markdown, no explanations). 
Each step must be atomic, actionable, and assigned to one of the agents.
Maximum of 7 steps.

🚫 DO NOT include steps like “create CSV”, “set up repo”, “install packages”.
✅ Focus on meaningful research tasks (search, extract, rank, draft, revise).
✅ The FIRST step MUST be exactly:
"{required_first}"
✅ The SECOND step MUST be exactly:
"{required_second}"

🔚 The FINAL step MUST instruct the writer agent to generate a comprehensive Markdown report that:
- Uses all findings and outputs from previous steps
- Includes inline citations (e.g., [1], (Wikipedia/arXiv))
- Includes a References section with clickable links for all citations
- Preserves earlier sources
- Is detailed and self-contained

Topic: "{topic}"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )

    raw = response.choices[0].message.content.strip()

    # --- robust parsing: JSON -> ast -> fallback ---
    def _coerce_to_list(s: str) -> List[str]:
        # try strict JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except json.JSONDecodeError:
            pass
        # try Python literal list
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except Exception:
            pass
        # try to extract code fence if present
        if s.startswith("```") and s.endswith("```"):
            inner = s.strip("`")
            try:
                obj = ast.literal_eval(inner)
                if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                    return obj[:7]
            except Exception:
                pass
        return []

    steps = _coerce_to_list(raw)

    # enforce ordering & minimal contract
    def _ensure_contract(steps_list: List[str]) -> List[str]:
        if not steps_list:
            return [
                required_first,
                required_second,
                "Research agent: Synthesize and rank findings by relevance, recency, and authority; deduplicate by title/DOI.",
                "Writer agent: Draft a structured outline based on the ranked evidence.",
                "Editor agent: Review for coherence, coverage, and citation completeness; request fixes.",
                final_required,
            ]
        # inject/replace first two if missing or out of order
        steps_list = [s for s in steps_list if isinstance(s, str)]
        if not steps_list or steps_list[0] != required_first:
            steps_list = [required_first] + steps_list
        if len(steps_list) < 2 or steps_list[1] != required_second:
            # remove any generic arxiv step that is not tied to Tavily results
            steps_list = (
                [steps_list[0]]
                + [required_second]
                + [
                    s
                    for s in steps_list[1:]
                    if "arXiv" not in s or "For each collected item" in s
                ]
            )
        # cap to 7, ending always with the final report: antes el recorte a 7
        # podía eliminar este paso y el "informe" era la salida de otro agente.
        return [s for s in steps_list if s != final_required][:6] + [final_required]

    steps = _ensure_contract(steps)

    return steps


def executor_agent_step(step_title: str, history: list, prompt: str):
    """
    Executes a step of the executor agent.
    Returns:
        - step_title (str)
        - agent_name (str)
        - output (str)
        - fuentes (list): resultados de las herramientas; vacía si no es un paso de investigación
    """

    # Construir contexto enriquecido estructurado
    context = f"📘 User Prompt:\n{prompt}\n\n📜 History so far:\n"
    # Toda la investigación entra al contexto, pero de los borradores y revisiones
    # solo el último: los anteriores ya están incorporados y saturan el contexto
    # de un modelo local (Ollama usa 4096 tokens por defecto).
    ultimo_paso = {agent: i for i, (_, agent, _) in enumerate(history)}
    for i, (desc, agent, output) in enumerate(history):
        if agent in ("writer_agent", "editor_agent") and i != ultimo_paso[agent]:
            continue
        if "draft" in desc.lower() or agent == "writer_agent":
            context += f"\n✍️ Draft (Step {i + 1}):\n{output.strip()}\n"
        elif "feedback" in desc.lower() or agent == "editor_agent":
            context += f"\n🧠 Feedback (Step {i + 1}):\n{output.strip()}\n"
        elif "research" in desc.lower() or agent == "research_agent":
            context += f"\n🔍 Research (Step {i + 1}):\n{output.strip()}\n"
        else:
            context += f"\n🧩 Other (Step {i + 1}) by {agent}:\n{output.strip()}\n"

    enriched_task = f"""{context}

🧩 Your next task:
{step_title}
"""

    # Seleccionar agente basado en el paso. Primero por el prefijo ("Writer agent: ..."):
    # buscar la palabra en todo el título mandaba pasos como "Writer agent: draft
    # a report from the research findings" al agente de investigación.
    prefijo = step_title.split(":", 1)[0].lower()
    nombra_agente = any(k in prefijo for k in ("research", "writer", "editor"))
    step_lower = prefijo if nombra_agente else step_title.lower()
    if "research" in step_lower:
        content, fuentes = research_agent(prompt=enriched_task)
        print("🔍 Research Agent Output:", content)
        return step_title, "research_agent", content, fuentes
    elif "draft" in step_lower or "write" in step_lower:
        content, _ = writer_agent(prompt=enriched_task)
        return step_title, "writer_agent", content, []
    elif "revise" in step_lower or "edit" in step_lower or "feedback" in step_lower:
        content, _ = editor_agent(prompt=enriched_task)
        return step_title, "editor_agent", content, []
    else:
        raise ValueError(f"Unknown step type: {step_title}")
