from fastapi import FastAPI
from pydantic import BaseModel
from agents import planner_agent, investigar_y_escribir
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import Form
import csv

templates = Jinja2Templates(directory="templates")

app = FastAPI()

class PeticionInput(BaseModel):
    peticion: str
    
@app.get("/")                      # crea una ruta GET en la dirección raíz ("/")
def read_root():              
    return {"status": "ok", "mensaje": "el servidor está vivo"}   # responde con un JSON fijo

@app.post("/generate_report")      # crea una ruta POST en la dirección /generate_report
def generate_report(datos: PeticionInput):
    sub_temas = planner_agent(datos.peticion)   # descompone la petición amplia en una lista de sub-temas

    resultados = []                              # aquí se va guardando un resumen por cada sub-tema
    for sub_tema in sub_temas:
        resumen = investigar_y_escribir(sub_tema)        # ciclo completo: investigar, escribir, revisar, corregir
        resultados.append({"sub_tema": sub_tema, "resumen": resumen})

    return {"peticion": datos.peticion, "resultados": resultados}   # FastAPI convierte esto en JSON automáticamente

@app.get("/formulario")
def mostrar_formulario(request: Request):
    return templates.TemplateResponse(request, "index.html")   # orden nuevo: primero request, luego el nombre del archivo

@app.post("/generate_report_web")
def generate_report_web(request: Request, peticion: str = Form(...)):
    sub_temas = planner_agent(peticion)             

    resultados = []
    for sub_tema in sub_temas:
        resumen = investigar_y_escribir(sub_tema)
        resultados.append({"sub_tema": sub_tema, "resumen": resumen})

    return templates.TemplateResponse(
        request,
        "index.html",
        {"resultados": resultados, "peticion": peticion}   # estos datos quedan disponibles dentro del HTML
    )
    
@app.get("/bibliografia")
def ver_bibliografia(request: Request):
    referencias = []
    with open("referencias_malaria.csv", "r", newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)          # lee cada fila del CSV como un diccionario
        for fila in lector:
            referencias.append(fila)

    return templates.TemplateResponse(request, "bibliografia.html", {"referencias": referencias})