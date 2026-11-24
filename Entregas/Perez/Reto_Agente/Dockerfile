FROM python:3.11-slim
# arranca desde una imagen base que ya trae Python 3.11 instalado, en su versión liviana ("slim")

WORKDIR /app
# crea una carpeta /app dentro del contenedor y la deja como carpeta de trabajo

COPY requirements.txt .
# copia solo este archivo primero, antes que el resto del código

RUN pip install --no-cache-dir -r requirements.txt
# instala las librerías listadas en requirements.txt dentro del contenedor

COPY . .
# copia el resto del proyecto (main.py, agents.py, src/, templates/, etc.)

EXPOSE 8000
# documenta que el servidor adentro del contenedor escucha en el puerto 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# comando que arranca el servidor; --host 0.0.0.0 es necesario para que sea alcanzable desde fuera del contenedor