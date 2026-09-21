# Lab01 Agentes Locales

Este informe detalla la instalación desde cero, el funcionamiento interno y las ventajas de los dos patrones principales de diseño de agentes de Inteligencia Artificial explorados en el entorno de desarrollo: el Patrón de Reflexión (Self-Reflection) y el Patrón de Uso de Herramientas (Tool Calling).

Ambos patrones están diseñados para ejecutarse de manera 100% local utilizando Ollama, garantizando privacidad, control total sobre el entorno y sin depender de claves de API externas. Los ejemplos prácticos se enfocan en la resolución y explicación de problemas de física (ej. termodinámica cuántica, ajuste de datos experimentales).

Por conveniencia Para tener todos los notebooks agénticos en un solo directorio. Este informe estará en laboratorio 1 Y los demás archivos en la oratoria dos.

## 1. Instalación y Configuración Base (Entorno Local)

Para que ambos patrones funcionen correctamente, se requiere una base sólida que aísle las dependencias de Python y gestione los modelos locales sin comprometer el sistema operativo. El flujo recomendado utiliza WSL (Windows Subsystem for Linux) y pyenv.

### 1.1.a Configuración Avanzada de WSL (Opcional: Instalar desde cero)

#### Mover o Instalar WSL en un disco secundario

Si aún no tienes WSL, instálalo primero con `wsl --install -d Ubuntu` (o la distribución de tu preferencia) y configura tu usuario y contraseña. Dado que Windows instala WSL por defecto en el disco C:, el método más limpio es instalar la distribución normalmente y luego exportarla e importarla al disco deseado.

#### Verifica el nombre y estado de tu distribución

Abre PowerShell o la Terminal de Windows y comprueba que el estado sea "Stopped". Si dice "Running", apágala con `wsl --terminate <NombreDistro>` o `wsl --shutdown`.

```powershell
wsl --list -v

```

#### Crea los directorios de destino en tu nuevo disco

Por ejemplo, en el disco D:.

```powershell
mkdir D:\WSL\Backups
mkdir D:\WSL\Ubuntu

```

#### Exporta la distribución a un archivo `.tar`

Este paso crea una copia de seguridad exacta de tu sistema actual.

```powershell
wsl --export Ubuntu D:\WSL\Backups\ubuntu_backup.tar

```

#### Desregistra la instalación original del disco C

Esto elimina la distribución original del disco principal, por lo que debes asegurarte de que la exportación anterior terminó sin errores.

```powershell
wsl --unregister Ubuntu

```

#### Importa la distribución en el nuevo disco

Esto desempaqueta el sistema en la nueva ubicación.

```powershell
wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\Backups\ubuntu_backup.tar

```

### Restaurar el Usuario por Defecto y Permisos

Al importar una distribución desde un archivo `.tar`, WSL olvida cuál era el usuario por defecto y te iniciará sesión como `root`. Además, los permisos de los archivos montados desde Windows suelen romperse, mostrando permisos `777` y provocando que `chmod` deje de funcionar.

#### Inicia sesión en tu distribución recién movida

Notarás que el prompt ahora dice `root@...`.

```powershell
wsl -d Ubuntu

```

### 1.1.b Configuración Avanzada de WSL (Opcional: Migración a otro disco)

Si necesitas mover WSL para liberar espacio en C::

Exporta la distro actual: `wsl --export Ubuntu D:\WSL\Backups\ubuntu_backup.tar`

Desregistra la original: wsl --unregister Ubuntu`

Importa en el nuevo disco: `wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\Backups\ubuntu_backup.tar`

### 1.2 Restaurar permisos

Edita `/etc/wsl.conf` en Linux para definir el usuario por defecto y arreglar los permisos de los volúmenes montados:

```ini
[user]
default=tu_usuario
```

Reinicia WSL: `wsl --shutdown`

#### Habilita los metadatos para corregir permisos (`chmod`)

En el mismo archivo `/etc/wsl.conf`, añade la configuración de automontaje. Esto le indica a WSL que guarde los metadatos de Linux (como propietarios o permisos de `chmod`) en los archivos de los discos de Windows.

```ini
[automount]
enabled = true
options = "metadata,umask=022,fmask=011,case=dir"

```

**Nota:** Por alguna razón, esto no me funcionó del todo y cada. Es que reinicio el computador y activo Windows Subsystem for Linux. La primera vez me aparece un mensaje de error; luego todo parece funcionar correctamente.

#### Reinicia WSL para aplicar los cambios

Guarda el archivo, vuelve a PowerShell y apaga WSL por completo.

```powershell
wsl --shutdown

```

#### Verifica y repara permisos críticos

Abre WSL nuevamente; ahora deberías entrar con tu usuario normal. Si tienes claves SSH u otros archivos sensibles en tu directorio `~`, restaura sus permisos:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

```

## 2. Instalación de Ollama y Modelos

Instala el motor local ejecutando:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Descarga los modelos requeridos para los laboratorios:

```bash
ollama pull llama3.2:1b  # Modelo rápido y ligero
ollama pull qwen3:1.7b   # Modelo alternativo más robusto en reflexión
```

### 2.1 Entorno de Python con pyenv

Usar `pyenv` permite instalar múltiples versiones de Python a nivel de usuario aislando tus proyectos del Python interno del sistema operativo, evitando así conflictos críticos en Ubuntu/WSL.

* **Protección del sistema operativo:** Aisla los proyectos de desarrollo del Python global, evitando que modificaciones o instalaciones de librerías rompan herramientas críticas del sistema (como `apt`). Todo opera desde la carpeta local del usuario (`~/.pyenv`) sin requerir permisos de administrador.

* **Gestión de múltiples versiones:** Permite descargar, compilar y mantener varias versiones de Python coexistiendo pacíficamente para usarlas según lo requiera cada proyecto.

* **Activación automática:** Detecta el archivo oculto `.python-version` al entrar a un directorio y activa el entorno virtual correspondiente en segundo plano, eliminando la necesidad de comandos manuales como `source` o `deactivate`.

* **Ventajas frente a alternativas:** A diferencia de `venv`, `pyenv` puede descargar e instalar versiones nuevas de Python que el sistema no tiene. Además, es más ligero y transparente que Conda, integrándose perfectamente con el ecosistema estándar de `pip` sin modificar la terminal ni generar conflictos.

Instala dependencias de compilación (build-essential, libssl-dev, etc.). Ubuntu necesita estas librerías para compilar el código fuente de Python.

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev \
libffi-dev liblzma-dev python3-openssl git
```

#### Instala Pyenv

Ejecuta el script oficial para instalar `pyenv` y el plugin `pyenv-virtualenv`.

```bash
curl https://pyenv.run | bash

```

#### Configura las variables de entorno en tu Shell

Abre tu archivo de configuración (ej. `.bashrc` o `.zshrc`).

```bash
nano ~/.bashrc
```

#### Instala la versión de Python requerida y crea el entorno

```bash
pyenv install 3.11.16
pyenv virtualenv 3.11.16 Agente_Lab02
pyenv local Agente_Lab02
```

Actualiza `pip` e instala el stack científico y de IA (requirements.txt):

```bash
python -m pip install --upgrade pip
pip install fastapi sqlalchemy openai pdfminer.six scipy pandas matplotlib jupyter
```

**Nota:** En la arquitectura del proyecto, el script run.py automatiza la verificación de pyenv, la actualización de pip, la instalación de dependencias y el registro del kernel de Jupyter.

## 3. Patrón de Reflexión (Self-Reflection)

El patrón de reflexión permite que el LLM se audite a sí mismo antes de entregar una respuesta final, simulando un ciclo de revisión por pares.

### ¿Cómo funciona Self-Reflection?

A diferencia de los agentes tradicionales (Zero-shot) que generan texto en un solo paso, este patrón implementa un ciclo iterativo en el código (`run_reflection`): Generar $\rightarrow$ Criticar $\rightarrow$ Revisar.
Se divide el mismo LLM en dos "personajes" mediante instrucciones de sistema *(System Prompts)*:

* Generación (**Rol: Explicador**): El LLM actúa como un profesor. Recibe un concepto, por ejemplo, "¿Existe la temperatura absoluta negativa en un sistema cuántico basado en la beta termodinámica $\beta$?", y genera un borrador inicial.

* Crítica (**Rol: Revisor**): El mismo modelo asume un rol experto y estricto. Se le pasa la explicación y responde en un formato exacto (`<APROBADA>` o `<RECHAZADA>`), analizando errores conceptuales

* Corrección: Si es rechazada, se inyectan los comentarios críticos de vuelta al Explicador como un historial de chat: "Un revisor encontró este problema... Escribe una versión corregida completa."

* Bucle: Se repite hasta la aprobación o hasta alcanzar un límite (`max_rounds = 3`).

### Ventajas del Patrón de Reflexión

Aumento drástico de la precisión lógica: Modelos pequeños (como `llama3.2:1b`) suelen alucinar o dar respuestas ambiguas en el primer intento. La reflexión fuerza al modelo a notar sus propias contradicciones (ej. confundir cero absoluto con temperaturas termodinámicas negativas).

* **Independencia de APIs Externas:** No requiere herramientas externas de validación; todo ocurre explotando el espacio latente del LLM.

* **Transparencia (Traza):** Al guardar la traza de la conversación en Jupyter (show_trace), el usuario puede ver la evolución del razonamiento y confirmar que la corrección tiene fundamentos.

* **Flexibilidad Evaluativa:** Cambiando el REVISOR_PROMPT puedes alterar el objetivo de la revisión (rigor físico, tono empático, revisión ortográfica).

### Patrón de Uso de Herramientas (Tool Calling)

El Tool Calling permite que el LLM abandone la limitación de ser solo un "generador de texto estático" y tome acciones en el entorno, delegando tareas deterministas al lenguaje de programación.

#### ¿Cómo funciona Tool Calling?

En este laboratorio, el agente actúa como un analista de física experimental usando funciones en Python (`physics_tools.py`).

* **Definición de Herramientas:** Se exponen funciones reales al LLM documentadas mediante Type Hints y Docstrings. Por ejemplo:

* `fit_dataset(dataset: str, model: str)`: Realiza un ajuste por mínimos cuadrados usando scipy.optimize.curve_fit.

* `plot_dataset_fit(dataset: str, model: str)`: Genera y guarda un gráfico usando matplotlib.

* **Razonamiento y Decisión:** Ante un prompt ("Ajusta un modelo lineal a los datos del resorte y dame la constante $k$"), el LLM analiza los esquemas JSON de las funciones. Decide detener la generación de texto y emitir un comando estructurado: `{"name": "fit_dataset", "arguments": {"dataset": "resorte", "model": "lineal"}}`.

* **Ejecución Local:** El script `agent.py` intercepta esta intención, ejecuta la función de Python en la máquina local (calculando incertidumbres y métricas como $R^2$), y extrae los resultados numéricos o rutas de imagen.

* **Síntesis:** El programa inyecta el resultado crudo de Python en la memoria del agente bajo el rol de tool. El LLM lee esta información y redacta una respuesta natural (ej. "La constante $k$ es $25.71 \pm 0.68$ N/m").

#### Ventajas del Patrón de Uso de Herramientas

* **Precisión Matemática:** Los LLMs son inherentemente deficientes en aritmética compleja. Al delegar ajustes estadísticos y operaciones matemáticas a Python, se garantiza precisión científica siempre y cuando el modelo interprete el input y utilice bien la herramienta.

* **Interacción Real:** El agente puede leer archivos locales (.csv), generar imágenes y guardarlas en disco físico, interactuando con tu sistema operativo.

* **Mitigación de Alucinaciones:** Dado que las métricas (como la constante temporal $\tau$ en un circuito RC) se originan del retorno JSON de la herramienta y no del "conocimiento general" del modelo, las respuestas son factuales y basadas en los datos experimentales reales.

* **Extensibilidad Modular:** Solo requiere programar una nueva función en Python e incluirla en la lista tools. El LLM asimilará su esquema automáticamente para peticiones futuras.

## 4. Conclusión

El uso local de Ollama combinado con arquitecturas modulares demuestra que no es necesario un modelo gigantesco (ni suscripciones costosas) para lograr flujos de trabajo inteligentes.

El Tool Calling cubre los puntos ciegos técnicos del modelo, delegando el "trabajo pesado y determinista" a Python.

El Patrón de Reflexión cubre los puntos ciegos cognitivos, obligando al modelo a iterar y pulir la interpretación teórica de esos resultados.

La sinergia de ambos permite ecosistemas robustos: Un agente extrae y ajusta datos físicamente con herramientas, y luego emplea reflexión para asegurar que sus conclusiones escritas sean impecables teóricamente.
