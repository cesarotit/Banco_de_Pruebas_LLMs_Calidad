Banco de Pruebas Multicriterio para la Gestión de Calidad Industrial 




---

##  Estructura del Proyecto

```plaintext
tfg_calidad_benchmark/
│
├── data/
│   ├── imagenes/                       # 20 Evidencias visuales reales de planta (.jpg)
    ├── dataset_calidad_aumentada.json  # Dataset aumentado en texto
│   └── dataset_calidad.json            # Dataset híbrido (SOP + NCR + CAPA Ground Truth)
│
├── outputs/
│   ├── resultados_unimodal.json      # Logs de inferencia y notas del Juez de la Fase 1
│   ├── resultados_multimodal.json    # Logs de inferencia y notas del Juez de la Fase 2
│   └── resultados_lora.json          # Logs de inferencia y notas del Juez de la Fase 3
│   
│
│
├── evaluador_unimodal.py             # Simulación de la Fase 1.0 (Modelos × Casos × 4 Técnicas)
├── evaluador_multimodal.py           # Simulación de la Fase 2.0 (LMM Local + Visión Real)
├── evaluador_lora.py                 # Simulación de la Fase 3.0 (LMM Local + Visión Real)
├── requirements.txt                  # Dependencias estrictas del entorno virtual
└── README.md                         # Guía metodológica y de réplica del sistema
```

---

##  Diseño Experimental y Matriz de Pruebas

El sistema evalúa el rendimiento cruzado de la Inteligencia Artificial mediante una matriz que abarca 4 actividades de la ingeniería de calidad divididas en 5 casos por bloque:
1. **Calidad de Recepción (Gestión de Proveedores y Materiales)**
2. **Control Estadístico de Procesos (SPC)**
3. **Metrología Dimensional y Calibración**
4. **Aseguramiento y Cumplimiento Normativo (Auditorías / 5S)**

### Variables Tecnológicas del Experimento
*   **Modelos Evaluados:** `Llama-3.2-3B`, `Qwen-2.5-7B`, `Llama-3.1-8B`, y `Qwen-2.5-VL-7B`.
*   **Compresión de Peso:** Cuantizaciones nativas en **Q4_K_M**, **Q8_0** y precisión **FP16** sin comprimir.
*   **In-Context Learning (Prompting):** Ejecución combinatoria en *Zero-Shot*, *Few-Shot*, *Chain-of-Thought (CoT)* y *RAG-Injected*.
*   **LoRA:** Ajuste fino con adaptadores de bajo rango.
*   **Evaluación Científica:** Metodología *LLM-as-a-Judge* (con interfaz estructurada JSON) que analiza la precisión técnica de la acción correctiva (CAPA), alineación con las cláusulas ISO y viabilidad de costes en planta.

---

##  Instrucciones de Despliegue y Ejecución (Entorno RunPod / CUDA)

### 1. Preparación del Entorno Virtual e Instalación
Clone el repositorio en su servidor o máquina virtual local con soporte para GPU dedicada e instale las dependencias estrictas de PyTorch y Hugging Face:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuración de Variables de Entorno (Claves de Seguridad)
Antes de iniciar los bucles de inferencia, configure su clave de acceso para el modelo "Juez" en la terminal de Linux. Reemplace el valor con sus credenciales oficiales:

```bash
export OPENAI_API_KEY="sk-proj-tu-clave-aqui..."
export GEMINI_API_KEY="sk-proj-tu-clave-aqui..."
export ANTHROPIC_API_KEY="sk-proj-tu-clave-aqui..."
```

### 3. Ejecución de las Simulaciones 
Para garantizar que los experimentos masivos de la combinatoria de prompts no se detengan ante cortes de red o desconexiones del navegador en RunPod, ejecute los pipelines utilizando comandos: 

```bash
# Lanzar la Fase 1.0 Unimodal (Texto puro: 240 ejecuciones)
python evaluador_unimodal.py

# Lanzar la Fase 2.0 Multimodal (Visión + Imagen real)
 python evaluador_multimodal.py

# Lanzar la Fase 3.0 LoRA (Visión + Imagen real)
 python evaluador_lora.py
```






---

##  Métricas  Evaluadas
El sistema analiza de forma integrada los criterios de decisión que gobiernan una planta de operaciones:
*   **Precisión Técnica:** Puntuación (1-5) del diagnóstico analítico causa-raíz frente al Ground Truth experto.
*   **Alineación ISO:** Puntuación (1-5) de la exactitud en la identificación de la cláusula técnica infringida de la norma.
*   **Eficiencia Coste:** Puntuación (1-5) de la evaluación sobre el coste de las medidas propuestas por el modelo.   
*   **Latencia:** Tiempo total de generación de la respuesta del modelo.
*   **Velocidad de Inferencia:** Velocidad media de respuesta computacional medida de forma nativa en tokens por segundo.
*   **VRAM:** Demanda máxima de memoria de vídeo en la GPU local para delimitar los requerimientos de hardware.
*   **Soberanía del Dato:** Puntuación de seguridad de la propiedad intelectual (Mantenimiento 100% *On-Premise* sin fugas de información hacia nubes de terceros).

---

