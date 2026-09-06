import json
import time
import os
import torch
import base64
import shutil
from PIL import Image

# 1. FORZAR LA CACHÉ DE HUGGING FACE AL NETWORK VOLUME (/workspace)
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from huggingface_hub import login
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai

load_dotenv()

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
else:
    print("⚠️ Alerta: Falta HF_TOKEN en el archivo .env.")

client_juez = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ruta_salida = "outputs/resultados_multimodal.json"
os.makedirs("outputs", exist_ok=True)

resultados_multimodal = []
claves_completadas = set()

if os.path.exists(ruta_salida):
    print(f"🔄 Detectado archivo previo en {ruta_salida}. Cargando historial de checkpoints...")
    try:
        with open(ruta_salida, "r", encoding="utf-8") as f:
            resultados_multimodal = json.load(f)
        for r in resultados_multimodal:
            claves_completadas.add(f"{r['id_caso']}_{r['modelo']}")
        print(f"✅ Historial cargado. Se omitirán {len(claves_completadas)} ejecuciones visuales ya guardadas.")
    except Exception as e:
        print(f"⚠️ Error leyendo el archivo previo: {e}. Se iniciará la fase multimodal desde cero.")
        resultados_multimodal = []

with open("data/dataset_calidad.json", "r", encoding="utf-8") as f:
    casos = json.load(f)

def codificar_imagen_base64(ruta_img):
    with open(ruta_img, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

modelos_api_v = ["gemini-2.5-flash", "claude-sonnet-4-6"]

# =========================================================================
# ─── PARTE A: MODELO DE VISIÓN LOCAL (QWEN 2.5 VL 7B INSTRUCT) ────────────
# =========================================================================
model_id_local = "Qwen/Qwen2.5-VL-7B-Instruct"
nombre_modelo_local = "Qwen2.5-VL-7B (FP16)"

necesita_carga_local = False
for caso in casos:
    if f"{caso['id_caso']}_{nombre_modelo_local}" not in claves_completadas:
        necesita_carga_local = True
        break

if necesita_carga_local:
    print(f"\n📥 Cargando Large Multimodal Model local en GPU: {model_id_local}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id_local)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id_local, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        for caso in casos:
            clave_test = f"{caso['id_caso']}_{nombre_modelo_local}"
            if clave_test in claves_completadas:
                continue
                
            if not os.path.exists(caso["ruta_imagen"]):
                print(f"⚠️ Saltando {caso['id_caso']}: Foto no encontrada en {caso['ruta_imagen']}.")
                continue
                
            print(f"🔎 Analizando Localmente con Visión: {caso['id_caso']} | Qwen2.5-VL-7B")
            imagen_pil = Image.open(caso["ruta_imagen"]).convert("RGB")
            
            prompt_texto = f"Actividad Industrial: {caso['actividad_calidad']}\n<CONTEXTO_SOP>\n{caso['sop_contexto']}\n</CONTEXTO_SOP>\n<REPORTE_OPERARIO>\n{caso['ncr_reporte_planta']}\n</REPORTE_OPERARIO>\nInspecciona visualmente la imagen real adjunta. Extrae los datos numéricos o fallos de orden, contrástalos con el SOP y redacta el informe CAPA definitivo en español."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": imagen_pil},
                        {"type": "text", "text": prompt_texto}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1000, 
                    temperature=0.7, 
                    do_sample=True, 
                    repetition_penalty=1.15
                )
            latencia = time.time() - t0
            vram_pico = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            resp = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # CÁLKULO REAL DE TOKENS POR SEGUNDO EN QWEN (LOCAL)
            num_tokens_salida_qwen = len(generated_ids_trimmed[0])
            tokens_seg_real = round(num_tokens_salida_qwen / latencia, 2) if latencia > 0 else 0.0
            
            prompt_juez = f"""SOP Oficial: {caso['sop_contexto']}
Ground Truth: {caso['capa_ground_truth']}
Respuesta LMM Local: {resp}
INSTRUCCIÓN CRÍTICA DE PUNTUACIÓN:
Evalúa cada métrica asignando un número entero estrictamente en el rango de 1 a 5 (donde 1 es pésimo y 5 es excelente). No utilices escalas fuera del 1 al 5.
Devuelve UNICAMENTE un objeto JSON estricto con esta estructura exacta:
{{"precision_tecnica": <int de 1 a 5>, "alineacion_normativa": <int de 1 a 5>, "viabilidad_coste": <int de 1 a 5>, "comentario_justificacion": "<str>"}}"""
            
            try:
                comp_juez = client_juez.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "Eres un auditor técnico experto. Evalúas con rigor y tus notas numéricas deben obligatoriamente ser enteros del 1 al 5."},
                        {"role": "user", "content": prompt_juez}
                    ], 
                    response_format={"type": "json_object"}
                )
                eval_data = json.loads(comp_juez.choices[0].message.content)
            except:
                eval_data = {"precision_tecnica": 1, "alineacion_normativa": 1, "viabilidad_coste": 1, "comentario_justificacion": "Error en evaluación"}
            
            # CÁLCULO DE SOBERANÍA MULTIMODAL LOCAL (NUEVA FÓRMULA ISD)
            try:
                prompt_soberania = f"""Eres un auditor jefe de seguridad industrial y propiedad intelectual. Analiza con extrema rigurosidad la siguiente interacción industrial con imagen de planta:
- Prompt: {prompt_texto}
- Respuesta: {resp}
Evalúa el nivel de sensibilidad y riesgo de confidencialidad del secreto industrial expuesto usando una escala estricta del 1 al 5:
- Nivel 1 (Mínimo/Público): Incidencias genéricas, limpieza o fallos rutinarios.
- Nivel 2 (Bajo): Desvíos operativos menores en líneas secundarias.
- Nivel 3 (Moderado): Fallos en maquinaria estándar o procesos operativos internos.
- Nivel 4 (Alto): Averías graves en sistemas principales o exposición de SOPs avanzados.
- Nivel 5 (Crítico/Secreto Industrial): Fugas de fórmulas químicas, fallos estructurales graves o datos de control críticos.
Devuelve UNICAMENTE un JSON estricto con esta estructura exacta:
{{"nivel_sensibilidad": <int de 1 a 5>, "justificacion_soberania": "<explica detalladamente por qué>"}}"""
                
                resp_sob = client_juez.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_soberania}],
                    response_format={"type": "json_object"}
                )
                datos_sob = json.loads(resp_sob.choices[0].message.content)
                sensibilidad_caso = datos_sob.get("nivel_sensibilidad", 3)
                justificacion_sob = datos_sob.get("justificacion_soberania", "Sin justificación generada")
            except Exception as e:
                sensibilidad_caso = 3
                justificacion_sob = f"Error en generación de justificación: {e}"

            # NUEVA FÓRMULA ISD: ISD = 5.0 - [alpha * (R - 1) + beta * (S - 1)] con alpha=0.7, beta=0.3
            R = 1.0  # Ubicación local
            S = float(sensibilidad_caso)  # Sensibilidad (1 a 5)
            alpha = 0.7
            beta = 0.3
            isd_local_mm = round(5.0 - (alpha * (R - 1.0) + beta * (S - 1.0)), 2)

            resultados_multimodal.append({
                "id_caso": caso['id_caso'], "actividad_calidad": caso['actividad_calidad'],
                "modelo": nombre_modelo_local, "tecnica_prompt": "Multimodal-RAG", "cuantizacion": "FP16",
                "latencia_s": round(latencia, 2), "tokens_seg": tokens_seg_real, "vram_gb": round(vram_pico, 2),
                "respuesta_ia": resp, "evaluacion": eval_data, 
                "soberania_dato": isd_local_mm,
                "justificacion_soberania": justificacion_sob
            })
            
            with open(ruta_salida, "w", encoding="utf-8") as f_out:
                json.dump(resultados_multimodal, f_out, indent=4, ensure_ascii=False)
                
        del model, processor
        torch.cuda.empty_cache()
        
        carpeta_modelo_cache = os.path.join("/workspace/.cache/huggingface/hub", f"models--{model_id_local.replace('/', '--')}")
        if os.path.exists(carpeta_modelo_cache):
            print(f"🧹 Purgando archivos de caché en disco para {model_id_local}...")
            shutil.rmtree(carpeta_modelo_cache, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Error crítico en el bloque LMM local: {e}")
else:
    print("⏭️ Saltando bloque LMM local: Todas las imágenes ya fueron procesadas y guardadas.")

# =========================================================================
# ─── PARTE B: MODELOS MULTIMODALES EN LA NUBE (SDKs PROPIOS) ─────────────
# =========================================================================
print("\n🌐 Lanzando consultas visuales nativas a APIs externas...")
for m_api in modelos_api_v:
    for caso in casos:
        clave_test = f"{caso['id_caso']}_{m_api}"
        
        if clave_test in claves_completadas:
            continue
        if not os.path.exists(caso["ruta_imagen"]): 
            continue
            
        print(f"🔎 Analizando con API Visual Nativa: {caso['id_caso']} | {m_api}")
        
        prompt_texto = f"Actividad Industrial de Calidad: {caso['actividad_calidad']}\n<CONTEXTO_SOP>\n{caso['sop_contexto']}\n</CONTEXTO_SOP>\n<REPORTE_OPERARIO>\n{caso['ncr_reporte_planta']}\n</REPORTE_OPERARIO>\nGenera el plan CAPA estructurado en español basándote en el análisis óptico de la evidencia."
        
        t0 = time.time()
        resp = ""
        output_tokens = 0
        try:
            if "gemini" in m_api.lower():
                img_gemini = Image.open(caso["ruta_imagen"])
                response = client_gemini.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[img_gemini, prompt_texto]
                )
                resp = response.text
                # EXTRACCIÓN REAL DE TOKENS DE SALIDA EN GEMINI
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            elif "claude" in m_api.lower():
                base64_data = codificar_imagen_base64(caso["ruta_imagen"])
                response = client_anthropic.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_data,
                                    },
                                },
                                {"type": "text", "text": prompt_texto}
                            ],
                        }
                    ]
                )
                for block in response.content:
                    if block.type == "text":
                        resp = block.text
                        break
                # EXTRACCIÓN REAL DE TOKENS DE SALIDA EN CLAUDE
                if hasattr(response, 'usage') and response.usage:
                    output_tokens = getattr(response.usage, 'output_tokens', 0)
            
            latencia = time.time() - t0
        except Exception as e:
            print(f"⚠️ Fallo en el canal de red del SDK {m_api}: {e}")
            continue

        # CÁLCULO DE TOKENS POR SEGUNDO REAL EN APIS CLOUD
        tokens_seg_real = round(output_tokens / latencia, 2) if latencia > 0 and output_tokens > 0 else 0.0

        prompt_juez = f"""SOP Oficial: {caso['sop_contexto']}
Ground Truth: {caso['capa_ground_truth']}
Respuesta de la API Evaluada: {resp}
INSTRUCCIÓN CRÍTICA DE PUNTUACIÓN:
Evalúa cada métrica asignando un número entero estrictamente en el rango de 1 a 5 (donde 1 es pésimo y 5 es excelente). No utilices escalas fuera del 1 al 5.
Devuelve UNICAMENTE un objeto JSON estricto con esta estructura exacta:
{{"precision_tecnica": <int de 1 a 5>, "alineacion_normativa": <int de 1 a 5>, "viabilidad_coste": <int de 1 a 5>, "comentario_justificacion": "<str>"}}"""
        
        try:
            comp_juez = client_juez.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": "Eres un auditor técnico experto. Evalúas con rigor y tus notas numéricas deben obligatoriamente ser enteros del 1 al 5."},
                    {"role": "user", "content": prompt_juez}
                ], 
                response_format={"type": "json_object"}
            )
            eval_data = json.loads(comp_juez.choices[0].message.content)
        except:
            eval_data = {"precision_tecnica": 1, "alineacion_normativa": 1, "viabilidad_coste": 1, "comentario_justificacion": "Error en evaluación"}
        
        # CÁLCULO DE SOBERANÍA MULTIMODAL CLOUD (NUEVA FÓRMULA ISD)
        try:
            prompt_soberania = f"""Eres un auditor jefe de seguridad industrial y propiedad intelectual. Analiza con extrema rigurosidad la siguiente interacción industrial con imagen enviada a nube:
- Prompt: {prompt_texto}
- Respuesta: {resp}
Evalúa el nivel de sensibilidad y riesgo de confidencialidad del secreto industrial expuesto usando una escala estricta del 1 al 5:
- Nivel 1 (Mínimo/Público): Incidencias genéricas, limpieza o fallos rutinarios.
- Nivel 2 (Bajo): Desvíos operativos menores en líneas secundarias.
- Nivel 3 (Moderado): Fallos en maquinaria estándar o procesos operativos internos.
- Nivel 4 (Alto): Averías graves en sistemas principales o exposición de SOPs avanzados.
- Nivel 5 (Crítico/Secreto Industrial): Fugas de fórmulas químicas, fallos estructurales graves o datos de control críticos.
Devuelve UNICAMENTE un JSON estricto con esta estructura exacta:
{{"nivel_sensibilidad": <int de 1 a 5>, "justificacion_soberania": "<explica detalladamente por qué>"}}"""
            
            resp_sob = client_juez.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_soberania}],
                response_format={"type": "json_object"}
            )
            datos_sob = json.loads(resp_sob.choices[0].message.content)
            sensibilidad_caso = datos_sob.get("nivel_sensibilidad", 3)
            justificacion_sob = datos_sob.get("justificacion_soberania", "Sin justificación generada")
        except Exception as e:
            sensibilidad_caso = 3
            justificacion_sob = f"Error en generación de justificación: {e}"

        # NUEVA FÓRMULA ISD: Asumiendo R = 5.0 para modelos SaaS en la nube externa
        R = 5.0  
        S = float(sensibilidad_caso)  
        alpha = 0.7
        beta = 0.3
        isd_api_mm = round(5.0 - (alpha * (R - 1.0) + beta * (S - 1.0)), 2)

        resultados_multimodal.append({
            "id_caso": caso['id_caso'], 
            "actividad_calidad": caso['actividad_calidad'],
            "modelo": m_api, 
            "tecnica_prompt": "Multimodal-RAG", 
            "cuantizacion": "SaaS-Cloud",
            "latencia_s": round(latencia, 2), 
            "tokens_seg": tokens_seg_real, 
            "vram_gb": 0.0,
            "respuesta_ia": resp, 
            "evaluacion": eval_data, 
            "soberania_dato": isd_api_mm,
            "justificacion_soberania": justificacion_sob
        })
        
        with open(ruta_salida, "w", encoding="utf-8") as f_out:
            json.dump(resultados_multimodal, f_out, indent=4, ensure_ascii=False)

print("🏁 ¡Fase Multimodal completada, purgada en disco y blindada con Qwen2.5-VL!")