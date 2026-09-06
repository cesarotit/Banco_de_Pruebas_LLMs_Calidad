import os
import json
import time
import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# 0. ENTORNO Y RUTAS
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
load_dotenv()
client_juez = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_ADAPTER_DIR = "modelos/qwen2.5_vl_7b_lora_augmented_adaptador"
OUTPUT_MERGED_DIR = "modelos/qwen2.5_vl_7b_lora_augmented_merged"
RUTA_SALIDA = "outputs/resultados_lora_augmented_test.json"
DATASET_AUGMENTED_PATH = "data/dataset_calidad_augmented.json"

os.makedirs("outputs", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

# 1. PARTICIÓN (AISLAMIENTO TEST CIEGO)
with open(DATASET_AUGMENTED_PATH, "r", encoding="utf-8") as f:
    casos = json.load(f)

test_base_ids = {"CAL_005", "CAL_010", "CAL_015", "CAL_020"}
test_cases = [c for c in casos if c["id_caso"] in test_base_ids]
train_cases = [
    c for c in casos 
    if not any(c["id_caso"].startswith(t_id) for t_id in test_base_ids)
]

print(f"📊 Partición activa: {len(train_cases)} Train | {len(test_cases)} Test Ciego.")

def construir_prompt_usuario(caso):
    return (
        f"Actividad Industrial: {caso.get('actividad_calidad', '')}\n"
        f"<CONTEXTO_SOP>\n{caso.get('sop_contexto', '')}\n</CONTEXTO_SOP>\n"
        f"<REPORTE_OPERARIO>\n{caso.get('ncr_reporte_planta', '')}\n</REPORTE_OPERARIO>\n"
        "Inspecciona la imagen y redacta el informe CAPA en español, incorporando "
        "las causas físicas y parámetros numéricos definidos en el <CONTEXTO_SOP>."
    )

def formatear_conversacion(caso):
    prompt_texto = construir_prompt_usuario(caso)
    gt = caso.get("capa_ground_truth", {})
    resp = (
        f"### Plan CAPA Oficial:\n"
        f"Contención: {gt.get('contencion', '')} "
        f"Causa Raíz: {gt.get('causa_raiz', '')} "
        f"Acción Correctiva: {gt.get('accion_correctiva', '')}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": caso["ruta_imagen"]},
                {"type": "text", "text": prompt_texto},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": resp}]},
    ]

train_conversations = [formatear_conversacion(c) for c in train_cases if os.path.exists(c["ruta_imagen"])]

# 2. MODELO BASE Y PROCESADOR
processor = AutoProcessor.from_pretrained(MODEL_ID)
dtype_computo = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    dtype=dtype_computo, 
    device_map="auto"
)

# 3. LORA REGULARIZADO
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False

# 4. DATASET CON MASKING Y ANCLAJE <|im_end|>
class LoRAUniformDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        self.eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]
        image_inputs, video_inputs = process_vision_info(messages)

        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_inputs = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs, padding=False, return_tensors="pt"
        )

        prompt_text = self.processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs, padding=False, return_tensors="pt"
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        if input_ids[-1] != self.eos_id:
            input_ids = torch.cat([input_ids, torch.tensor([self.eos_id], dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.tensor([1], dtype=attention_mask.dtype)])

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            item["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)
        return item

dataset_train = LoRAUniformDataset(train_conversations, processor)

def vision_data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if "pixel_values" in features[0]:
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
    if "image_grid_thw" in features[0]:
        batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features], dim=0)
    return batch

# 5. ARGUMENTOS DE ENTRENAMIENTO
training_args = TrainingArguments(
    output_dir=OUTPUT_ADAPTER_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=2,
    num_train_epochs=3,
    weight_decay=0.05,
    bf16=(dtype_computo == torch.bfloat16),
    fp16=(dtype_computo == torch.float16),
    logging_steps=5,
    save_strategy="no",
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset_train,
    data_collator=vision_data_collator
)

print("\n🚀 Iniciando entrenamiento LoRA balanceado...")
trainer.train()

model.save_pretrained(OUTPUT_ADAPTER_DIR)
processor.save_pretrained(OUTPUT_ADAPTER_DIR)

# 6. FUSIÓN DE CAPAS (FP16/BF16)
print("\n⚙️ Fusionando adaptadores LoRA...")
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_MERGED_DIR)
processor.save_pretrained(OUTPUT_MERGED_DIR)
print(f"✅ Modelo consolidado guardado en: {OUTPUT_MERGED_DIR}")

# 7. INFERENCIA Y EVALUACIÓN CIEGA
print("\n🔬 Evaluando casos de test...")
model.eval()
model.config.use_cache = True
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
stop_ids = [eos_id]
if processor.tokenizer.eos_token_id is not None:
    stop_ids.append(processor.tokenizer.eos_token_id)
stop_ids = list(set([i for i in stop_ids if i is not None and isinstance(i, int)]))

resultados = []

for caso in test_cases:
    if not os.path.exists(caso["ruta_imagen"]):
        continue

    prompt_texto = construir_prompt_usuario(caso)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": caso["ruta_imagen"]},
            {"type": "text", "text": prompt_texto},
        ],
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=250,
            eos_token_id=stop_ids,
            pad_token_id=processor.tokenizer.pad_token_id or eos_id,
            repetition_penalty=1.15,
            use_cache=True
        )

    torch.cuda.synchronize()
    latencia = round(time.perf_counter() - t0, 2)
    vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    tokens_in = int(inputs["input_ids"].shape[1])
    tokens_out = int(generated_ids.shape[1] - tokens_in)
    vel_t_s = round(tokens_out / latencia, 2) if latencia > 0 else 0.0

    out_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    resp = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    gt = caso.get("capa_ground_truth", {})
    gt_str = json.dumps(gt, ensure_ascii=False)

    prompt_juez = f"""SOP: {caso.get('sop_contexto', '')}
Ground Truth: {gt_str}
Respuesta Evaluada: {resp}

Devuelve ÚNICAMENTE un objeto JSON:
{{"precision_tecnica": <int 1-5>, "alineacion_normativa": <int 1-5>, "viabilidad_coste": <int 1-5>, "comentario_justificacion": "<str>"}}"""

    comp = client_juez.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un auditor técnico de calidad industrial. Responde siempre en formato JSON."},
            {"role": "user", "content": prompt_juez},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    eval_data = json.loads(comp.choices[0].message.content)

    resultados.append({
        "id_caso": caso["id_caso"],
        "modelo": "Qwen2.5-VL-7B (LoRA Homogéneo N=80)",
        "prompt_tokens": tokens_in,
        "tokens_salida": tokens_out,
        "latencia_s": latencia,
        "tokens_seg": vel_t_s,
        "vram_gb": vram_gb,
        "respuesta_ia": resp,
        "evaluacion": eval_data,
    })
    print(f"Caso {caso['id_caso']} -> Latencia: {latencia}s | Vel: {vel_t_s} t/s | Tokens: {tokens_out} | Prec={eval_data.get('precision_tecnica')} Norm={eval_data.get('alineacion_normativa')}")

with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

print(f"\n🏁 Pipeline finalizado con éxito. Métricas en: {RUTA_SALIDA}")