import os
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# === CONFIGURACIÓN ===
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minam_llm_data")
os.makedirs(BASE, exist_ok=True)

# === 1️⃣ Textos legales base (resúmenes reales del MINAM) ===
documentos = {
    "Decreto Supremo 006-2024-MINAM":
        """Aprueba el Reglamento para la gestión integral de residuos sólidos a nivel nacional.
        Define responsabilidades compartidas entre productores, distribuidores, gobiernos locales y ciudadanía.
        Promueve la economía circular y la reducción de la contaminación ambiental.""",

    "Resolución Ministerial 111-2024-MINAM":
        """Designa a los funcionarios encargados de supervisar la implementación de los Objetivos de Desarrollo Sostenible (ODS)
        vinculados al sector ambiental. Establece medidas de coordinación intersectorial y elaboración de informes de progreso.""",

    "Resolución Ministerial 099-2024-MINAM":
        """Modifica los lineamientos para la formulación de instrumentos de gestión ambiental del sector energía.
        Incluye requisitos adicionales de mitigación y adaptación al cambio climático en los proyectos energéticos.""",
}

# === 2️⃣ Fragmentar texto ===
def trocear(texto, max_chars=800):
    return [texto[i:i+max_chars] for i in range(0, len(texto), max_chars)]

@st.cache_resource(show_spinner="Cargando modelos...")
def cargar_modelos():
    with st.spinner('Cargando documentos...'):
        chunks = []
        for fuente, texto in documentos.items():
            for frag in trocear(texto):
                chunks.append({"fuente": fuente, "texto": frag})
    
    with st.spinner('Inicializando modelo de embeddings...'):
        # === 3️⃣ Embeddings livianos ===
        modelo_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
        X = modelo_emb.encode([c["texto"] for c in chunks], convert_to_numpy=True)
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
    
    with st.spinner('Cargando modelo de lenguaje...'):
        # === 4️⃣ Modelo de lenguaje ===
        # Usar modelo más pequeño y simple para Streamlit Cloud
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        modelo = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    modelo.eval()  # Poner el modelo en modo evaluación

    # Devolver también la matriz de embeddings X para poder calcular similitudes
    return chunks, modelo_emb, X, index, tokenizer, modelo

def responder(pregunta, chunks, modelo_emb, X, index, tokenizer, modelo, top_k=3, min_sim=0.1):
    # Normalizar y limpiar la pregunta
    pregunta = pregunta.strip()
    if not pregunta.endswith('?'):
        pregunta += '?'
    
    # Embed pregunta
    q_emb = modelo_emb.encode([pregunta], convert_to_numpy=True)

    # Calcular similitud coseno contra la matriz X
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    sims = (X_norm @ q_norm.T).squeeze()
    
    # Obtener documentos relevantes
    top_indices = np.argsort(-sims)[:top_k]
    
    # Mostrar similitudes y fuentes
    st.write("📊 Relevancia de documentos:")
    for idx in top_indices:
        sim_percent = float(sims[idx]) * 100
        fuente = chunks[idx]['fuente']
        st.write(f"- {fuente}: {sim_percent:.1f}%")
    
    # Si no hay documentos relevantes, devolver mensaje
    if float(sims[top_indices[0]]) < min_sim:
        return "Lo siento, no encontré información suficientemente relevante en la normativa para responder esta pregunta. Por favor, reformula tu pregunta o consulta sobre otros temas del MINAM."

    # Preparar contexto para el modelo
    contexto = "\n".join(chunks[idx]["texto"] for idx in top_indices)
    
    # Prompt simple y directo
    prompt = f"Basándote en este contexto: {contexto}\n\nResponde a esta pregunta: {pregunta}\n\nRespuesta:"
    
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generar respuesta con parámetros simples y robustos
    with torch.no_grad():  # Desactivar gradientes para inferencia
        outputs = modelo.generate(
            **inputs,
            max_length=200,  # Permitir respuestas más largas
            min_length=20,
            num_beams=1,     # Búsqueda más simple
            temperature=0.1,  # Más determinista
            do_sample=False,
            early_stopping=True
        )
    
    # Decodificar respuesta
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Formatear respuesta con fuentes
    fuentes = ", ".join(sorted(set(chunks[idx]['fuente'] for idx in top_indices)))
    return f"{respuesta}\n\n📄 Fuentes consultadas: {fuentes}"

    # Obtener top_k índices por similitud
    top_idx = np.argsort(-sims)[:top_k]
    top_sims = sims[top_idx]

    # Si la mejor similitud es baja, devolver fallback prudente
    if float(top_sims[0]) < min_sim:
        return "No tengo información suficiente en la normativa cargada para responder esa pregunta. Intenta reformularla o agrega más documentos."

    contexto = "\n\n".join(chunks[int(i)]["texto"] for i in top_idx)
    fuentes = ", ".join(sorted(set(chunks[int(i)]['fuente'] for i in top_idx)))

    # Prompt simplificado para el modelo fine-tuned
    prompt = f"Pregunta: {pregunta}\nContexto: {contexto}\nRespuesta:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generación más determinista para evitar invenciones
    output = modelo.generate(**inputs, max_new_tokens=120, num_beams=4, early_stopping=True, do_sample=False)
    respuesta = tokenizer.decode(output[0], skip_special_tokens=True)

    return f"{respuesta}\n\n📄 Fuente consultada: {fuentes}"

# === Interfaz de Streamlit ===
st.set_page_config(
    page_title="🌿 Asistente MINAM-LLM",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Asistente del Ministerio del Ambiente (MINAM-LLM)")

st.markdown("""
Bienvenido al asistente virtual del MINAM. Aquí puedes consultar información sobre:
- Decretos Supremos 📜
- Resoluciones Ministeriales 📋
- Normativa ambiental 🌱
""")

# Cargar modelos
chunks, modelo_emb, X, index, tokenizer, modelo = cargar_modelos()

# Input del usuario
pregunta = st.text_area(
    "Haz tu pregunta sobre normativa del MINAM:",
    placeholder="Ejemplo: ¿Qué regula el Decreto Supremo 006-2024-MINAM?",
    height=100
)

if st.button("📋 Consultar", type="primary"):
    if pregunta:
        with st.spinner("🔍 Analizando documentos relevantes..."):
            # Crear contenedor para la respuesta con estilo
            resp_container = st.container()
            with resp_container:
                respuesta = responder(pregunta, chunks, modelo_emb, X, index, tokenizer, modelo)
                st.success(respuesta)
    else:
        st.warning("⚠️ Por favor, ingresa una pregunta.")

# Ejemplos de preguntas
with st.expander("🔍 Ver ejemplos de preguntas"):
    st.markdown("""
    - ¿Qué regula el Decreto Supremo 006-2024-MINAM?
    - ¿Qué responsabilidades establece para los gobiernos locales?
    - ¿Qué objetivo tiene la Resolución Ministerial 111-2024-MINAM?
    - ¿Qué norma menciona el cambio climático?
    - ¿Cómo contribuyen las tres normas a la sostenibilidad ambiental?
    """)

# Información adicional
st.markdown("---")
st.info("""
#### ℹ️ Información
Este asistente utiliza inteligencia artificial para responder preguntas sobre la normativa 
del Ministerio del Ambiente del Perú. Las respuestas son generadas automáticamente 
basándose en documentos oficiales.
""")