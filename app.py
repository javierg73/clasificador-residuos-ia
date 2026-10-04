import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_residuos_savedmodel")
CLASES = ["PAPEL", "CARTÓN", "VIDRIO", "PLÁSTICO"]
CONTENEDORES = {
    "PAPEL": "🔵 Contenedor azul",
    "CARTÓN": "🔵 Contenedor azul",
    "VIDRIO": "🟢 Si es un envase de vidrio: contenedor verde. Para cristalería u otros objetos de vidrio, consulta las normas locales.",
    "PLÁSTICO": "🟡 Si es un envase de plástico: contenedor amarillo. Para otros objetos de plástico, consulta las normas locales.",
}

modelo = tf.saved_model.load(MODEL_PATH)
inferir = modelo.signatures["serve"]

def clasificar(imagen):
    if imagen is None:
        return "Sube o toma una fotografía para realizar la clasificación."
    if not isinstance(imagen, Image.Image):
        imagen = Image.fromarray(imagen)
    imagen = imagen.convert("RGB")
    array = np.array(imagen, dtype=np.float32)
    array = tf.image.resize(array, (224, 224))
    tensor = tf.expand_dims(array, 0)
    salida = inferir(keras_tensor=tensor)
    probabilidades = list(salida.values())[0].numpy()[0]
    orden = np.argsort(probabilidades)[::-1]
    indice, segundo = int(orden[0]), int(orden[1])
    clase = CLASES[indice]
    confianza = float(probabilidades[indice]) * 100
    clase_2 = CLASES[segundo]
    confianza_2 = float(probabilidades[segundo]) * 100
    if confianza < 70:
        aviso = f"⚠️ Resultado poco concluyente. Segunda posibilidad: {clase_2} ({confianza_2:.1f}%). Prueba otra fotografía con buena iluminación."
    else:
        aviso = "ℹ️ Resultado orientativo. El prototipo puede equivocarse y las normas de separación dependen del tipo de objeto y de la normativa local."
    return (f"## ♻️ Material identificado: **{clase}**\n\n"
            f"**Confianza del modelo:** {confianza:.1f}%\n\n"
            f"### Orientación\n{CONTENEDORES[clase]}\n\n{aviso}")

with gr.Blocks(title="Asistente inteligente para la separación de residuos") as demo:
    gr.Markdown("""# ♻️ Asistente inteligente para la separación de residuos
Sube o toma una fotografía de un residuo doméstico. El prototipo intenta identificar si es **papel, cartón, vidrio o plástico** y ofrece una orientación para su separación.

**Proyecto académico de IA · Samsung Innovation Campus – UMA**""")
    with gr.Row():
        entrada = gr.Image(type="pil", sources=["upload", "webcam"], label="Fotografía del residuo")
        salida = gr.Markdown("Sube o toma una fotografía para comenzar.")
    boton = gr.Button("🔎 Analizar residuo", variant="primary")
    boton.click(fn=clasificar, inputs=entrada, outputs=salida)
    gr.Markdown("""---
**Alcance del prototipo:** reconoce cuatro categorías: papel, cartón, vidrio y plástico. La recomendación es orientativa y no sustituye las normas locales de gestión de residuos.""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
