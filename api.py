from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64

app_to_flask = Flask(__name__)

# 🔥 PARCHE PARA ERRORES DE KERAS (quantization_config)
# 🔥 PARCHE GLOBAL para quantization_config
import keras.src.saving.serialization_lib as _serial_lib

_original_deserialize = _serial_lib.deserialize_keras_object

def _patched_deserialize(config, *args, **kwargs):
    # Limpiar quantization_config de capas Dense antes de deserializar
    if isinstance(config, dict):
        inner = config.get("config", {})
        if isinstance(inner, dict):
            inner.pop("quantization_config", None)
    return _original_deserialize(config, *args, **kwargs)

_serial_lib.deserialize_keras_object = _patched_deserialize

# ----------- CARGA DE MODELOS -----------
clasificador1 = tf.keras.models.load_model(
    "animalclassification.keras",
    compile=False,
    safe_mode=False
)

clasificador2 = tf.keras.models.load_model(
    "animalclassificationpreentrenado.keras",
    compile=False,
    safe_mode=False
)

# ----------- MAPA DE CLASES -----------
labels_map = {
    0: "oso",
    1: "cuervo",
    2: "elefante",
    3: "rata"
}

# ----------- FUNCIÓN DE PREDICCIÓN -----------
def predecir_modelos(img_array):

    pred1 = clasificador1.predict(img_array, verbose=0)[0]
    pred2 = clasificador2.predict(img_array, verbose=0)[0]

    class1 = int(np.argmax(pred1))
    class2 = int(np.argmax(pred2))

    conf1 = float(np.max(pred1))
    conf2 = float(np.max(pred2))

    return {
        "modelo1": {
            "clase": class1,
            "animal": labels_map.get(class1, "Desconocido"),
            "confianza": conf1,
            "probabilidades": pred1.tolist()
        },
        "modelo2": {
            "clase": class2,
            "animal": labels_map.get(class2, "Desconocido"),
            "confianza": conf2,
            "probabilidades": pred2.tolist()
        },
        "raw1": pred1,
        "raw2": pred2
    }

# ----------- FUNCIÓN PARA GRAFICAR -----------
def generar_grafica(pred1, pred2):

    clases = list(labels_map.values())
    x = np.arange(len(clases))

    fig, ax = plt.subplots()

    ax.bar(x - 0.2, pred1, width=0.4, label="Modelo 1", color="#7c6fc9")
    ax.bar(x + 0.2, pred2, width=0.4, label="Modelo 2", color="#e879a0")

    ax.set_xticks(x)
    ax.set_xticklabels(clases)
    ax.set_title("Comparación de Modelos")
    ax.set_ylabel("Probabilidad")
    ax.legend()

    buffer = io.BytesIO()
    fig.patch.set_facecolor('#f5f4f9')
    ax.set_facecolor('#eeecf6')
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    grafica = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return grafica

# ----------- RUTA HOME -----------
@app_to_flask.route("/")
def home():
    return render_template("index.html")

# ----------- PREDICCIÓN -----------
@app_to_flask.route("/predict", methods=["POST"])
def predict():

    if 'file' not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400

    file = request.files['file']

    try:
        from keras.utils import load_img, img_to_array
        import io

        img_bytes = io.BytesIO(file.read())
        img = load_img(img_bytes, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        resultados = predecir_modelos(img_array)
        grafica = generar_grafica(resultados["raw1"], resultados["raw2"])

        return jsonify({
            "modelo1": resultados["modelo1"],
            "modelo2": resultados["modelo2"],
            "grafica": grafica
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app_to_flask.run(debug=True)