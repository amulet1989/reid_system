import streamlit as st
import requests
import base64
import time
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

# URL interna del contenedor API de FastAPI
API_URL = "http://api:8000/api/v1"

st.set_page_config(page_title="Retail Re-ID Demo", layout="wide")
st.title("🛒 Sistema SOTA de Re-Identificación de Productos")

# --- BARRA LATERAL DE ADMINISTRACIÓN ---
with st.sidebar:
    st.header("⚙️ Administración")
    st.info("Actualiza la base de datos vectorial si agregaste nuevos productos al CSV o imágenes a la carpeta /data/Catalogo.")
    
    if st.button("🔄 Sincronizar Catálogo"):
        with st.spinner("Enviando orden a la GPU..."):
            try:
                res = requests.post(f"{API_URL}/catalog/sync")
                res.raise_for_status()
                sync_task_id = res.json()["task_id"]
            except Exception as e:
                st.error(f"Error conectando con la API: {e}")
                sync_task_id = None
        
        if sync_task_id:
            status_placeholder = st.empty()
            while True:
                res_status = requests.get(f"{API_URL}/results/{sync_task_id}").json()
                estado = res_status["status"]
                
                if estado == "PENDING":
                    status_placeholder.info("⏳ Esperando turno en la cola...")
                elif estado == "PROCESSING":
                    status_placeholder.warning("⚙️ Ingestando imágenes y extrayendo tensores...")
                elif estado == "SUCCESS":
                    status_placeholder.success("✅ ¡Catálogo sincronizado exitosamente!")
                    break
                elif estado == "FAILED":
                    status_placeholder.error(f"❌ Error en la sincronización: {res_status.get('error')}")
                    break
                
                time.sleep(1) # Consultamos cada 1 segundo para no saturar
# ---------------------------------------

col1, col2 = st.columns([1.2, 1]) # Le damos un poco más de ancho a la columna de la imagen

with col1:
    st.header("1. Imagen de Consulta")
    uploaded_file = st.file_uploader("Sube una foto del producto o estante", type=["jpg", "jpeg", "png"])
    
    bboxes_to_send = []
    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_width, img_height = image.size
        
        # Pestañas para elegir el método de ingreso de BBoxes
        tab_draw, tab_manual = st.tabs(["🖌️ Dibujar BBox", "⌨️ Ingreso Manual"])
        
        with tab_draw:
            st.info("Dibuja uno o varios rectángulos sobre los productos. Si no dibujas nada, se analizará toda la imagen.")
            
            # Calculamos el escalado para que el canvas quepa en la columna (ej. max 600px de ancho)
            canvas_display_width = 600
            scale_factor = canvas_display_width / img_width
            canvas_display_height = int(img_height * scale_factor)

            # Lienzo interactivo
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.2)",  # Verde semitransparente
                stroke_width=2,
                stroke_color="#00FF00",
                background_image=image,
                update_streamlit=True,
                height=canvas_display_height,
                width=canvas_display_width,
                drawing_mode="rect",
                key="canvas",
            )

            # Extraer coordenadas del dibujo y escalarlas al tamaño original de la foto
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    if obj["type"] == "rect":
                        x1 = int(obj["left"] / scale_factor)
                        y1 = int(obj["top"] / scale_factor)
                        # Drawable canvas usa scaleX y scaleY al arrastrar
                        w = int((obj["width"] * obj["scaleX"]) / scale_factor)
                        h = int((obj["height"] * obj["scaleY"]) / scale_factor)
                        bboxes_to_send.append([x1, y1, x1 + w, y1 + h])

        with tab_manual:
            # Corregido el warning: usamos use_container_width=True
            st.image(image, caption="Imagen original", use_column_width=True)
            st.info("Ingresa un BBox por línea con el formato: x1, y1, x2, y2")
            
            default_bbox = f"0, 0, {img_width}, {img_height}"
            bbox_input = st.text_area("Coordenadas:", value=default_bbox, height=100)
            
            # Botón para forzar la lectura del texto en lugar del dibujo
            if st.button("Usar coordenadas manuales"):
                bboxes_to_send = []
                for line in bbox_input.split('\n'):
                    if line.strip():
                        try:
                            coords = [int(x.strip()) for x in line.split(',')]
                            if len(coords) == 4:
                                bboxes_to_send.append(coords)
                        except Exception:
                            st.error(f"Línea inválida: {line}")

        # Fallback: Si no dibujaron ni ingresaron nada válido, tomar toda la foto
        if not bboxes_to_send:
            bboxes_to_send = [[0, 0, img_width, img_height]]

with col2:
    st.header("2. Resultados")
    if uploaded_file is not None:
        st.write(f"📦 **Recortes a procesar:** {len(bboxes_to_send)}")
        
        if st.button("🔍 Identificar Productos", type="primary", use_container_width=True):
            # 1. Preparar la imagen en Base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            payload = {"image_b64": img_b64, "bboxes": bboxes_to_send}
            
            # 2. Enviar a FastAPI
            with st.spinner("Enviando trabajos a la cola..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    response.raise_for_status()
                    task_id = response.json()["task_id"]
                    st.success(f"Ticket generado: `{task_id}`")
                except Exception as e:
                    st.error(f"Error conectando con la API: {e}")
                    st.stop()

            # 3. Polling (Consultar estado a Redis a través de FastAPI)
            status_placeholder = st.empty()
            while True:
                res = requests.get(f"{API_URL}/results/{task_id}").json()
                estado = res["status"]
                
                if estado == "PENDING":
                    status_placeholder.info("⏳ En cola de espera...")
                elif estado == "PROCESSING":
                    status_placeholder.warning("⚙️ Procesando recortes en RTX 3080 Ti...")
                elif estado == "SUCCESS":
                    status_placeholder.empty()
                    st.success("✅ Procesamiento completado")
                    
                    detections = res["result"].get("detections", [])
                    for det in detections:
                        st.markdown(f"### BBox {det['bbox_index'] + 1}: `{det['bbox_coords']}`")
                        st.json(det["prediction"])
                    break
                elif estado == "FAILED":
                    status_placeholder.error(f"❌ Error: {res.get('error')}")
                    break
                    
                time.sleep(0.5)