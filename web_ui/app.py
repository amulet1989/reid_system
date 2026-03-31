import streamlit as st
import requests
import base64
import time
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
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

# --- CREACIÓN DE PESTAÑAS PARA SEPARAR FLUJOS ---
tab_image, tab_video = st.tabs(["🖼️ Análisis por Imagen", "📹 Auditoría de Góndola (Video)"])

# ==========================================================
# FLUJO 1: ANÁLISIS POR IMAGEN (Tu código original)
# ==========================================================
with tab_image:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.header("1. Imagen de Consulta")
        uploaded_file = st.file_uploader("Sube una foto del producto o estante", type=["jpg", "jpeg", "png"])
        
        bboxes_to_send = []
        image = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            # --- PARCHE DE ROTACIÓN EXIF ---
            image = ImageOps.exif_transpose(image) # ¡VITAL! Hornear rotación física
            # -------------------------------
            img_width, img_height = image.size
            
            tab_draw, tab_manual = st.tabs(["🖌️ Dibujar BBox", "⌨️ Ingreso Manual"])
            
            with tab_draw:
                st.info("Dibuja uno o varios rectángulos sobre los productos. Si no dibujas nada, se analizará toda la imagen.")
                canvas_display_width = 600
                scale_factor = canvas_display_width / img_width
                canvas_display_height = int(img_height * scale_factor)

                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#00FF00",
                    background_image=image,
                    update_streamlit=True,
                    height=canvas_display_height,
                    width=canvas_display_width,
                    drawing_mode="rect",
                    key="canvas",
                )

                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    for obj in objects:
                        if obj["type"] == "rect":
                            x1 = int(obj["left"] / scale_factor)
                            y1 = int(obj["top"] / scale_factor)
                            w = int((obj["width"] * obj["scaleX"]) / scale_factor)
                            h = int((obj["height"] * obj["scaleY"]) / scale_factor)
                            bboxes_to_send.append([x1, y1, x1 + w, y1 + h])

            with tab_manual:
                st.image(image, caption="Imagen original", use_column_width=True)
                st.info("Ingresa un BBox por línea con el formato: x1, y1, x2, y2")
                
                default_bbox = f"0, 0, {img_width}, {img_height}"
                bbox_input = st.text_area("Coordenadas:", value=default_bbox, height=100)
                
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

            if not bboxes_to_send:
                bboxes_to_send = [[0, 0, img_width, img_height]]

    with col2:
        st.header("2. Resultados")
        if uploaded_file is not None:
            st.write(f"📦 **Recortes a procesar:** {len(bboxes_to_send)}")
            
            if st.button("🔍 Identificar Productos", type="primary"):
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=95)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                payload = {"image_b64": img_b64, "bboxes": bboxes_to_send}
                
                with st.spinner("Enviando trabajos a la cola..."):
                    try:
                        response = requests.post(f"{API_URL}/predict", json=payload)
                        response.raise_for_status()
                        task_id = response.json()["task_id"]
                        st.success(f"Ticket generado: `{task_id}`")
                    except Exception as e:
                        st.error(f"Error conectando con la API: {e}")
                        st.stop()

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
                            
                            img_path = det["prediction"].get("image_path")
                            if img_path:
                                with st.expander("🔬 Ver Validación Visual del Recorte"):
                                    with st.spinner("Generando visualización del recorte..."):
                                        try:
                                            res_img = requests.get(f"{API_URL}/image", params={"path": img_path})
                                            if res_img.status_code == 200:
                                                ref_image_pil = Image.open(io.BytesIO(res_img.content))
                                                x1, y1, x2, y2 = det['bbox_coords']
                                                img_crop = image.crop((x1, y1, x2, y2))

                                                st.markdown("**Izquierda:** Tu recorte exacto. **Derecha:** Referencia oficial.")
                                                col_c, col_r = st.columns(2)
                                                with col_c:
                                                    st.image(img_crop, caption="Crop enviado a la IA", use_column_width=True)
                                                with col_r:
                                                    st.image(ref_image_pil, caption="Referencia del Catálogo", use_column_width=True)
                                            else:
                                                st.error(f"Error {res_img.status_code}: Imagen de catálogo no encontrada.")
                                        except Exception as e:
                                            st.error(f"No se pudo generar la visualización: {e}")
                        break
                    elif estado == "FAILED":
                        status_placeholder.error(f"❌ Error: {res.get('error')}")
                        break
                        
                    time.sleep(0.5)


# ==========================================================
# FLUJO 2: AUDITORÍA POR VIDEO (NUEVO)
# ==========================================================
with tab_video:
    st.header("Auditoría de Góndolas (Patrón Serpiente)")
    st.markdown("Sube un video barriendo los estantes. El sistema extraerá cada producto, evitará conteos duplicados, y generará tu reporte de stock.")
    
    uploaded_video = st.file_uploader("Sube el video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🚀 Iniciar Conteo de Stock", type="primary"):
            with st.spinner("Subiendo video al servidor..."):
                try:
                    # Enviar video al endpoint de la API
                    files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                    response = requests.post(f"{API_URL}/video/analyze", files=files)
                    response.raise_for_status()
                    
                    video_task_id = response.json()["task_id"]
                    st.success(f"Video encolado exitosamente. Ticket ID: `{video_task_id}`")
                except Exception as e:
                    st.error(f"Error al subir el video: {e}")
                    st.stop()

            # Polling del procesamiento de video
            status_placeholder_vid = st.empty()
            while True:
                try:
                    res_vid = requests.get(f"{API_URL}/results/{video_task_id}").json()
                    estado_vid = res_vid["status"]
                    
                    if estado_vid == "PENDING":
                        status_placeholder_vid.info("⏳ Esperando un Worker de video disponible...")
                    elif estado_vid == "PROCESSING":
                        msg = res_vid.get("message", "⚙️ Extrayendo y trackeando productos frame por frame...")
                        status_placeholder_vid.warning(msg)
                    elif estado_vid == "SUCCESS":
                        status_placeholder_vid.empty()
                        st.success("✅ Análisis de video completado.")
                        
                        resultado_video = res_vid["result"]
                        
                        # Mostrar métrica general
                        total_items = resultado_video.get("total_unique_items_detected", 0)
                        st.metric("Total de Productos Detectados", total_items)
                        
                        # Renderizar Inventario Consolidado
                        st.subheader("📊 Reporte de Stock Real")
                        inventario = resultado_video.get("stock_inventory", {})
                        
                        if inventario:
                            # 'inventario' ahora es una lista de diccionarios con SKU, Producto, Verificado, Cantidad
                            df_stock = pd.DataFrame(inventario)
                            
                            # Ordenamos para que los que tienen más cantidad salgan arriba
                            df_stock = df_stock.sort_values(by="Cantidad", ascending=False).reset_index(drop=True)
                            
                            # Pintamos la tabla
                            st.dataframe(df_stock)
                        else:
                            st.warning("⚠️ No se identificó con seguridad geométrica ningún producto de tu catálogo.")
                        
                        break
                    elif estado_vid == "FAILED":
                        status_placeholder_vid.error(f"❌ Fallo en el análisis: {res_vid.get('error')}")
                        break
                        
                    time.sleep(2) # Polling cada 2 segundos (el video tarda más que una imagen)
                    
                except Exception as e:
                    status_placeholder_vid.error(f"Error consultando resultados: {e}")
                    break