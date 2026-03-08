import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import io
import streamlit as st
# ── INICIALIZAR SESSION STATE ─────────────────────────────────────
if "resultados_cart" not in st.session_state:
    st.session_state.resultados_cart = None
if "informe_cart" not in st.session_state:
    st.session_state.informe_cart = None
if "resultados_enc" not in st.session_state:
    st.session_state.resultados_enc = None
if "informe_enc" not in st.session_state:
    st.session_state.informe_enc = None
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from umap import UMAP
from mistralai import Mistral

st.set_page_config(page_title="Análisis Semántico", layout="wide")
st.title("🔍 Análisis Semántico de Encuestas y Cartografía Social")

client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

modelo = cargar_modelo()

def encontrar_clusters_optimos(vectores, min_k=2, max_k=8):
    max_k = min(max_k, len(vectores) - 1)
    if max_k < 2:
        return 2
    scores = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectores)
        scores[k] = silhouette_score(vectores, labels)
    return max(scores, key=scores.get)

def etiquetar_grupos_ia(frases_por_grupo, contexto):
    etiquetas = {}
    for grupo, frases in frases_por_grupo.items():
        muestra = "\n".join(f"- {f}" for f in frases[:5])
        prompt = f"""Eres un experto en análisis social y organizacional.
Analiza estas frases de un grupo semántico en el contexto de {contexto}:
{muestra}
Genera UN título descriptivo de máximo 6 palabras que capture el tema central.
Responde SOLO con el título, sin explicaciones."""
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20
            )
            etiquetas[grupo] = resp.choices[0].message.content.strip()
        except:
            etiquetas[grupo] = f"Grupo {grupo + 1}"
    return etiquetas

def asociar_lineas_inversion(frase, lineas, contexto):
    lineas_str = "\n".join(f"- {l}" for l in lineas)
    prompt = f"""Eres un experto en planificación territorial y desarrollo comunitario.
Analiza esta frase del contexto de {contexto}:
"{frase}"

Líneas de inversión disponibles:
{lineas_str}

¿A cuáles líneas de inversión corresponde esta frase? Puede ser una o varias.
Responde SOLO con los nombres exactos de las líneas separados por coma, sin explicaciones."""
    try:
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    except:
        return "No determinado"

def categorizar_hallazgos(frases, contexto):
    texto = "\n".join(f"- {f}" for f in frases[:30])
    prompt = f"""Eres un experto en diagnóstico social y organizacional.
Analiza estas frases del contexto de {contexto}:
{texto}

Clasifica los hallazgos en:
- PROBLEMAS: situaciones negativas
- POTENCIALIDADES: recursos o aspectos positivos
- PROPUESTAS: sugerencias planteadas por participantes
- ALERTAS: situaciones urgentes

Formato exacto:
PROBLEMAS:
- [hallazgo]
POTENCIALIDADES:
- [hallazgo]
PROPUESTAS:
- [hallazgo]
ALERTAS:
- [hallazgo]

Máximo 3 puntos por categoría. Si no hay, escribe "No identificados"."""
    try:
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def generar_informe_ia(resumen_datos, contexto):
    prompt = f"""Eres un consultor experto en {contexto}.
Basándote en este resumen de análisis semántico participativo:
{resumen_datos}

Genera un informe ejecutivo profesional con:
1. DIAGNÓSTICO GENERAL (2-3 párrafos)
2. HALLAZGOS PRINCIPALES (máximo 5, por relevancia)
3. RECOMENDACIONES ESTRATÉGICAS (máximo 5, priorizadas)
4. ACCIONES INMEDIATAS (máximo 3 para los próximos 30 días)
5. ALERTAS (atención urgente)
6. NOTA SOBRE SUBREGISTRO

Lenguaje profesional, directo y orientado a toma de decisiones."""
    try:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def detectar_subregistro(df, cols_texto=None):
    alertas = []
    total = len(df)
    if total < 10:
        alertas.append(f"⚠️ Muestra pequeña: {total} registros. Resultados pueden no ser representativos.")
    if cols_texto:
        for col in cols_texto:
            if col in df.columns:
                vacias = df[col].isna().sum() + (df[col].astype(str) == "").sum()
                pct = round(vacias / total * 100, 1)
                if pct > 20:
                    alertas.append(f"⚠️ Columna '{col}': {pct}% de celdas vacías.")
        longitudes = df[cols_texto[0]].dropna().astype(str).str.len()
        if longitudes.mean() < 20:
            alertas.append("⚠️ Respuestas muy cortas en promedio. Posible baja profundidad.")
    if "cluster" in df.columns:
        dist = df["cluster"].value_counts(normalize=True)
        if dist.min() < 0.05:
            alertas.append("⚠️ Hay grupos con menos del 5% de respuestas. Posible subrepresentación.")
    if not alertas:
        alertas.append("✅ No se detectaron señales evidentes de subregistro.")
    return alertas

# ════════════════════════════════════════════════════════════════
# SELECTOR DE MODO
# ════════════════════════════════════════════════════════════════
modo = st.sidebar.radio("Selecciona el tipo de análisis:",
                        ["📋 Encuesta", "🗺️ Cartografía Social"])

# ════════════════════════════════════════════════════════════════
# MÓDULO 1 — ENCUESTA
# ════════════════════════════════════════════════════════════════
if modo == "📋 Encuesta":
    st.header("Módulo de Encuesta")
    if "resultados_encuesta" not in st.session_state:
        st.session_state.resultados_encuesta = None
    if "informe_encuesta" not in st.session_state:
        st.session_state.informe_encuesta = None
    if "resultados_encuesta" not in st.session_state:
        st.session_state.resultados_encuesta = None        
    if "informe_encuesta" not in st.session_state:
        st.session_state.informe_encuesta = None
    st.write("Sube tu Excel con columnas Likert, respuestas abiertas y opcionalmente datos sociodemográficos.")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

    if archivo:
        df = pd.read_excel(archivo)
        st.subheader("Vista previa de tus datos")
        st.dataframe(df.head())

        columnas = list(df.columns)
        col1, col2, col3 = st.columns(3)
        with col1:
            cols_likert = st.multiselect("Columnas Likert (numéricas):", columnas)
        with col2:
            cols_texto = st.multiselect("Columnas de texto abierto:", columnas)
        with col3:
            cols_socio = st.multiselect("Columnas sociodemográficas (opcional):", columnas)

        contexto = st.selectbox("Contexto del análisis:",
            ["consultoría y diagnóstico organizacional",
             "bienestar social y desarrollo comunitario",
             "diagnóstico territorial participativo"])

        if cols_texto and st.button("▶ Analizar"):
            with st.spinner("Procesando con modelo semántico..."):
                df["texto_unido"] = df[cols_texto].fillna("").apply(
                    lambda row: " ".join(str(v) for v in row if v != ""), axis=1)
                textos = df["texto_unido"].tolist()
                vectores = modelo.encode(textos, show_progress_bar=False)

                n_opt = encontrar_clusters_optimos(vectores)
                st.info(f"📊 Número óptimo de grupos detectado: **{n_opt}**")

                kmeans = KMeans(n_clusters=n_opt, random_state=42, n_init=10)
                df["cluster_num"] = kmeans.fit_predict(vectores)

                centroides = kmeans.cluster_centers_
                sims = cosine_similarity(vectores, centroides)
                df["peso_semantico"] = sims.max(axis=1).round(3)
                df["peso_ponderado"] = (df["peso_semantico"] / df.groupby("cluster_num")["peso_semantico"].transform("sum")).round(3)

                frases_por_grupo = {}
                for g in range(n_opt):
                    frases_por_grupo[g] = df[df["cluster_num"] == g]["texto_unido"].tolist()

            with st.spinner("Etiquetando grupos con IA..."):
                etiquetas = etiquetar_grupos_ia(frases_por_grupo, contexto)
                df["cluster"] = df["cluster_num"].map(etiquetas)
                if len(vectores) >= 10:
                    umap_model = UMAP(n_components=2, random_state=42)
                    coords = umap_model.fit_transform(vectores)
                    df["x"] = coords[:, 0]
                    df["y"] = coords[:, 1]
                else:
                    df["x"] = np.random.rand(len(vectores))
                    df["y"] = np.random.rand(len(vectores))

            st.session_state.resultados_encuesta = df
            st.session_state.informe_encuesta = None
            st.success("Análisis completado.")
            st.session_state.resultados_enc = df
            st.session_state.resultados_encuesta = df
            st.subheader("🔎 Detección de subregistro")
            for a in detectar_subregistro(df, cols_texto):
                st.write(a)

            grupos_disponibles = sorted(df["cluster"].unique())
            grupo_sel = st.multiselect("Filtrar por grupo:", grupos_disponibles, default=grupos_disponibles)
            df_filtrado = df[df["cluster"].isin(grupo_sel)]

            st.subheader("Mapa semántico de respuestas")
            fig = px.scatter(df_filtrado, x="x", y="y", color="cluster",
                           hover_data=cols_texto + ["peso_semantico"],
                           title="Agrupación semántica de respuestas")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Distribución de grupos")
            conteo = df_filtrado["cluster"].value_counts().reset_index()
            conteo.columns = ["Grupo", "Cantidad"]
            conteo["Porcentaje"] = (conteo["Cantidad"] / conteo["Cantidad"].sum() * 100).round(1)
            col1, col2 = st.columns(2)
            with col1:
                fig_t = px.pie(conteo, names="Grupo", values="Cantidad", title="Porcentaje por grupo", hole=0.3)
                st.plotly_chart(fig_t, use_container_width=True)
            with col2:
                fig_b = px.bar(conteo, x="Grupo", y="Porcentaje", text="Porcentaje",
                              color="Grupo", title="Porcentaje por grupo (%)")
                fig_b.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_b, use_container_width=True)

            if cols_likert:
                st.subheader("Promedio Likert por grupo")
                resumen_lk = df_filtrado.groupby("cluster")[cols_likert].mean().round(2)
                fig_lk = px.imshow(resumen_lk, text_auto=True, color_continuous_scale="RdYlGn",
                                  title="Mapa de calor: Likert por grupo")
                st.plotly_chart(fig_lk, use_container_width=True)

            if cols_socio:
                st.subheader("👥 Participación sociodemográfica")
                for col_s in cols_socio:
                    if col_s in df_filtrado.columns:
                        conteo_s = df_filtrado[col_s].value_counts().reset_index()
                        conteo_s.columns = [col_s, "Cantidad"]
                        fig_s = px.bar(conteo_s, x=col_s, y="Cantidad", color=col_s,
                                      title=f"Participación por {col_s}")
                        st.plotly_chart(fig_s, use_container_width=True)
                        cruce = df_filtrado.groupby(["cluster", col_s]).size().reset_index(name="n")
                        fig_cruce = px.bar(cruce, x="cluster", y="n", color=col_s, barmode="group",
                                          title=f"Grupos semánticos por {col_s}")
                        st.plotly_chart(fig_cruce, use_container_width=True)

            st.subheader("🗂️ Categorización de hallazgos")
            with st.spinner("Categorizando con IA..."):
                cat = categorizar_hallazgos(df_filtrado["texto_unido"].tolist(), contexto)
            st.markdown(cat)

            st.subheader("📋 Resumen ejecutivo por grupo")
            resumen_exec = df_filtrado.groupby("cluster").agg(
                Cantidad=("cluster", "count"),
                Cohesion=("peso_semantico", "mean")
            ).round(3).reset_index()
            resumen_exec["Porcentaje"] = (resumen_exec["Cantidad"] / resumen_exec["Cantidad"].sum() * 100).round(1).astype(str) + "%"
            resumen_exec.columns = ["Grupo", "Respuestas", "Cohesión semántica", "Porcentaje"]
            st.dataframe(resumen_exec, use_container_width=True)

            st.subheader("⚖️ Frecuencia simple vs. ponderada")
            pond = df_filtrado.groupby("cluster").agg(
                Frecuencia_simple=("cluster", "count"),
                Peso_ponderado=("peso_ponderado", "sum")
            ).round(3).reset_index()
            fig_pond = px.bar(pond, x="cluster", y=["Frecuencia_simple", "Peso_ponderado"],
                             barmode="group", title="Frecuencia simple vs. ponderada")
            st.plotly_chart(fig_pond, use_container_width=True)

            st.subheader("💬 Frases más representativas por grupo")
            for grupo in grupos_disponibles:
                if grupo in grupo_sel:
                    with st.expander(f"📌 {grupo}"):
                        top = df_filtrado[df_filtrado["cluster"] == grupo].nlargest(3, "peso_semantico")
                        for _, row in top.iterrows():
                            for col in cols_texto:
                                st.write(f"• {row[col]}")

            st.subheader("📄 Informe ejecutivo generado por IA")
            resumen_para_ia = f"Contexto: {contexto}\nTotal participantes: {len(df_filtrado)}\nGrupos: {n_opt}\n"
            for grupo in grupos_disponibles:
                if grupo in grupo_sel:
                    top = df_filtrado[df_filtrado["cluster"] == grupo].nlargest(2, "peso_semantico")
                    resumen_para_ia += f"\n{grupo}:\n"
                    for _, row in top.iterrows():
                        resumen_para_ia += f"  - {row['texto_unido']}\n"
            with st.spinner("Generando informe con IA..."):
                informe = generar_informe_ia(resumen_para_ia, contexto)
                st.session_state.informe_enc = informa
            st.markdown(informe)

            st.subheader("⬇ Descargas")
            col1, col2 = st.columns(2)
            with col1:
                buffer_enc = io.BytesIO()
                df_filtrado.to_excel(buffer_enc, index=False, engine="openpyxl")
                buffer_enc.seek(0)
                st.download_button(
                    "⬇ Descargar datos Excel",
                    data=buffer_enc,
                    file_name="resultados_encuesta.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descarga_encuesta"
                )
            with col2:
                st.download_button(
                    "⬇ Descargar informe TXT",
                    data=informe.encode("utf-8"),
                    file_name="informe_ejecutivo.txt",
                    key="descarga_informe_encuesta"
                )

# ════════════════════════════════════════════════════════════════
# MÓDULO 2 — CARTOGRAFÍA SOCIAL
# ════════════════════════════════════════════════════════════════
elif modo == "🗺️ Cartografía Social":
    st.header("Módulo de Cartografía Social")
    st.write("Sube tu Excel con columnas: año, semestre, municipio, vereda, participantes, componentes y lineas de inversion.")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

    if archivo:
        archivo_hash = str(hash(archivo.read()))
        archivo.seek(0)
        df = pd.read_excel(archivo)
        df.columns = [c.strip().lower() for c in df.columns]
        st.subheader("Vista previa de tus datos")
        st.dataframe(df.head())

        columnas = list(df.columns)

        # ── COLUMNAS FIJAS ────────────────────────────────────────
        # ── LEER CONFIGURACIÓN DESDE FILA 2 ──────────────────────
        fila_config = df.iloc[0]  # fila 2 del Excel = fila 0 del dataframe
        df = df.iloc[1:].reset_index(drop=True)  # eliminar fila de configuración

        cols_meta = [c for c in columnas if str(fila_config[c]).strip().lower() == "meta"]
        col_lineas_list = [c for c in columnas if str(fila_config[c]).strip().lower() == "lineas"]
        col_lineas = col_lineas_list[0] if col_lineas_list else "lineas de inversion"
        cols_componentes_default = [c for c in columnas if str(fila_config[c]).strip().lower() == "componente"]
        cols_componentes = st.multiselect("Selecciona los componentes a analizar:",
                                          cols_componentes_default,
                                          default=cols_componentes_default)

        contexto = st.selectbox("Contexto del análisis:",
            ["consultoría y diagnóstico organizacional",
             "bienestar social y desarrollo comunitario",
             "diagnóstico territorial participativo"])

        cache_key = f"{archivo_hash}_{','.join(sorted(cols_componentes))}_{contexto}"
        if cols_componentes and st.button("▶ Analizar"):
            usar_cache = (
                st.session_state.get("cache_key_saved") == cache_key and
                st.session_state.get("cache_result_cart") is not None
            )
            if usar_cache:
                st.info("✅ Mismos datos detectados — usando resultados anteriores.")
                df_result = st.session_state.cache_result_cart
                total_participantes = pd.to_numeric(df.get("participantes", pd.Series([0])), errors="coerce").sum() if "participantes" in df.columns else 0
            else:

                # ── PARTICIPANTES ─────────────────────────────────────
                total_participantes = 0
                if "participantes" in df.columns:
                    total_participantes = pd.to_numeric(df["participantes"], errors="coerce").sum()
                    st.metric("👥 Total participantes", int(total_participantes))

                # ── PARTICIPACIÓN TERRITORIAL ─────────────────────────
                if "municipio" in df.columns and "vereda" in df.columns:
                    st.subheader("🗺️ Participación territorial")
                    part_muni = df.groupby("municipio")["participantes"].apply(
                        lambda x: pd.to_numeric(x, errors="coerce").sum()).reset_index()
                    part_muni.columns = ["Municipio", "Participantes"]
                    fig_muni = px.bar(part_muni, x="Municipio", y="Participantes",
                                     color="Municipio", title="Participantes por municipio")
                    st.plotly_chart(fig_muni, use_container_width=True)

                    part_vereda = df.groupby(["municipio", "vereda"])["participantes"].apply(
                        lambda x: pd.to_numeric(x, errors="coerce").sum()).reset_index()
                    part_vereda.columns = ["Municipio", "Vereda", "Participantes"]
                    fig_vereda = px.bar(part_vereda, x="Vereda", y="Participantes",
                                       color="Municipio", title="Participantes por vereda")
                    st.plotly_chart(fig_vereda, use_container_width=True)

                with st.spinner("Fragmentando frases y procesando con el modelo semántico..."):

                    # ── FRAGMENTAR FRASES ─────────────────────────────
                    registros = []
                    for _, fila in df.iterrows():
                        lineas_celda = str(fila.get(col_lineas, "")).split(",") if col_lineas in df.columns else []
                        lineas_celda = [l.strip() for l in lineas_celda if l.strip()]

                        for comp in cols_componentes:
                            celda = str(fila.get(comp, ""))
                            if celda and celda != "nan":
                                frases = [f.strip() for f in celda.split(".") if len(f.strip()) > 5]
                                for frase in frases:
                                    registro = {
                                        "componente": comp,
                                        "frase": frase,
                                        "lineas_disponibles": lineas_celda
                                    }
                                    for col_m in cols_meta:
                                        registro[col_m] = fila.get(col_m, "")
                                    registros.append(registro)

                    df_frases = pd.DataFrame(registros)
                    st.write(f"Total de frases extraídas: {len(df_frases)}")

                    textos_lista = df_frases["frase"].tolist()
                    vectores = np.array(modelo.encode(textos_lista, show_progress_bar=False, batch_size=16))

                    # ── CLUSTERING POR COMPONENTE ─────────────────────
                    resultados = []
                    for comp in cols_componentes:
                        mask = df_frases["componente"] == comp
                        vecs_comp = vectores[mask]
                        subset = df_frases[mask].reset_index(drop=True)

                        if len(subset) >= 4:
                            n_opt = encontrar_clusters_optimos(vecs_comp, max_k=min(6, len(subset)-1))
                        else:
                            n_opt = min(2, len(subset))

                        if n_opt >= 2:
                            km = KMeans(n_clusters=n_opt, random_state=42, n_init=10)
                            clusters = km.fit_predict(vecs_comp)
                            sims = cosine_similarity(vecs_comp, km.cluster_centers_)
                            pesos = sims.max(axis=1).round(3)
                        else:
                            clusters = [0] * len(subset)
                            pesos = [1.0] * len(subset)

                        for i, (_, row) in enumerate(subset.iterrows()):
                            resultado = {
                                "componente": comp,
                                "frase": row["frase"],
                                "grupo_num": clusters[i],
                                "peso_semantico": pesos[i],
                                "lineas_disponibles": row["lineas_disponibles"]
                            }
                            for col_m in cols_meta:
                                resultado[col_m] = row.get(col_m, "")
                            resultados.append(resultado)

                    df_result = pd.DataFrame(resultados)

                letras = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
                for comp in cols_componentes:
                    mask = df_result["componente"] == comp
                    grupos_unicos = sorted(df_result[mask]["grupo_num"].unique())
                    mapa = {g: f"Grupo {letras[i]}" for i, g in enumerate(grupos_unicos)}
                    df_result.loc[mask, "grupo"] = df_result.loc[mask, "grupo_num"].map(mapa)
                with st.spinner("Asociando frases a líneas de inversión con IA..."):
                    for comp in cols_componentes:
                        mask = df_result["componente"] == comp
                        subset = df_result[mask].reset_index(drop=True)
                        if len(subset) == 0:
                            continue

                        lineas_disponibles = subset.iloc[0]["lineas_disponibles"]
                        if not lineas_disponibles:
                            df_result.loc[mask, "lineas_inversion"] = "Sin líneas definidas"
                            continue

                        frases_lote = subset["frase"].tolist()
                        lineas_str = "\n".join(f"- {l}" for l in lineas_disponibles)
                        lineas_validas_set = set(l.lower() for l in lineas_disponibles)
                        nombres_componentes = ["economico", "social", "ambiental", "gobernanza", "alianzas", "proyeccion", "governance", "económico", "proyección"]

                        for inicio in range(0, len(frases_lote), 20):
                            sublote = frases_lote[inicio:inicio+20]
                            frases_str = "\n".join(f"{i+1}. {f}" for i, f in enumerate(sublote))
                            prompt = f"""Eres un experto en desarrollo territorial con profundo conocimiento en Investigación-Acción Participativa (IAP), educación popular y etnografía crítica, en la línea de Orlando Fals Borda y Paulo Freire.

    Estás analizando frases de una cartografía social participativa en el contexto de {contexto}. Esta cartografía no solo describe el territorio — lo interpreta desde la memoria, el conflicto, el poder y la esperanza colectiva.

    Líneas de inversión disponibles (usa EXACTAMENTE estos nombres, sin modificar):
    {lineas_str}

    Para cada frase numerada, indica a cuáles líneas de inversión corresponde. Una frase puede corresponder a varias líneas.

    Frases:
    {frases_str}

    Responde SOLO en este formato exacto, una línea por frase numerada:
    1. Agua y territorio, Desarrollo rural
    2. Fortalecimiento comunitario
    (usa los nombres EXACTOS de las líneas disponibles, no inventes nombres nuevos)
    Sin explicaciones adicionales."""
                            try:
                                resp = client.chat.complete(
                                    model="mistral-small-latest",
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=1500
                                )
                                lineas_resp = resp.choices[0].message.content.strip().split("\n")
                                for i, frase in enumerate(sublote):
                                    if i < len(lineas_resp):
                                        linea_raw = lineas_resp[i].split(". ", 1)[-1].strip().rstrip(".")
                                        partes = [p.strip().rstrip(".") for p in linea_raw.split(",")]
                                        partes_normalizadas = []
                                        for parte in partes:
                                            parte_lower = parte.lower().strip()
                                            if any(cn in parte_lower for cn in nombres_componentes):
                                                continue
                                            mejor = min(lineas_disponibles,
                                                       key=lambda l: sum(c not in l.lower() for c in parte_lower))
                                            palabras_parte = set(parte_lower.split())
                                            palabras_mejor = set(mejor.lower().split())
                                            if len(palabras_parte & palabras_mejor) >= 1:
                                                partes_normalizadas.append(mejor)
                                        linea_final = ", ".join(partes_normalizadas) if partes_normalizadas else "No determinado"
                                    else:
                                        linea_final = "No determinado"
                                    idx = df_result[(df_result["componente"] == comp) & (df_result["frase"] == frase)].index
                                    if len(idx) > 0:
                                        df_result.loc[idx[0], "lineas_inversion"] = linea_final
                            except Exception as e:
                                for frase in sublote:
                                    idx = df_result[(df_result["componente"] == comp) & (df_result["frase"] == frase)].index
                                    if len(idx) > 0:
                                        df_result.loc[idx[0], "lineas_inversion"] = "No determinado"

                st.session_state.cache_result_cart = df_result
                st.session_state.cache_key_saved = cache_key
                st.success("Análisis completado.")            
            st.session_state.resultados_cart = df_result

            # ── SUBREGISTRO ───────────────────────────────────────
            st.subheader("🔎 Detección de subregistro")
            for a in detectar_subregistro(df_result):
                st.write(a)

            # ── FILTROS ───────────────────────────────────────────
            st.subheader("Filtrar resultados")
            col1, col2, col3 = st.columns(3)
            with col1:
                comp_filtro = st.multiselect("Componente:", cols_componentes, default=cols_componentes)
            with col2:
                municipios_disp = sorted(df_result["municipio"].dropna().unique())
                muni_filtro = st.multiselect("Municipio:", municipios_disp, default=municipios_disp)
            with col3:
                grupos_disp = sorted(df_result["grupo"].dropna().unique())
                grupo_filtro = st.multiselect("Grupo:", grupos_disp, default=grupos_disp)

            df_filtrado = df_result[
                (df_result["componente"].isin(comp_filtro)) &
                (df_result["municipio"].isin(muni_filtro)) &
                (df_result["grupo"].isin(grupo_filtro))
            ]

            # ── DISTRIBUCIÓN ──────────────────────────────────────
            st.subheader("📊 Distribución por componente")
            conteo_comp = df_filtrado["componente"].value_counts().reset_index()
            conteo_comp.columns = ["Componente", "Frases"]
            conteo_comp["Porcentaje"] = (conteo_comp["Frases"] / conteo_comp["Frases"].sum() * 100).round(1)
            col1, col2 = st.columns(2)
            with col1:
                fig_t = px.pie(conteo_comp, names="Componente", values="Frases",
                              title="Distribución por componente", hole=0.3)
                st.plotly_chart(fig_t, use_container_width=True)
            with col2:
                fig_b = px.bar(conteo_comp, x="Componente", y="Porcentaje", text="Porcentaje",
                              color="Componente", title="Porcentaje por componente (%)")
                fig_b.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_b, use_container_width=True)

            # ── LÍNEAS DE INVERSIÓN ───────────────────────────────
            st.subheader("💰 Distribución por línea de inversión")
            lineas_canonicas = []
            for _, fila in df_filtrado.iterrows():
                for l in fila["lineas_disponibles"]:
                    l_clean = l.strip().rstrip(".").strip()
                    if l_clean and l_clean not in lineas_canonicas:
                        lineas_canonicas.append(l_clean)

            def normalizar_linea(texto):
                texto = texto.strip().rstrip(".").strip().lower()
                if not texto or len(texto) < 3:
                    return None
                mejor_coincidencia = None
                mejor_puntaje = 0
                for canon in lineas_canonicas:
                    palabras_canon = set(canon.lower().split())
                    palabras_texto = set(texto.split())
                    coincidencias = len(palabras_canon & palabras_texto)
                    puntaje = coincidencias / max(len(palabras_canon), 1)
                    if puntaje > mejor_puntaje and puntaje >= 0.4:
                        mejor_puntaje = puntaje
                        mejor_coincidencia = canon
                return mejor_coincidencia

            todas_lineas = []
            for lineas_str in df_filtrado["lineas_inversion"]:
                for l in str(lineas_str).split(","):
                    normalizada = normalizar_linea(l)
                    if normalizada:
                        todas_lineas.append(normalizada)                                
            if todas_lineas:
                df_lineas = pd.DataFrame({"linea": todas_lineas})
                conteo_lineas = df_lineas["linea"].value_counts().reset_index()
                conteo_lineas.columns = ["Línea de inversión", "Frecuencia"]
                fig_li = px.bar(conteo_lineas, x="Línea de inversión", y="Frecuencia",
                               color="Línea de inversión", title="Frecuencia por línea de inversión")
                st.plotly_chart(fig_li, use_container_width=True)

                st.subheader("Cruce: Componente × Línea de inversión")
                cruce_data = []
                for _, row in df_filtrado.iterrows():
                    for l in str(row["lineas_inversion"]).split(","):
                        normalizada = normalizar_linea(l)
                        if normalizada:
                            cruce_data.append({"Componente": row["componente"], "Línea": normalizada})
                if cruce_data:
                    df_cruce = pd.DataFrame(cruce_data)
                    pivot = df_cruce.groupby(["Componente", "Línea"]).size().reset_index(name="n")
                    fig_cruce = px.bar(pivot, x="Componente", y="n", color="Línea",
                                      barmode="group", title="Componentes por línea de inversión")
                    st.plotly_chart(fig_cruce, use_container_width=True)

           # ── GRUPOS POR COMPONENTE ─────────────────────────────
            st.subheader("Grupos semánticos por componente")
            cols_graf = st.columns(2)
            for i, comp in enumerate(comp_filtro):
                subset_comp = df_filtrado[df_filtrado["componente"] == comp]
                conteo_g = subset_comp["grupo"].value_counts().reset_index()
                conteo_g.columns = ["Grupo", "Frecuencia"]
                frases_rep = {}
                for grupo in conteo_g["Grupo"]:
                    top_frase = subset_comp[subset_comp["grupo"] == grupo].nlargest(1, "peso_semantico")
                    if len(top_frase) > 0:
                        frase = top_frase.iloc[0]["frase"]
                        frase_corta = frase[:90] + "..." if len(frase) > 90 else frase
                        frases_rep[grupo] = frase_corta
                conteo_g["Tema"] = conteo_g["Grupo"].map(frases_rep)
                with cols_graf[i % 2]:
                    fig_g = px.bar(
                        conteo_g, x="Grupo", y="Frecuencia",
                        color="Grupo", title=f"{comp.upper()}",
                        text="Frecuencia",
                        hover_data={"Tema": True}
                    )
                    fig_g.update_traces(textposition="outside")
                    fig_g.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_g, use_container_width=True)
                    for _, row in conteo_g.iterrows():
                        st.caption(f"**{row['Grupo']}:** {row['Tema']}")
            # ── COHESIÓN ──────────────────────────────────────────
            st.subheader("Cohesión semántica por componente")
            pesos_c = df_filtrado.groupby("componente")["peso_semantico"].mean().round(3).reset_index()
            fig_p = px.bar(pesos_c, x="componente", y="peso_semantico",
                          color="peso_semantico", color_continuous_scale="Teal",
                          title="Cohesión semántica promedio")
            st.plotly_chart(fig_p, use_container_width=True)

            # ── CATEGORIZACIÓN ────────────────────────────────────
            st.subheader("🗂️ Categorización de hallazgos por componente")
            for comp in comp_filtro:
                with st.expander(f"📌 {comp}"):
                    st.markdown(cat_por_componente.get(comp, "No disponible"))

            st.subheader("🗂️ Categorización de hallazgos por vereda")
            veredas_unicas = sorted(df_filtrado["vereda"].dropna().unique())
            vereda_sel = st.selectbox("Selecciona una vereda:", veredas_unicas)
            for comp in comp_filtro:
                if vereda_sel in cat_por_vereda.get(comp, {}):
                    with st.expander(f"📌 {comp}"):
                        st.markdown(cat_por_vereda[comp][vereda_sel])

            # ── RESUMEN EJECUTIVO ─────────────────────────────────
            st.subheader("📋 Resumen ejecutivo por componente")
            resumen_exec = df_filtrado.groupby("componente").agg(
                Total_frases=("frase", "count"),
                Cohesion=("peso_semantico", "mean")
            ).round(3).reset_index()
            resumen_exec["Porcentaje"] = (resumen_exec["Total_frases"] / resumen_exec["Total_frases"].sum() * 100).round(1).astype(str) + "%"
            resumen_exec.columns = ["Componente", "Total frases", "Cohesión semántica", "Porcentaje"]
            st.dataframe(resumen_exec, use_container_width=True)

            # ── FRASES REPRESENTATIVAS ────────────────────────────
            st.subheader("💬 Frases más representativas por componente")
            for comp in comp_filtro:
               with st.expander(f"📌 {comp}"):
                    subset_comp = df_filtrado[df_filtrado["componente"] == comp]
                    top = subset_comp.nlargest(5, "peso_semantico")
                    frases_vistas = []
                    vectores_vistos = []
                    for _, row in top.iterrows():
                        frase = row["frase"].strip()
                        if vectores_vistos:
                            vec_frase = modelo.encode([frase])
                            sims = cosine_similarity(vec_frase, vectores_vistos)[0]
                            es_repetida = any(s > 0.75 for s in sims)
                        else:
                            es_repetida = False
                        if not es_repetida:
                            grupo_frase = row["grupo"]
                            subset_grupo = subset_comp[subset_comp["grupo"] == grupo_frase]
                            frecuencia = subset_grupo.shape[0]
                            peso = round(float(row['peso_semantico']), 3)
                            if peso >= 0.85:
                                relevancia = "Alta relevancia 🔴"
                            elif peso >= 0.65:
                                relevancia = "Media relevancia 🟡"
                            else:
                                relevancia = "Baja relevancia 🟢"
                            st.write(f"• {frase} (peso: {peso} — {relevancia})")
                            if frecuencia > 1:
                                st.caption(f"  💬 {frecuencia} frases con temática similar en este grupo")
                                otras = subset_grupo[subset_grupo["frase"] != frase]["frase"].head(2).tolist()
                                for otra in otras:
                                    st.caption(f"  ↳ {otra}")
                            lineas = str(row['lineas_inversion'])
                            if lineas not in ["Sin líneas definidas", "No determinado", "nan"]:
                                st.caption(f"  Líneas: {lineas}")
                            frases_vistas.append(frase)
                            vectores_vistos.append(modelo.encode([frase])[0])                            
            # ── INFORME IA ────────────────────────────────────────
            st.subheader("📄 Informe ejecutivo generado por IA")
            resumen_para_ia = f"""
Contexto: {contexto}
Total participantes: {int(total_participantes)}
Total frases analizadas: {len(df_filtrado)}
Municipios: {', '.join(muni_filtro)}
Componentes: {', '.join(comp_filtro)}
Distribución por componente:
{conteo_comp.to_string(index=False)}
Cohesión semántica:
{pesos_c.to_string(index=False)}
Frases más representativas:
"""
            for comp in comp_filtro:
                top = df_filtrado[df_filtrado["componente"] == comp].nlargest(2, "peso_semantico")
                resumen_para_ia += f"\n{comp}:\n"
                for _, row in top.iterrows():
                    resumen_para_ia += f"  - {row['frase']} (Líneas: {row['lineas_inversion']})\n"

            with st.spinner("Generando informe ejecutivo con IA..."):
                informe = generar_informe_ia(resumen_para_ia, contexto)
                st.session_state.informe_cart = informe
            st.markdown(informe)

            # ── TABLA DETALLADA ───────────────────────────────────
            st.subheader("Tabla detallada")
            comp_vista = st.selectbox("Ver frases del componente:", comp_filtro)
            cols_vista = ["municipio", "vereda", "componente", "frase", "grupo", "peso_semantico", "lineas_inversion"]
            st.dataframe(df_filtrado[df_filtrado["componente"] == comp_vista][cols_vista], use_container_width=True)

            # ── DESCARGA ──────────────────────────────────────────
            st.session_state.resultados_cart = df_filtrado

            # Pre-calcular datos para Excel (mismos que la app)
            # Categorización por componente (calculada una sola vez)
            if "cache_cat_cart" not in st.session_state or st.session_state.get("cache_key_saved") != cache_key:
                if "cache_cat_cart" not in st.session_state or st.session_state.get("cache_key_saved") != cache_key:
                cat_por_componente = {}
                cat_por_vereda = {}
                veredas_unicas = sorted(df_filtrado["vereda"].dropna().unique())
                for comp in comp_filtro:
                    frases_comp = df_filtrado[df_filtrado["componente"] == comp]["frase"].tolist()
                    cat_por_componente[comp] = categorizar_hallazgos(frases_comp, contexto)
                    cat_por_vereda[comp] = {}
                    for vereda in veredas_unicas:
                        frases_vereda = df_filtrado[
                            (df_filtrado["componente"] == comp) &
                            (df_filtrado["vereda"] == vereda)
                        ]["frase"].tolist()
                        if frases_vereda:
                            cat_por_vereda[comp][vereda] = categorizar_hallazgos(frases_vereda, contexto)
                st.session_state.cache_cat_cart = cat_por_componente
                st.session_state.cache_cat_vereda = cat_por_vereda
            else:
                cat_por_componente = st.session_state.cache_cat_cart
                cat_por_vereda = st.session_state.cache_cat_vereda

            st.subheader("⬇ Descargas")
            col1, col2 = st.columns(2)
            with col1:
                buffer_cart = io.BytesIO()
                with pd.ExcelWriter(buffer_cart, engine="openpyxl") as writer:

                    # Hoja 1 — Datos completos
                    cols_meta_export = [c for c in ["municipio", "vereda", "año", "semestre"] if c in df_filtrado.columns]
                    cols_descarga = cols_meta_export + ["componente", "frase", "grupo", "peso_semantico", "lineas_inversion"]
                    cols_disponibles = [c for c in cols_descarga if c in df_filtrado.columns]
                    df_filtrado[cols_disponibles].to_excel(writer, sheet_name="Datos completos", index=False)
                    
                    # Hoja 2 — Distribución por componente
                    conteo_comp.to_excel(writer, sheet_name="Distribucion componentes", index=False)

                    # Hoja 3 — Líneas de inversión (misma normalización que la app)
                    lineas_export = []
                    for lineas_str in df_filtrado["lineas_inversion"]:
                        for l in str(lineas_str).split(","):
                            l_clean = l.strip().rstrip(".").strip()
                            normalizada = normalizar_linea(l_clean)
                            if normalizada:
                                lineas_export.append(normalizada)
                    if lineas_export:
                        df_lineas_exp = pd.DataFrame({"linea": lineas_export})
                        conteo_lineas_exp = df_lineas_exp["linea"].value_counts().reset_index()
                        conteo_lineas_exp.columns = ["Línea de inversión", "Frecuencia"]
                        conteo_lineas_exp.to_excel(writer, sheet_name="Lineas de inversion", index=False)

                    # Hoja 4 — Cruce componente x línea (misma normalización que la app)
                    cruce_export = []
                    for _, row in df_filtrado.iterrows():
                        for l in str(row["lineas_inversion"]).split(","):
                            l_clean = l.strip().rstrip(".").strip()
                            normalizada = normalizar_linea(l_clean)
                            if normalizada:
                                cruce_export.append({"Componente": row["componente"], "Línea": normalizada})
                    if cruce_export:
                        df_cruce_exp = pd.DataFrame(cruce_export)
                        pivot_exp = df_cruce_exp.groupby(["Componente", "Línea"]).size().reset_index(name="Cantidad")
                        pivot_exp.to_excel(writer, sheet_name="Cruce componente x linea", index=False)

                    # Hoja 5 — Grupos por componente
                    grupos_export = []
                    for comp in comp_filtro:
                        subset_comp = df_filtrado[df_filtrado["componente"] == comp]
                        for grupo in subset_comp["grupo"].unique():
                            subset_grupo = subset_comp[subset_comp["grupo"] == grupo]
                            frase_rep = subset_grupo.nlargest(1, "peso_semantico").iloc[0]["frase"]
                            grupos_export.append({
                                "Componente": comp,
                                "Grupo": grupo,
                                "Frecuencia": len(subset_grupo),
                                "Frase representativa": frase_rep
                            })
                    pd.DataFrame(grupos_export).to_excel(writer, sheet_name="Grupos por componente", index=False)

                    # Hoja 6 — Cohesión semántica
                    pesos_c.to_excel(writer, sheet_name="Cohesion semantica", index=False)

                    # Hoja 7 — Resumen ejecutivo
                    resumen_exec.to_excel(writer, sheet_name="Resumen ejecutivo", index=False)

                   # Hoja 8 — Frases representativas
                    frases_rep_export = []
                    for comp in comp_filtro:
                        subset_comp = df_filtrado[df_filtrado["componente"] == comp]
                        frases_vistas_exp = []
                        vectores_vistos_exp = []
                        top = subset_comp.nlargest(5, "peso_semantico")
                        for _, row in top.iterrows():
                            frase = row["frase"].strip()
                            if vectores_vistos_exp:
                                vec = modelo.encode([frase])
                                sims = cosine_similarity(vec, vectores_vistos_exp)[0]
                                es_rep = any(s > 0.75 for s in sims)
                            else:
                                es_rep = False
                            if not es_rep:
                                peso = round(float(row["peso_semantico"]), 3)
                                relevancia = "Alta 🔴" if peso >= 0.85 else ("Media 🟡" if peso >= 0.65 else "Baja 🟢")
                                grupo_frase = row["grupo"]
                                subset_grupo = subset_comp[subset_comp["grupo"] == grupo_frase]
                                frecuencia = subset_grupo.shape[0]
                                otras = subset_grupo[subset_grupo["frase"] != frase]["frase"].head(2).tolist()
                                lineas_frase = str(row["lineas_inversion"])
                                frases_rep_export.append({
                                    "Componente": comp,
                                    "Frase": frase,
                                    "Peso": peso,
                                    "Relevancia": relevancia,
                                    "Frases similares en grupo": frecuencia,
                                    "Variaciones": " | ".join(otras),
                                    "Líneas de inversión": lineas_frase
                                })
                                frases_vistas_exp.append(frase)
                                vectores_vistos_exp.append(modelo.encode([frase])[0])
                    pd.DataFrame(frases_rep_export).to_excel(writer, sheet_name="Frases representativas", index=False)

                    # Hoja 10 — Tabla detallada
                    cols_tabla = [c for c in ["municipio", "vereda", "año", "semestre", "componente", "frase", "grupo", "peso_semantico", "lineas_inversion"] if c in df_filtrado.columns]
                    df_filtrado[cols_tabla].to_excel(writer, sheet_name="Tabla detallada", index=False)
                    
                    # Hoja 9 — Categorización por componente
                    cat_export = []
                    for comp, cat_texto in cat_por_componente.items():
                        for linea in cat_texto.split("\n"):
                            if linea.strip():
                                cat_export.append({"Vereda": "TODAS", "Componente": comp, "Categorización": linea.strip()})
                    pd.DataFrame(cat_export).to_excel(writer, sheet_name="Categorizacion componente", index=False)

                    # Hoja 10 — Categorización por vereda
                    cat_vereda_export = []
                    for comp, veredas in cat_por_vereda.items():
                        for vereda, cat_texto in veredas.items():
                            for linea in cat_texto.split("\n"):
                                if linea.strip():
                                    cat_vereda_export.append({"Vereda": vereda, "Componente": comp, "Categorización": linea.strip()})
                    pd.DataFrame(cat_vereda_export).to_excel(writer, sheet_name="Categorizacion por vereda", index=False)

                buffer_cart.seek(0)
                st.download_button(
                    "⬇ Descargar análisis completo Excel",
                    data=buffer_cart,
                    file_name="analisis_cartografia.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descarga_cart"
                )
            with col2:
                if st.session_state.informe_cart:
                    st.download_button(
                        "⬇ Descargar informe TXT",
                        data=st.session_state.informe_cart.encode("utf-8"),
                        file_name="informe_cartografia.txt",
                        key="descarga_informe_cart"
                    )


