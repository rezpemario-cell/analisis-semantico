import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from umap import UMAP

# ── CONFIGURACIÓN DE LA APP ──────────────────────────────────────
st.set_page_config(page_title="Análisis Semántico", layout="wide")
st.title("🔍 Análisis Semántico de Encuestas y Cartografía Social")

# ── CARGAR MODELO ─────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

modelo = cargar_modelo()

# ── SELECTOR DE MODO ─────────────────────────────────────────────
modo = st.sidebar.radio("Selecciona el tipo de análisis:",
                         ["📋 Encuesta", "🗺️ Cartografía Social"])

# ════════════════════════════════════════════════════════════════
# MÓDULO 1 — ENCUESTA
# ════════════════════════════════════════════════════════════════
if modo == "📋 Encuesta":
    st.header("Módulo de Encuesta")
    st.write("Sube tu Excel con columnas Likert y respuestas abiertas.")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

    if archivo:
        df = pd.read_excel(archivo)
        st.subheader("Vista previa de tus datos")
        st.dataframe(df.head())

        columnas = list(df.columns)

        col1, col2 = st.columns(2)
        with col1:
            cols_likert = st.multiselect(
                "Selecciona las columnas Likert (numéricas):",
                columnas
            )
        with col2:
            cols_texto = st.multiselect(
                "Selecciona las columnas de texto abierto:",
                columnas
            )

        if cols_texto and st.button("▶ Analizar"):
            with st.spinner("Procesando textos con el modelo semántico..."):

                df["texto_unido"] = df[cols_texto].fillna("").apply(
                    lambda row: " ".join(str(v) for v in row if v != ""), axis=1
                )
                textos = df["texto_unido"].tolist()

                vectores = modelo.encode(textos, show_progress_bar=False)

                n_clusters = min(4, len(textos))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df["cluster"] = kmeans.fit_predict(vectores)

                umap_model = UMAP(n_components=2, random_state=42)
                coords = umap_model.fit_transform(vectores)
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]

                centroides = kmeans.cluster_centers_
                sims = cosine_similarity(vectores, centroides)
                df["peso_semantico"] = sims.max(axis=1).round(3)

            st.success("Análisis completado.")

            st.subheader("Mapa semántico de respuestas")
            fig = px.scatter(
                df, x="x", y="y", color=df["cluster"].astype(str),
                hover_data=cols_texto + ["peso_semantico"],
                title="Agrupación semántica de respuestas",
                labels={"color": "Grupo"}
            )
            st.plotly_chart(fig, use_container_width=True)

            if cols_likert:
                st.subheader("Promedio Likert por grupo semántico")
                resumen = df.groupby("cluster")[cols_likert].mean().round(2)
                fig2 = px.imshow(
                    resumen,
                    text_auto=True,
                    color_continuous_scale="RdYlGn",
                    title="Mapa de calor: Likert promedio por grupo"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Tabla de resultados")
            cols_mostrar = cols_texto + ["cluster", "peso_semantico"]
            if cols_likert:
                cols_mostrar = cols_likert + cols_mostrar
            st.dataframe(df[cols_mostrar])

            output = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Descargar resultados CSV",
                output,
                file_name="resultados_encuesta.csv"
            )

# ════════════════════════════════════════════════════════════════
# MÓDULO 2 — CARTOGRAFÍA SOCIAL
# ════════════════════════════════════════════════════════════════
elif modo == "🗺️ Cartografía Social":
    st.header("Módulo de Cartografía Social")
    st.write("Sube tu Excel donde cada columna es un componente y las celdas contienen frases separadas por punto.")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

    if archivo:
        df = pd.read_excel(archivo)
        st.subheader("Vista previa de tus datos")
        st.dataframe(df.head())

        componentes = list(df.columns)
        cols_sel = st.multiselect(
            "Selecciona los componentes a analizar:",
            componentes,
            default=componentes
        )

        if cols_sel and st.button("▶ Analizar"):
            with st.spinner("Fragmentando frases y procesando con el modelo..."):

                registros = []
                for componente in cols_sel:
                    for celda in df[componente].dropna():
                        frases = [f.strip() for f in str(celda).split(".") if len(f.strip()) > 5]
                        for frase in frases:
                            registros.append({"componente": componente, "frase": frase})

                df_frases = pd.DataFrame(registros)
                st.write(f"Total de frases extraídas: {len(df_frases)}")

                vectores = modelo.encode(df_frases["frase"].tolist(), show_progress_bar=False)

                n_clusters = 3
                resultados = []
                for comp in cols_sel:
                    mask = df_frases["componente"] == comp
                    vecs_comp = vectores[mask]
                    frases_comp = df_frases[mask]["frase"].tolist()

                    k = min(n_clusters, len(frases_comp))
                    if k >= 2:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        clusters = km.fit_predict(vecs_comp)
                        sims = cosine_similarity(vecs_comp, km.cluster_centers_)
                        pesos = sims.max(axis=1).round(3)
                    else:
                        clusters = [0] * len(frases_comp)
                        pesos = [1.0] * len(frases_comp)

                    for frase, cluster, peso in zip(frases_comp, clusters, pesos):
                        resultados.append({
                            "componente": comp,
                            "frase": frase,
                            "grupo": cluster,
                            "peso_semantico": peso
                        })

                df_result = pd.DataFrame(resultados)

            st.success("Análisis completado.")

            st.subheader("Distribución de grupos por componente")
            conteo = df_result.groupby(["componente", "grupo"]).size().reset_index(name="frecuencia")
            fig = px.bar(
                conteo, x="componente", y="frecuencia",
                color=conteo["grupo"].astype(str),
                barmode="group",
                title="Grupos semánticos por componente"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Peso semántico promedio por componente")
            pesos_comp = df_result.groupby("componente")["peso_semantico"].mean().round(3).reset_index()
            fig2 = px.bar(
                pesos_comp, x="componente", y="peso_semantico",
                color="peso_semantico", color_continuous_scale="Teal",
                title="Cohesión semántica por componente"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Tabla detallada de frases")
            componente_sel = st.selectbox("Ver frases del componente:", cols_sel)
            st.dataframe(df_result[df_result["componente"] == componente_sel])

            output = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Descargar resultados CSV",
                output,
                file_name="resultados_cartografia.csv"
            )    