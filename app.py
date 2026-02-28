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

st.set_page_config(page_title="Análisis Semántico", layout="wide")
st.title("🔍 Análisis Semántico de Encuestas y Cartografía Social")

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

modelo = cargar_modelo()

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
            cols_likert = st.multiselect("Selecciona las columnas Likert (numéricas):", columnas)
        with col2:
            cols_texto = st.multiselect("Selecciona las columnas de texto abierto:", columnas)

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
                df["cluster"] = df["cluster"].apply(lambda x: f"Grupo {x+1}")

                umap_model = UMAP(n_components=2, random_state=42)
                coords = umap_model.fit_transform(vectores)
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]

                centroides = kmeans.cluster_centers_
                sims = cosine_similarity(vectores, centroides)
                df["peso_semantico"] = sims.max(axis=1).round(3)

            st.success("Análisis completado.")

            # ── FILTRO POR GRUPO ──────────────────────────────────
            st.subheader("🔎 Filtrar resultados")
            grupos_disponibles = sorted(df["cluster"].unique())
            grupo_sel = st.multiselect(
                "Filtrar por grupo semántico:",
                grupos_disponibles,
                default=grupos_disponibles
            )
            df_filtrado = df[df["cluster"].isin(grupo_sel)]

            # ── MAPA SEMÁNTICO ────────────────────────────────────
            st.subheader("Mapa semántico de respuestas")
            fig = px.scatter(
                df_filtrado, x="x", y="y", color="cluster",
                hover_data=cols_texto + ["peso_semantico"],
                title="Agrupación semántica de respuestas"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── GRÁFICA DE TORTA Y BARRAS ─────────────────────────
            st.subheader("📊 Distribución de grupos")
            conteo = df_filtrado["cluster"].value_counts().reset_index()
            conteo.columns = ["Grupo", "Cantidad"]
            conteo["Porcentaje"] = (conteo["Cantidad"] / conteo["Cantidad"].sum() * 100).round(1)

            col1, col2 = st.columns(2)
            with col1:
                fig_torta = px.pie(
                    conteo, names="Grupo", values="Cantidad",
                    title="Porcentaje por grupo",
                    hole=0.3
                )
                st.plotly_chart(fig_torta, use_container_width=True)
            with col2:
                fig_barras = px.bar(
                    conteo, x="Grupo", y="Porcentaje",
                    text="Porcentaje",
                    title="Porcentaje por grupo (%)",
                    color="Grupo"
                )
                fig_barras.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_barras, use_container_width=True)

            # ── MAPA DE CALOR LIKERT ──────────────────────────────
            if cols_likert:
                st.subheader("Promedio Likert por grupo semántico")
                resumen = df_filtrado.groupby("cluster")[cols_likert].mean().round(2)
                fig2 = px.imshow(
                    resumen, text_auto=True,
                    color_continuous_scale="RdYlGn",
                    title="Mapa de calor: Likert promedio por grupo"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── TABLA RESUMEN EJECUTIVO ───────────────────────────
            st.subheader("📋 Resumen ejecutivo por grupo")
            resumen_exec = df_filtrado.groupby("cluster").agg(
                Cantidad=("cluster", "count"),
                Peso_promedio=("peso_semantico", "mean")
            ).round(3).reset_index()
            resumen_exec["Porcentaje"] = (
                resumen_exec["Cantidad"] / resumen_exec["Cantidad"].sum() * 100
            ).round(1).astype(str) + "%"
            resumen_exec.columns = ["Grupo", "Cantidad de respuestas", "Cohesión semántica promedio", "Porcentaje del total"]
            st.dataframe(resumen_exec, use_container_width=True)

            # ── FRASES REPRESENTATIVAS POR GRUPO ─────────────────
            st.subheader("💬 Frases más representativas por grupo")
            for grupo in grupos_disponibles:
                if grupo in grupo_sel:
                    with st.expander(f"Ver frases — {grupo}"):
                        top = df_filtrado[df_filtrado["cluster"] == grupo].nlargest(3, "peso_semantico")
                        for _, row in top.iterrows():
                            for col in cols_texto:
                                st.write(f"• {row[col]}")

            # ── TABLA COMPLETA ────────────────────────────────────
            st.subheader("Tabla completa de resultados")
            cols_mostrar = cols_texto + ["cluster", "peso_semantico"]
            if cols_likert:
                cols_mostrar = cols_likert + cols_mostrar
            st.dataframe(df_filtrado[cols_mostrar], use_container_width=True)

            output = df_filtrado.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Descargar resultados CSV", output, file_name="resultados_encuesta.csv")

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
        cols_sel = st.multiselect("Selecciona los componentes a analizar:", componentes, default=componentes)

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
                            "grupo": f"Grupo {cluster+1}",
                            "peso_semantico": peso
                        })

                df_result = pd.DataFrame(resultados)

            st.success("Análisis completado.")

            # ── FILTROS ───────────────────────────────────────────
            st.subheader("🔎 Filtrar resultados")
            col1, col2 = st.columns(2)
            with col1:
                comp_filtro = st.multiselect("Filtrar por componente:", cols_sel, default=cols_sel)
            with col2:
                grupos_disp = sorted(df_result["grupo"].unique())
                grupo_filtro = st.multiselect("Filtrar por grupo:", grupos_disp, default=grupos_disp)

            df_filtrado = df_result[
                (df_result["componente"].isin(comp_filtro)) &
                (df_result["grupo"].isin(grupo_filtro))
            ]

            # ── GRÁFICA DE TORTA Y BARRAS ─────────────────────────
            st.subheader("📊 Distribución por componente")
            col1, col2 = st.columns(2)
            conteo_comp = df_filtrado["componente"].value_counts().reset_index()
            conteo_comp.columns = ["Componente", "Frases"]
            conteo_comp["Porcentaje"] = (conteo_comp["Frases"] / conteo_comp["Frases"].sum() * 100).round(1)

            with col1:
                fig_torta = px.pie(
                    conteo_comp, names="Componente", values="Frases",
                    title="Distribución de frases por componente",
                    hole=0.3
                )
                st.plotly_chart(fig_torta, use_container_width=True)
            with col2:
                fig_barras = px.bar(
                    conteo_comp, x="Componente", y="Porcentaje",
                    text="Porcentaje", color="Componente",
                    title="Porcentaje de frases por componente (%)"
                )
                fig_barras.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_barras, use_container_width=True)

            # ── GRUPOS POR COMPONENTE ─────────────────────────────
            st.subheader("Grupos semánticos por componente")
            conteo_grupos = df_filtrado.groupby(["componente", "grupo"]).size().reset_index(name="frecuencia")
            fig3 = px.bar(
                conteo_grupos, x="componente", y="frecuencia",
                color="grupo", barmode="group",
                title="Distribución de grupos por componente"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # ── PESO SEMÁNTICO ────────────────────────────────────
            st.subheader("Cohesión semántica por componente")
            pesos_comp = df_filtrado.groupby("componente")["peso_semantico"].mean().round(3).reset_index()
            fig4 = px.bar(
                pesos_comp, x="componente", y="peso_semantico",
                color="peso_semantico", color_continuous_scale="Teal",
                title="Peso semántico promedio por componente"
            )
            st.plotly_chart(fig4, use_container_width=True)

            # ── RESUMEN EJECUTIVO ─────────────────────────────────
            st.subheader("📋 Resumen ejecutivo por componente")
            resumen_exec = df_filtrado.groupby("componente").agg(
                Total_frases=("frase", "count"),
                Cohesion_promedio=("peso_semantico", "mean")
            ).round(3).reset_index()
            resumen_exec["Porcentaje"] = (
                resumen_exec["Total_frases"] / resumen_exec["Total_frases"].sum() * 100
            ).round(1).astype(str) + "%"
            resumen_exec.columns = ["Componente", "Total frases", "Cohesión semántica promedio", "Porcentaje del total"]
            st.dataframe(resumen_exec, use_container_width=True)

            # ── FRASES REPRESENTATIVAS ────────────────────────────
            st.subheader("💬 Frases más representativas por componente")
            for comp in comp_filtro:
                with st.expander(f"Ver frases — {comp}"):
                    top = df_filtrado[df_filtrado["componente"] == comp].nlargest(3, "peso_semantico")
                    for _, row in top.iterrows():
                        st.write(f"• {row['frase']} (peso: {row['peso_semantico']})")

            # ── TABLA DETALLADA ───────────────────────────────────
            st.subheader("Tabla detallada")
            componente_sel = st.selectbox("Ver frases del componente:", comp_filtro)
            st.dataframe(df_filtrado[df_filtrado["componente"] == componente_sel], use_container_width=True)

            output = df_filtrado.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Descargar resultados CSV", output, file_name="resultados_cartografia.csv")
            )    
