import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from groq import Groq

st.set_page_config(page_title="Análisis Semántico", layout="wide")
st.title("🔍 Análisis Semántico de Encuestas y Cartografía Social")

# ── CLIENTE GROQ ──────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── CARGAR MODELO ─────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

modelo = cargar_modelo()

# ── FUNCIONES COMPARTIDAS ─────────────────────────────────────────

def encontrar_clusters_optimos(vectores, min_k=2, max_k=8):
    """Determina el número óptimo de clusters con silhouette score."""
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
    """Genera etiquetas descriptivas para cada grupo usando Groq."""
    etiquetas = {}
    for grupo, frases in frases_por_grupo.items():
        muestra = "\n".join(f"- {f}" for f in frases[:5])
        prompt = f"""Eres un experto en análisis social y organizacional.
Analiza estas frases de un grupo semántico en el contexto de {contexto}:
{muestra}

Genera UN título descriptivo de máximo 6 palabras que capture el tema central.
Responde SOLO con el título, sin explicaciones."""
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20
            )
            etiquetas[grupo] = resp.choices[0].message.content.strip()
        except:
            etiquetas[grupo] = f"Grupo {grupo + 1}"
    return etiquetas

def categorizar_hallazgos(frases, contexto):
    """Categoriza frases en problemas, potencialidades, propuestas y alertas."""
    texto = "\n".join(f"- {f}" for f in frases[:30])
    prompt = f"""Eres un experto en diagnóstico social y organizacional.
Analiza estas frases del contexto de {contexto}:
{texto}

Clasifica los hallazgos principales en estas categorías:
- PROBLEMAS: situaciones negativas que afectan a la comunidad u organización
- POTENCIALIDADES: recursos, capacidades o aspectos positivos existentes
- PROPUESTAS: sugerencias o iniciativas planteadas por los participantes
- ALERTAS: situaciones urgentes que requieren atención inmediata

Responde en este formato exacto:
PROBLEMAS:
- [hallazgo]
POTENCIALIDADES:
- [hallazgo]
PROPUESTAS:
- [hallazgo]
ALERTAS:
- [hallazgo]

Máximo 3 puntos por categoría. Si no hay hallazgos para una categoría escribe "No identificados"."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar categorización: {e}"

def generar_informe_ia(resumen_datos, contexto):
    """Genera informe ejecutivo completo con recomendaciones priorizadas."""
    prompt = f"""Eres un consultor experto en {contexto}.
Basándote en este resumen de análisis semántico participativo:

{resumen_datos}

Genera un informe ejecutivo profesional con estas secciones:

1. DIAGNÓSTICO GENERAL (2-3 párrafos)
2. HALLAZGOS PRINCIPALES (máximo 5, ordenados por relevancia)
3. RECOMENDACIONES ESTRATÉGICAS (máximo 5, priorizadas de mayor a menor urgencia)
4. ACCIONES INMEDIATAS (máximo 3 acciones concretas para los próximos 30 días)
5. ALERTAS (situaciones que requieren atención urgente)
6. NOTA SOBRE SUBREGISTRO (analiza si la información parece completa o hay vacíos)

Usa lenguaje profesional, directo y orientado a la toma de decisiones."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar informe: {e}"

def detectar_subregistro(df_analisis, cols_texto=None):
    """Detecta señales de subregistro en los datos."""
    alertas = []
    total = len(df_analisis)

    if total < 10:
        alertas.append(f"⚠️ Muestra pequeña: solo {total} participantes. Los resultados pueden no ser representativos.")

    if cols_texto:
        for col in cols_texto:
            if col in df_analisis.columns:
                vacias = df_analisis[col].isna().sum() + (df_analisis[col] == "").sum()
                pct = round(vacias / total * 100, 1)
                if pct > 20:
                    alertas.append(f"⚠️ La columna '{col}' tiene {pct}% de respuestas vacías.")

        longitudes = df_analisis[cols_texto[0]].dropna().str.len()
        if longitudes.mean() < 20:
            alertas.append("⚠️ Las respuestas son muy cortas en promedio. Puede indicar baja profundidad en las respuestas.")

    if "cluster" in df_analisis.columns:
        dist = df_analisis["cluster"].value_counts(normalize=True)
        if dist.min() < 0.05:
            alertas.append(f"⚠️ Hay grupos con menos del 5% de respuestas. Posible subrepresentación de algunas perspectivas.")

    if not alertas:
        alertas.append("✅ No se detectaron señales evidentes de subregistro con la información disponible.")

    return alertas

# ════════════════════════════════════════════════════════════════
# MÓDULO 1 — ENCUESTA
# ════════════════════════════════════════════════════════════════
if "modo" not in st.session_state:
    st.session_state.modo = "📋 Encuesta"

modo = st.sidebar.radio("Selecciona el tipo de análisis:",
                         ["📋 Encuesta", "🗺️ Cartografía Social"])

if modo == "📋 Encuesta":
    st.header("Módulo de Encuesta")
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
                    lambda row: " ".join(str(v) for v in row if v != ""), axis=1
                )
                textos = df["texto_unido"].tolist()
                vectores = modelo.encode(textos, show_progress_bar=False)

                # Clusters óptimos automáticos
                n_opt = encontrar_clusters_optimos(vectores)
                st.info(f"📊 Número óptimo de grupos detectado automáticamente: **{n_opt}**")

                kmeans = KMeans(n_clusters=n_opt, random_state=42, n_init=10)
                df["cluster_num"] = kmeans.fit_predict(vectores)

                # Ponderación doble
                centroides = kmeans.cluster_centers_
                sims = cosine_similarity(vectores, centroides)
                df["peso_semantico"] = sims.max(axis=1).round(3)
                df["peso_ponderado"] = (df["peso_semantico"] / df.groupby("cluster_num")["peso_semantico"].transform("sum")).round(3)

                # Etiquetas IA
                frases_por_grupo = {}
                for g in range(n_opt):
                    mask = df["cluster_num"] == g
                    frases_por_grupo[g] = df[mask]["texto_unido"].tolist()

            with st.spinner("Generando etiquetas con IA..."):
                etiquetas = etiquetar_grupos_ia(frases_por_grupo, contexto)
                df["cluster"] = df["cluster_num"].map(etiquetas)

                # UMAP
                umap_model = UMAP(n_components=2, random_state=42)
                coords = umap_model.fit_transform(vectores)
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]

            st.success("Análisis completado.")

            # ── SUBREGISTRO ───────────────────────────────────────
            st.subheader("🔎 Detección de subregistro")
            alertas = detectar_subregistro(df, cols_texto)
            for a in alertas:
                st.write(a)

            # ── FILTROS ───────────────────────────────────────────
            st.subheader("Filtrar resultados")
            grupos_disponibles = sorted(df["cluster"].unique())
            grupo_sel = st.multiselect("Filtrar por grupo:", grupos_disponibles, default=grupos_disponibles)
            df_filtrado = df[df["cluster"].isin(grupo_sel)]

            # ── MAPA SEMÁNTICO ────────────────────────────────────
            st.subheader("Mapa semántico de respuestas")
            fig = px.scatter(
                df_filtrado, x="x", y="y", color="cluster",
                hover_data=cols_texto + ["peso_semantico"],
                title="Agrupación semántica de respuestas"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── TORTA Y BARRAS ────────────────────────────────────
            st.subheader("📊 Distribución de grupos")
            conteo = df_filtrado["cluster"].value_counts().reset_index()
            conteo.columns = ["Grupo", "Cantidad"]
            conteo["Porcentaje"] = (conteo["Cantidad"] / conteo["Cantidad"].sum() * 100).round(1)

            col1, col2 = st.columns(2)
            with col1:
                fig_t = px.pie(conteo, names="Grupo", values="Cantidad", title="Porcentaje por grupo", hole=0.3)
                st.plotly_chart(fig_t, use_container_width=True)
            with col2:
                fig_b = px.bar(conteo, x="Grupo", y="Porcentaje", text="Porcentaje", color="Grupo", title="Porcentaje por grupo (%)")
                fig_b.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_b, use_container_width=True)

            # ── LIKERT ────────────────────────────────────────────
            if cols_likert:
                st.subheader("Promedio Likert por grupo")
                resumen_lk = df_filtrado.groupby("cluster")[cols_likert].mean().round(2)
                fig_lk = px.imshow(resumen_lk, text_auto=True, color_continuous_scale="RdYlGn",
                                   title="Mapa de calor: Likert por grupo")
                st.plotly_chart(fig_lk, use_container_width=True)

            # ── SOCIODEMOGRAFÍA ───────────────────────────────────
            if cols_socio:
                st.subheader("👥 Participación sociodemográfica")
                for col_s in cols_socio:
                    if col_s in df_filtrado.columns:
                        conteo_s = df_filtrado[col_s].value_counts().reset_index()
                        conteo_s.columns = [col_s, "Cantidad"]
                        fig_s = px.bar(conteo_s, x=col_s, y="Cantidad", color=col_s,
                                      title=f"Participación por {col_s}")
                        st.plotly_chart(fig_s, use_container_width=True)

                        st.write(f"**Distribución por grupo semántico — {col_s}:**")
                        cruce = df_filtrado.groupby(["cluster", col_s]).size().reset_index(name="n")
                        fig_cruce = px.bar(cruce, x="cluster", y="n", color=col_s, barmode="group",
                                          title=f"Grupos semánticos por {col_s}")
                        st.plotly_chart(fig_cruce, use_container_width=True)

            # ── CATEGORIZACIÓN ────────────────────────────────────
            st.subheader("🗂️ Categorización de hallazgos")
            with st.spinner("Categorizando hallazgos con IA..."):
                todas_frases = df_filtrado["texto_unido"].tolist()
                categorizacion = categorizar_hallazgos(todas_frases, contexto)
            st.markdown(categorizacion)

            # ── RESUMEN EJECUTIVO ─────────────────────────────────
            st.subheader("📋 Resumen ejecutivo por grupo")
            resumen_exec = df_filtrado.groupby("cluster").agg(
                Cantidad=("cluster", "count"),
                Cohesion=("peso_semantico", "mean")
            ).round(3).reset_index()
            resumen_exec["Porcentaje"] = (resumen_exec["Cantidad"] / resumen_exec["Cantidad"].sum() * 100).round(1).astype(str) + "%"
            resumen_exec.columns = ["Grupo", "Respuestas", "Cohesión semántica", "Porcentaje"]
            st.dataframe(resumen_exec, use_container_width=True)

            # ── PONDERACIÓN DOBLE ─────────────────────────────────
            st.subheader("⚖️ Frecuencia simple vs. ponderada")
            pond = df_filtrado.groupby("cluster").agg(
                Frecuencia_simple=("cluster", "count"),
                Peso_ponderado=("peso_ponderado", "sum")
            ).round(3).reset_index()
            fig_pond = px.bar(pond, x="cluster", y=["Frecuencia_simple", "Peso_ponderado"],
                             barmode="group", title="Comparación frecuencia simple vs. ponderada")
            st.plotly_chart(fig_pond, use_container_width=True)

            # ── FRASES REPRESENTATIVAS ────────────────────────────
            st.subheader("💬 Frases más representativas por grupo")
            for grupo in grupos_disponibles:
                if grupo in grupo_sel:
                    with st.expander(f"📌 {grupo}"):
                        top = df_filtrado[df_filtrado["cluster"] == grupo].nlargest(3, "peso_semantico")
                        for _, row in top.iterrows():
                            for col in cols_texto:
                                st.write(f"• {row[col]}")

            # ── INFORME IA ────────────────────────────────────────
            st.subheader("📄 Informe ejecutivo generado por IA")
            resumen_para_ia = f"""
Contexto: {contexto}
Total participantes: {len(df_filtrado)}
Grupos identificados: {n_opt}
Distribución: {conteo.to_string(index=False)}
Cohesión semántica promedio: {df_filtrado['peso_semantico'].mean().round(3)}
Frases más representativas por grupo:
"""
            for grupo in grupos_disponibles:
                if grupo in grupo_sel:
                    top = df_filtrado[df_filtrado["cluster"] == grupo].nlargest(2, "peso_semantico")
                    resumen_para_ia += f"\n{grupo}:\n"
                    for _, row in top.iterrows():
                        resumen_para_ia += f"  - {row['texto_unido']}\n"

            with st.spinner("Generando informe ejecutivo con IA..."):
                informe = generar_informe_ia(resumen_para_ia, contexto)
            st.markdown(informe)

            # ── DESCARGA ──────────────────────────────────────────
            st.subheader("⬇ Descargas")
            col1, col2 = st.columns(2)
            with col1:
                output = df_filtrado.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar datos CSV", output, file_name="resultados_encuesta.csv")
            with col2:
                informe_bytes = informe.encode("utf-8")
                st.download_button("Descargar informe TXT", informe_bytes, file_name="informe_ejecutivo.txt")

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

        columnas = list(df.columns)
        col1, col2 = st.columns(2)
        with col1:
            cols_sel = st.multiselect("Componentes a analizar:", columnas, default=columnas)
        with col2:
            cols_socio = st.multiselect("Columnas sociodemográficas (opcional):", columnas)

        contexto = st.selectbox("Contexto del análisis:",
            ["consultoría y diagnóstico organizacional",
             "bienestar social y desarrollo comunitario",
             "diagnóstico territorial participativo"])

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

                resultados = []
                for comp in cols_sel:
                    mask = df_frases["componente"] == comp
                    vecs_comp = vectores[mask]
                    frases_comp = df_frases[mask]["frase"].tolist()

                    if len(frases_comp) >= 4:
                        n_opt = encontrar_clusters_optimos(vecs_comp, max_k=min(6, len(frases_comp)-1))
                    else:
                        n_opt = min(2, len(frases_comp))

                    if n_opt >= 2:
                        km = KMeans(n_clusters=n_opt, random_state=42, n_init=10)
                        clusters = km.fit_predict(vecs_comp)
                        sims = cosine_similarity(vecs_comp, km.cluster_centers_)
                        pesos = sims.max(axis=1).round(3)
                    else:
                        clusters = [0] * len(frases_comp)
                        pesos = [1.0] * len(frases_comp)

                    frases_por_grupo = {}
                    for i, (f, c) in enumerate(zip(frases_comp, clusters)):
                        frases_por_grupo.setdefault(c, []).append(f)

                    for frase, cluster, peso in zip(frases_comp, clusters, pesos):
                        resultados.append({
                            "componente": comp,
                            "frase": frase,
                            "grupo_num": cluster,
                            "peso_semantico": peso
                        })

                df_result = pd.DataFrame(resultados)

            with st.spinner("Etiquetando grupos con IA..."):
                etiquetas_comp = {}
                for comp in cols_sel:
                    mask = df_result["componente"] == comp
                    fp = {}
                    for _, row in df_result[mask].iterrows():
                        fp.setdefault(row["grupo_num"], []).append(row["frase"])
                    etiquetas_comp[comp] = etiquetar_grupos_ia(fp, contexto)

                def get_etiqueta(row):
                    return etiquetas_comp.get(row["componente"], {}).get(row["grupo_num"], f"Grupo {row['grupo_num']+1}")

                df_result["grupo"] = df_result.apply(get_etiqueta, axis=1)

            st.success("Análisis completado.")

            # ── SUBREGISTRO ───────────────────────────────────────
            st.subheader("🔎 Detección de subregistro")
            alertas = detectar_subregistro(df_result)
            for a in alertas:
                st.write(a)

            # ── FILTROS ───────────────────────────────────────────
            st.subheader("Filtrar resultados")
            col1, col2 = st.columns(2)
            with col1:
                comp_filtro = st.multiselect("Componente:", cols_sel, default=cols_sel)
            with col2:
                grupos_disp = sorted(df_result["grupo"].unique())
                grupo_filtro = st.multiselect("Grupo:", grupos_disp, default=grupos_disp)

            df_filtrado = df_result[
                (df_result["componente"].isin(comp_filtro)) &
                (df_result["grupo"].isin(grupo_filtro))
            ]

            # ── TORTA Y BARRAS ────────────────────────────────────
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

            # ── GRUPOS POR COMPONENTE ─────────────────────────────
            st.subheader("Grupos semánticos por componente")
            conteo_g = df_filtrado.groupby(["componente", "grupo"]).size().reset_index(name="frecuencia")
            fig_g = px.bar(conteo_g, x="componente", y="frecuencia", color="grupo",
                          barmode="group", title="Grupos por componente")
            st.plotly_chart(fig_g, use_container_width=True)

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
                with st.expander(f"📌 Categorización — {comp}"):
                    frases_comp = df_filtrado[df_filtrado["componente"] == comp]["frase"].tolist()
                    with st.spinner(f"Analizando {comp}..."):
                        cat = categorizar_hallazgos(frases_comp, contexto)
                    st.markdown(cat)

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
                    top = df_filtrado[df_filtrado["componente"] == comp].nlargest(3, "peso_semantico")
                    for _, row in top.iterrows():
                        st.write(f"• {row['frase']} (peso: {row['peso_semantico']})")

            # ── INFORME IA ────────────────────────────────────────
            st.subheader("📄 Informe ejecutivo generado por IA")
            resumen_para_ia = f"""
Contexto: {contexto}
Total frases analizadas: {len(df_filtrado)}
Componentes: {', '.join(comp_filtro)}
Distribución:
{conteo_comp.to_string(index=False)}
Cohesión semántica por componente:
{pesos_c.to_string(index=False)}
Frases más representativas:
"""
            for comp in comp_filtro:
                top = df_filtrado[df_filtrado["componente"] == comp].nlargest(2, "peso_semantico")
                resumen_para_ia += f"\n{comp}:\n"
                for _, row in top.iterrows():
                    resumen_para_ia += f"  - {row['frase']}\n"

            with st.spinner("Generando informe ejecutivo con IA..."):
                informe = generar_informe_ia(resumen_para_ia, contexto)
            st.markdown(informe)

            # ── TABLA DETALLADA ───────────────────────────────────
            st.subheader("Tabla detallada")
            comp_vista = st.selectbox("Ver frases del componente:", comp_filtro)
            st.dataframe(df_filtrado[df_filtrado["componente"] == comp_vista], use_container_width=True)

            # ── DESCARGA ──────────────────────────────────────────
            st.subheader("⬇ Descargas")
            col1, col2 = st.columns(2)
            with col1:
                output = df_filtrado.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar datos CSV", output, file_name="resultados_cartografia.csv")
            with col2:
                informe_bytes = informe.encode("utf-8")
                st.download_button("Descargar informe TXT", informe_bytes, file_name="informe_cartografia.txt")

