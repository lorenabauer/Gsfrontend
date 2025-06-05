import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1) Configuração da página
# -----------------------------------
st.set_page_config(
    page_title='Simulador - Risco de Deslizamento',
    page_icon='⛰️',
    layout='wide'
)
st.title('Simulador de Risco de Deslizamento de Terra')

with st.expander('Descrição do App', expanded=False):
    st.markdown("""
        Este simulador utiliza um modelo de regressão para prever o **Volume Deslizado (m³)**.
        - **Modo CSV**: faça upload de um arquivo `.csv` para predição em lote e análise gráfica.
        - **Entrada Manual**: preencha manualmente as variáveis para uma predição instantânea.
    """)

# -----------------------------------
# 2) Carregamento dos artefatos
# -----------------------------------
@st.cache_resource
def load_model_and_artifacts():
    try:
        model = joblib.load("pickle_deslizamento/modelo_deslizamento.pkl")
        colunas_modelo = joblib.load("pickle_deslizamento/colunas_modelo_deslizamento.pkl")
        scaler = joblib.load("pickle_deslizamento/scaler_deslizamento.pkl")
        return model, colunas_modelo, scaler
    except FileNotFoundError:
        st.error("Erro: Arquivos do modelo não encontrados.")
        st.info("Certifique-se de executar o script 'treinar_modelo_deslizamento.py' primeiro.")
        return None, None, None

model, colunas_modelo, scaler = load_model_and_artifacts()

if not model:
    st.stop()

# Categorias originais para preencher os menus de seleção
cat_deslizamento_originais = ['Fluxo de Lama', 'Queda de Rochas', 'Deslizamento Planar']
cat_solo_originais = ['Argiloso', 'Arenoso', 'Siltoso']
cat_cobertura_originais = ['Rasteira', 'Densa', 'Nenhuma']

# -----------------------------------
# 3) Sidebar com Threshold
# -----------------------------------
threshold = st.sidebar.slider(
    'Definir Threshold de Risco (Volume em m³)',
    min_value=0.0, max_value=50000.0,
    value=100.0, step=100.0,
    help="Qualquer volume previsto ACIMA deste threshold será classificado como 'Alto Risco'."
)
st.sidebar.info(f"Threshold atual: **{threshold:.2f} m³**. Volumes acima são 'Alto Risco'.")

# -----------------------------------
# 4) Abas: Modo CSV e Entrada Manual
# -----------------------------------
tab_csv, tab_manual = st.tabs(["Modo CSV", "Entrada Manual"])

# --- MODO CSV ---
with tab_csv:
    st.subheader("Upload de CSV para Previsão em Lote")
    uploaded_file = st.file_uploader("Faça upload do seu CSV", type="csv")

    if uploaded_file is not None:
        try:
            Xtest_raw = pd.read_csv(uploaded_file)
            st.info(f"CSV carregado com {Xtest_raw.shape[0]} linhas e {Xtest_raw.shape[1]} colunas.")

            # 1) Reindexar, escalar e prever
            Xtest = Xtest_raw.reindex(columns=colunas_modelo, fill_value=0)
            Xtest_scaled = scaler.transform(Xtest)
            y_scores = model.predict(Xtest_scaled)
            y_pred_binary = (y_scores >= threshold).astype(int)

            # 2) Construir DataFrame de saída
            df_pred = Xtest_raw.copy()
            df_pred["volume_previsto_m3"] = y_scores
            df_pred["classificacao_risco"] = np.where(y_pred_binary == 1, 'Alto Risco', 'Baixo Risco')

            # 3) Exibir tabela de resultados
            st.subheader("Resultados das Previsões")
            c1, c2 = st.columns(2)
            alto_risco_count = (df_pred["classificacao_risco"] == 'Alto Risco').sum()
            c1.metric("Total de 'Alto Risco'", int(alto_risco_count))
            c2.metric("Total de 'Baixo Risco'", int(len(df_pred) - alto_risco_count))
            
            def highlight_risk(series):
                return ['background-color: #ff7f7f' if val == 'Alto Risco' else '' for val in series]
            st.dataframe(df_pred.style.apply(highlight_risk, subset=['classificacao_risco']))

            # --- [NOVO] Seção de Análise Gráfica ---
            st.subheader("Análise Gráfica Comparativa")
            with st.expander("Visualizar gráficos", expanded=False):
                numeric_cols = Xtest_raw.select_dtypes(include=np.number).columns.tolist()

                if len(numeric_cols) < 2:
                    st.warning("São necessárias pelo menos duas colunas numéricas no arquivo para gerar gráficos.")
                else:
                    st.markdown("##### Selecione as variáveis para os eixos:")
                    c1_select, c2_select = st.columns(2)
                    with c1_select:
                        selected_col_x = st.selectbox("Variável para o Eixo X:", numeric_cols, index=0)
                    with c2_select:
                        default_y_index = 1 if len(numeric_cols) > 1 else 0
                        selected_col_y = st.selectbox("Variável para o Eixo Y:", numeric_cols, index=default_y_index)

                    st.markdown("##### Ajustes do gráfico:")
                    c1_plot, c2_plot, c3_plot = st.columns(3)
                    with c1_plot:
                        width = st.number_input("Largura", 3, 15, 7, 1)
                    with c2_plot:
                        height = st.number_input("Altura", 3, 15, 5, 1)
                    with c3_plot:
                        alpha = st.slider("Transparência (alpha)", 0.1, 1.0, 0.7, 0.05)
                    
                    plot_size = (width, height)
                    tab_scatter, tab_hist = st.tabs(["Gráfico de Dispersão", "Histograma"])

                    with tab_scatter:
                        if selected_col_x == selected_col_y:
                            st.warning("Selecione variáveis diferentes para os eixos X e Y.")
                        
                        fig_scatter, ax_scatter = plt.subplots(figsize=plot_size)
                        sns.scatterplot(
                            data=df_pred,
                            x=selected_col_x,
                            y=selected_col_y,
                            hue='classificacao_risco', # Adaptado
                            hue_order=['Baixo Risco', 'Alto Risco'], # Adaptado
                            palette={'Baixo Risco': 'green', 'Alto Risco': 'red'}, # Cores
                            ax=ax_scatter,
                            alpha=alpha
                        )
                        ax_scatter.set_title(f"Relação entre '{selected_col_x}' e '{selected_col_y}'")
                        st.pyplot(fig_scatter)

                    with tab_hist:
                        hist_col_choice = st.radio(
                            "Para o Histograma, qual variável deseja analisar?",
                            (selected_col_x, selected_col_y),
                            horizontal=True,
                        )
                        
                        fig_hist, ax_hist = plt.subplots(figsize=plot_size)
                        sns.histplot(
                            data=df_pred,
                            x=hist_col_choice,
                            hue='classificacao_risco', # Adaptado
                            kde=True,
                            stat="density",
                            common_norm=False,
                            hue_order=['Baixo Risco', 'Alto Risco'], # Adaptado
                            palette={'Baixo Risco': 'green', 'Alto Risco': 'red'}, # Cores
                        )
                        ax_hist.set_title(f"Distribuição de '{hist_col_choice}' por Classe de Risco")
                        st.pyplot(fig_hist)
        except Exception as e:
            st.error(f"Erro ao processar o CSV: {e}")

# --- MODO ENTRADA MANUAL ---
with tab_manual:
    st.subheader("Preenchimento Manual de Variáveis")
    with st.form("manual_input_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Condições Climáticas e Geográficas")
            precipitacao_mm_24h = st.slider('Precipitação Acumulada (mm/24h)', 0.0, 300.0, 50.0, step=1.0)
            declividade_graus = st.slider('Declividade da Encosta (graus)', 0.0, 60.0, 25.0, step=0.5)
            horario = st.slider('Hora do Dia (0–23)', 0, 23, 14, step=1)
        with c2:
            st.markdown("##### Características do Local")
            tipo_solo_sel = st.selectbox("Tipo de Solo Predominante", cat_solo_originais)
            tipo_cobertura_sel = st.selectbox("Tipo de Cobertura Vegetal", cat_cobertura_originais)
            tipo_deslizamento_sel = st.selectbox("Tipo de Deslizamento mais Provável", cat_deslizamento_originais)

        submit_button = st.form_submit_button(label="Avaliar Risco de Deslizamento")

    if submit_button:
        try:
            data_dict = {col: 0 for col in colunas_modelo}
            data_dict["precipitacao_mm_24h"] = precipitacao_mm_24h
            data_dict["declividade_graus"] = declividade_graus
            data_dict["horario"] = horario
            
            key_deslizamento = f"tipo_deslizamento_{tipo_deslizamento_sel}"
            if key_deslizamento in data_dict: data_dict[key_deslizamento] = 1
            key_solo = f"tipo_solo_{tipo_solo_sel}"
            if key_solo in data_dict: data_dict[key_solo] = 1
            key_cobertura = f"cobertura_vegetal_{tipo_cobertura_sel}"
            if key_cobertura in data_dict: data_dict[key_cobertura] = 1

            df_input = pd.DataFrame([data_dict], columns=colunas_modelo)
            df_input_scaled = scaler.transform(df_input)
            score = model.predict(df_input_scaled)[0]
            
            st.write("---")
            st.subheader("Resultado da Análise de Risco")
            if score >= threshold:
                st.error(f"🚨 ALTO RISCO PREVISTO! (Volume Estimado: {score:.2f} m³)")
            else:
                st.success(f"✅ Baixo risco previsto. (Volume Estimado: {score:.2f} m³)")

            with st.expander("Detalhes da predição"):
                st.write(f"**Volume Estimado**: {score:.2f} m³")
                st.write(f"**Threshold de Risco definido**: {threshold:.2f} m³")
                st.progress(min(score / (threshold * 1.5), 1.0))

        except Exception as e:
            st.error(f"Erro ao fazer a predição: {e}")