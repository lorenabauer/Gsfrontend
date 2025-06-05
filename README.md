# Gsfrontend
Integrantes:
Herbertt Di Franco Marques RM: 556640 
Lorena Bauer Nogueira RM: 555272 
Paulo Carvalho Ruiz Borba RM: 554562

Simulador de Potencial de Deslizamentos de Terra
Visão Geral do Projeto

Este projeto consiste em uma solução de ponta a ponta para prever o potencial de deslizamentos de terra, medido por um índice de suscetibilidade ou uma métrica de impacto. A solução inclui um modelo de Machine Learning treinado com dados de deslizamentos e uma aplicação web interativa construída com Streamlit que serve como interface para o modelo.

A ferramenta permite que usuários, como analistas de risco, planejadores urbanos e gestores de defesa civil, simulem cenários e obtenham predições instantâneas, auxiliando em uma tomada de decisão mais estratégica e baseada em dados para a mitigação e prevenção de deslizamentos.

1.1. Motivação do Projeto
Os deslizamentos de terra representam um dos desastres naturais mais recorrentes e devastadores no Brasil, resultando em perdas humanas, danos à infraestrutura e graves impactos socioeconômicos. Muitas vezes, as análises de risco são processos complexos e demorados que exigem alto conhecimento técnico, tornando-as pouco acessíveis para gestores de municípios menores ou para a tomada de decisão em tempo hábil.

Este projeto nasce da necessidade de uma solução acessível que traduza dados complexos (como informações geológicas, meteorológicas e topográficas) em insights práticos, democratizando a análise de risco e apoiando a prevenção de desastres.

1.2. Objetivo
O objetivo principal deste projeto é desenvolver uma plataforma web interativa, impulsionada por um modelo de Machine Learning, capaz de simular o potencial de deslizamentos de terra em diferentes cenários.

Especificamente, o projeto visa:

Treinar e validar um modelo preditivo para estimar um índice de suscetibilidade a deslizamentos com base em variáveis relevantes.
Construir uma aplicação web com Streamlit que permita aos usuários realizar predições de forma intuitiva, tanto para cenários individuais (entrada manual) quanto para múltiplos cenários (upload de arquivo).
Fornecer uma ferramenta de apoio à decisão para profissionais da Defesa Civil, planejadores urbanos e analistas de risco, permitindo a identificação rápida de áreas críticas.
1.3. Resultados Esperados
Ao final do projeto, espera-se alcançar os seguintes resultados:

Um modelo preditivo funcional: Um artefato de modelo (.pkl) treinado e validado, pronto para ser integrado e realizar predições de potencial de deslizamento.
Uma aplicação web interativa e pública: A plataforma déployée na Streamlit Cloud, acessível via web, permitindo que qualquer usuário com os dados necessários possa realizar simulações.
Melhora na capacidade de análise: A ferramenta permitirá que os usuários testem o impacto de diferentes variáveis (ex: "Qual o risco se a precipitação aumentar em 30%?"), gerando uma compreensão mais profunda sobre os fatores de risco.
Agilidade na tomada de decisão: Redução do tempo necessário para avaliar a suscetibilidade de uma determinada área, apoiando a emissão de alertas e o planejamento de ações preventivas de forma mais dinâmica.
Funcionalidades

Modelo Preditivo: Utiliza Regressão Linear (ou outro modelo de sua escolha) para estimar o Potencial de Deslizamento (PD).
Interface Interativa: App web limpo e intuitivo construído com Streamlit.
Modo de Predição Manual: Permite ao usuário inserir manualmente variáveis geotécnicas, topográficas, hidrológicas e contextuais (ex: precipitação, tipo de solo, inclinação, uso do solo) para obter uma predição para um único cenário.
Modo de Predição em Lote: Permite o upload de um arquivo .csv com múltiplos cenários para análise em lote.
Visualização de Dados: Geração de gráficos comparativos (Gráfico de Dispersão e Histograma) para analisar os resultados das predições em lote.
Threshold Customizável: Um slider permite ao usuário definir um limiar para classificar o potencial previsto como "Alto Risco" ou "Baixo Risco".


 Como Funciona: O Treinamento do Modelo (gerador_pkl.py)

Este script é responsável por todo o ciclo de vida do modelo de Machine Learning, desde os dados brutos até os artefatos prontos para produção.

Carga e Limpeza de Dados: O script inicia carregando o dataset deslizamentos.csv e removendo colunas que não serão utilizadas no modelo (como IDs e coordenadas geográficas, se não forem usadas como features).
Pré-processamento e Engenharia de Features:
Processamento de data_hora: Se relevante, a coluna data_hora (ou similar, representando o momento de um evento ou medição) é processada para extrair componentes temporais (ex: hora, dia, mês, duração de chuvas antecedentes).
One-Hot Encoding: Colunas categóricas como tipo_uso_solo ou tipo_geologico são transformadas em colunas numéricas (dummies) para que o modelo possa processá-las.
Remoção de Outliers: Utiliza o método de Amplitude Interquartil (IQR) ou outra técnica para remover valores extremos das principais colunas numéricas, tornando o modelo mais robusto.
Divisão dos Dados: O dataset é dividido em um conjunto de features (X) e a variável alvo (y, que é o potencial_deslizamento_pd ou sua métrica definida).
Escalonamento (Scaling): As features são escalonadas usando StandardScaler. Este passo é crucial para modelos sensíveis à escala das features, garantindo que todas as variáveis contribuam de forma equilibrada.
Treinamento do Modelo: Um modelo (ex: LinearRegression) é instanciado e treinado com os dados de treino (X_train_scaled, y_train).
Salvando os Artefatos: Ao final, o script salva três arquivos essenciais na pasta pickle/:
modelo_regressao_linear.pkl (ou o nome do seu modelo): O objeto do modelo treinado.
scaler.pkl: O objeto do StandardScaler "ajustado" aos dados de treino. É fundamental para processar novas entradas da mesma forma.
colunas_modelo.pkl: A lista exata de colunas que o modelo espera receber. Garante que a aplicação web envie os dados na ordem e formato corretos.
 Como Funciona: A Aplicação Streamlit (app.py)

Este script cria a interface web interativa para o modelo.

Carregamento dos Artefatos: No início, o app carrega o modelo, o scaler e a lista de colunas salvos pelo script de treinamento.
Interface do Usuário:
A tela é dividida em duas abas principais: "Modo CSV" e "Entrada Manual".
Uma barra lateral (sidebar) contém um slider para que o usuário defina o threshold de risco de deslizamento.
Modo de Entrada Manual:
Renderiza sliders, caixas de seleção e campos de entrada para que o usuário insira os valores de cada feature (ex: precipitação acumulada, ângulo da encosta, tipo de solo, etc.).
Ao clicar no botão "Realizar Predição", o app:
Cria um DataFrame de uma única linha com os dados inseridos.
Aplica o scaler carregado para transformar os dados.
Usa o modelo.predict() para obter a previsão do Potencial de Deslizamento (PD).
Exibe o resultado de forma clara, com uma indicação visual de risco.
Modo CSV:
Permite o upload de um arquivo .csv.
Após o upload, o script:
Usa reindex para garantir que as colunas do arquivo correspondam exatamente às colunas_modelo, preenchendo com zero (ou outra estratégia definida) as que estiverem faltando.
Aplica o scaler e o modelo.predict() em todas as linhas do arquivo (predição em lote).
Exibe os resultados em uma tabela, colorindo as linhas com base na classificação de risco (Alto/Baixo).
Oferece uma seção de Análise Gráfica, onde o usuário pode selecionar variáveis para gerar gráficos de dispersão e histogramas, comparando as distribuições entre os grupos de "Alto Risco" e "Baixo Risco".
 Como Executar Localmente e Fazer Deploy na Streamlit Cloud

Parte 1: Executando o Projeto Localmente

Pré-requisitos: Python 3.8+ instalado.
Passo 1: Crie o arquivo requirements.txt Crie um arquivo chamado requirements.txt na pasta principal do seu projeto e adicione as seguintes bibliotecas (ajuste conforme necessário):
Passo 2: Configure o Ambiente Virtual (Recomendado)
Passo 3: Instale as Dependências
Passo 4: Treine o Modelo Execute o script de treinamento para gerar os arquivos .pkl na pasta pickle/.
Passo 5: Execute o App Streamlit Seu navegador abrirá com o aplicativo rodando localmente!
Parte 2: Deploy na Streamlit Cloud

Pré-requisitos: Uma conta no GitHub e uma conta na Streamlit Cloud.
Passo 1: Crie um Repositório no GitHub Crie um novo repositório no GitHub e envie todos os arquivos do seu projeto para ele.
Passo 2: Acesse a Streamlit Cloud Faça login na sua conta da Streamlit Cloud e clique em "New app".
Passo 3: Configure o Deploy
Repository: Escolha o repositório do GitHub que você criou.
Branch: Selecione a branch principal (main ou master).
Main file path: Verifique se o caminho é app.py.
Passo 4: Faça o Deploy! Clique no botão "Deploy!". Seu aplicativo estará no ar em instantes!

Link do Streamlit app:
https://gsfrontenddeslizamentos.streamlit.app/
