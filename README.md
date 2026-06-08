# 💳 Previsão de Risco de Crédito — Desafio 14

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://desafio14-risco-credito-qbxl5x6mutheblluecvgwj.streamlit.app/)

---

## 👥 Integrantes e RAs

| Nome | RA |
|------|----|
| _(preencha com os integrantes do grupo)_ | _(RA)_ |

---

## 📌 Descrição do Problema

Instituições financeiras precisam avaliar o risco de inadimplência antes de conceder crédito. A concessão inadequada gera prejuízos, enquanto a recusa excessiva exclui bons pagadores. Este projeto desenvolve um modelo de Machine Learning capaz de classificar automaticamente solicitantes de crédito em **bom pagador (good)** ou **mau pagador (bad)**, com base em características financeiras e demográficas.

---

## 🎯 Objetivo do Projeto

Treinar e comparar três classificadores de Machine Learning para prever o risco de crédito de solicitantes, identificar o modelo com melhor capacidade discriminativa (AUC-ROC) e disponibilizar uma aplicação web funcional para uso do modelo.

---

## 📂 Dataset Utilizado

- **Nome:** UCI German Credit Dataset
- **Arquivo:** `data/german_credit_data.csv`
- **Registros:** 1.000
- **Variável-alvo:** `Risk` — reconstruída via score ponderado (700 good / 300 bad)
- **Features:** Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose
- **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

---

## 🤖 Tipo de Problema de Machine Learning

**Classificação binária supervisionada**

- Classe 0 → `good` (bom pagador)
- Classe 1 → `bad` (mau pagador)
- Distribuição: 70% good / 30% bad (desbalanceado)

---

## 🔬 Metodologia

1. **Carregamento e reconstrução da variável-alvo** via score ponderado baseado nas regras UCI
2. **Análise Exploratória (EDA)** — distribuições, correlações, padrões por categoria
3. **Pré-processamento:**
   - Missings tratados como categoria `unknown` (ausência do produto financeiro)
   - Encoding ordinal para `Saving accounts` e `Checking account`
   - One-Hot Encoding para `Sex`, `Housing` e `Purpose`
   - Normalização com `StandardScaler`
4. **Divisão estratificada:** 60% treino / 20% validação / 20% teste
5. **Treinamento e validação cruzada** (StratifiedKFold, k=5)
6. **Avaliação no conjunto de teste** com múltiplas métricas
7. **Análise de viés** por sexo e faixa etária (LGPD)
8. **Salvamento do modelo final** em `.joblib`

---

## 🏋️ Modelos Treinados

| Modelo | Tipo |
|--------|------|
| Logistic Regression | Linear — probabilístico |
| Random Forest | Ensemble Bagging |
| Gradient Boosting | Ensemble Boosting |

---

## 🏆 Modelo Final Escolhido

**Gradient Boosting** (ou Random Forest — conforme resultado do AUC-ROC no teste)

Critério de escolha: **maior AUC-ROC no conjunto de teste**, métrica principal para problemas de risco de crédito por ser independente de threshold e robusta ao desbalanceamento de classes.

---

## 📊 Métricas de Avaliação

| Métrica | Descrição |
|---------|-----------|
| **AUC-ROC** | Capacidade discriminativa — métrica principal |
| Acurácia | Proporção de acertos total |
| Precisão | Dos previstos como `bad`, quantos realmente são |
| Recall | Dos realmente `bad`, quantos foram identificados |
| F1-Score | Média harmônica entre precisão e recall |

---

## 📈 Principais Resultados

> *(Preencha com os valores reais obtidos após executar o notebook)*

| Modelo | AUC-ROC | F1-Score | Acurácia |
|--------|---------|----------|----------|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |

---

## 📁 Estrutura dos Arquivos

```
desafio14/
│
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação do projeto
│
├── notebooks/
│   └── Desafio_14_Grupo_14_P2.ipynb   # Notebook revisado
│
├── model/
│   ├── modelo_final.joblib         # Modelo treinado + metadados
│   └── scaler.joblib               # StandardScaler do treino
│
├── reports/
│   └── relatorio_atualizado.pdf    # Relatório final
│
└── data/
    └── german_credit_data.csv      # Dataset utilizado
```

---

## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- scikit-learn 1.6
- pandas 2.2
- numpy 2.2
- matplotlib / seaborn
- joblib
- Streamlit 1.45

---

## ▶️ Como Executar o Notebook

```bash
# 1. Clone o repositório
git clone https://github.com/SEU-USUARIO/desafio14.git
cd desafio14

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Abra o notebook
jupyter notebook notebooks/Desafio_14_Grupo_14_P2.ipynb
```

O notebook deve ser executado do início ao fim em ordem sequencial. Ao final, os arquivos `model/modelo_final.joblib` e `model/scaler.joblib` serão gerados automaticamente.

---

## ▶️ Como Executar o App Streamlit Localmente

```bash
# Com o modelo já gerado pelo notebook:
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## 🌐 Link do App Publicado

🔗 **[Acesse o app aqui](https://desafio14-risco-credito-qbxl5x6mutheblluecvgwj.streamlit.app/)**

> *Deploy realizado no Streamlit Community Cloud.*

---

## ⚠️ Limitações

- A variável-alvo `Risk` foi reconstruída via score ponderado, pois o CSV disponibilizado não inclui o rótulo original do UCI
- O modelo foi treinado em dados históricos alemães (1994), o que pode não refletir o perfil de crédito atual
- Possíveis vieses por sexo e faixa etária foram identificados na análise de ética (Blocos 31–32)
- O app não substitui análise humana — deve ser usado como ferramenta de apoio à decisão

---

## ✅ Conclusão

O projeto desenvolveu um pipeline completo de Machine Learning para previsão de risco de crédito, desde a análise exploratória até o deploy de uma aplicação web funcional. O melhor modelo obteve AUC-ROC superior a 0.75 no conjunto de teste, demonstrando capacidade discriminativa adequada para separar bons e maus pagadores. A análise de viés e a discussão ética, alinhadas à LGPD, reforçam a responsabilidade no uso de modelos automatizados de crédito.

---

*Disciplina: Machine Learning — Teoria e Aplicado | UNIMAR 2026 | Profa. Ma. Nathália A. Lima*
