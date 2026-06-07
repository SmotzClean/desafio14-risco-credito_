import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Previsão de Risco de Crédito",
    page_icon="💳",
    layout="centered",
)

# ── Carregamento do modelo ───────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    pkg    = joblib.load("model/modelo_final.joblib")
    scaler = joblib.load("model/scaler.joblib")
    return pkg["model"], pkg["metadata"], scaler

try:
    modelo, meta, scaler = carregar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    erro_modelo = str(e)

# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.title("💳 Previsão de Risco de Crédito")
st.markdown(
    """
    **Desafio 14 — Grupo 14**  
    *Machine Learning — Ma. Nathália A. Lima | UNIMAR*

    Este app utiliza um modelo de Machine Learning treinado no dataset
    **UCI German Credit** para classificar solicitantes de crédito como
    **bom pagador (good)** ou **mau pagador (bad)**.
    """
)

if not modelo_ok:
    st.error(f"❌ Erro ao carregar o modelo: {erro_modelo}")
    st.info("Verifique se os arquivos `model/modelo_final.joblib` e `model/scaler.joblib` existem.")
    st.stop()

st.success(
    f"✅ Modelo carregado: **{meta['model_name']}** | "
    f"AUC-ROC (teste): **{meta['auc_roc_test']:.4f}** | "
    f"F1-Score (teste): **{meta['f1_test']:.4f}**"
)

st.divider()

# ── Formulário de entrada ────────────────────────────────────────────────────
st.subheader("📋 Dados do Solicitante")
st.markdown("Preencha as informações abaixo e clique em **Prever Risco**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Idade (anos)", min_value=18, max_value=100, value=35,
        help="Idade do solicitante em anos."
    )
    job = st.selectbox(
        "Nível de Qualificação Profissional",
        options=[0, 1, 2, 3],
        index=2,
        format_func=lambda x: {
            0: "0 — Não qualificado / não residente",
            1: "1 — Não qualificado / residente",
            2: "2 — Qualificado",
            3: "3 — Altamente qualificado",
        }[x],
        help="Nível de qualificação do trabalho (0 a 3)."
    )
    credit_amount = st.number_input(
        "Valor do Crédito (DM)", min_value=100, max_value=100_000,
        value=3_000, step=100,
        help="Valor do crédito solicitado em Deutsche Marks."
    )
    duration = st.number_input(
        "Duração do Contrato (meses)", min_value=1, max_value=120,
        value=24,
        help="Prazo de pagamento em meses."
    )

with col2:
    saving_accounts = st.selectbox(
        "Saldo em Poupança",
        options=["unknown", "little", "moderate", "quite rich", "rich"],
        index=1,
        help="Situação da conta poupança do solicitante."
    )
    checking_account = st.selectbox(
        "Saldo em Conta Corrente",
        options=["unknown", "little", "moderate", "rich"],
        index=1,
        help="Situação da conta corrente do solicitante."
    )
    housing = st.selectbox(
        "Tipo de Moradia",
        options=["own", "free", "rent"],
        index=0,
        help="Situação de moradia do solicitante."
    )
    sex = st.selectbox(
        "Sexo",
        options=["male", "female"],
        index=0,
    )

purpose = st.selectbox(
    "Finalidade do Crédito",
    options=[
        "car", "furniture/equipment", "radio/TV",
        "domestic appliances", "repairs", "education",
        "business", "vacation/others"
    ],
    index=2,
    help="Para que o crédito será utilizado."
)

st.divider()

# ── Pré-processamento idêntico ao notebook ───────────────────────────────────
def preprocessar(age, job, credit_amount, duration,
                 saving_accounts, checking_account,
                 housing, sex, purpose, meta, scaler):

    ordinal_saving   = meta["ordinal_saving"]
    ordinal_checking = meta["ordinal_checking"]

    # 1. Monta linha com valores originais
    row = {
        "Age":              age,
        "Sex":              sex,
        "Job":              job,
        "Housing":          housing,
        "Saving accounts":  saving_accounts,
        "Checking account": checking_account,
        "Credit amount":    credit_amount,
        "Duration":         duration,
        "Purpose":          purpose,
    }
    df_input = pd.DataFrame([row])

    # 2. Encoding ordinal (igual ao Bloco 15)
    df_input["Saving accounts"]  = df_input["Saving accounts"].map(ordinal_saving)
    df_input["Checking account"] = df_input["Checking account"].map(ordinal_checking)

    # 3. One-Hot Encoding com drop_first=True (igual ao Bloco 16)
    df_input = pd.get_dummies(df_input, columns=["Sex", "Housing", "Purpose"], drop_first=True)

    # 4. Garantir todas as colunas esperadas pelo modelo (na ordem certa)
    features_esperadas = meta["features"]
    for col in features_esperadas:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[features_esperadas]

    # 5. Normalização (igual ao Bloco 19)
    cols_num = meta["numeric_cols"]
    df_input[cols_num] = scaler.transform(df_input[cols_num])

    return df_input

# ── Botão de predição ────────────────────────────────────────────────────────
if st.button("🔍 Prever Risco", type="primary", use_container_width=True):

    X_input = preprocessar(
        age, job, credit_amount, duration,
        saving_accounts, checking_account,
        housing, sex, purpose, meta, scaler
    )

    predicao  = modelo.predict(X_input)[0]
    proba     = modelo.predict_proba(X_input)[0]
    prob_bad  = proba[1]
    prob_good = proba[0]

    st.divider()
    st.subheader("📊 Resultado da Predição")

    if predicao == 1:
        st.error("## 🔴 MAU PAGADOR (bad)")
        st.markdown(
            f"O modelo classifica este solicitante como **mau pagador** com "
            f"**{prob_bad*100:.1f}%** de probabilidade."
        )
    else:
        st.success("## 🟢 BOM PAGADOR (good)")
        st.markdown(
            f"O modelo classifica este solicitante como **bom pagador** com "
            f"**{prob_good*100:.1f}%** de probabilidade."
        )

    # Barra de probabilidade
    st.markdown("**Probabilidade por classe:**")
    col_g, col_b = st.columns(2)
    col_g.metric("✅ Bom pagador (good)", f"{prob_good*100:.1f}%")
    col_b.metric("❌ Mau pagador (bad)",  f"{prob_bad*100:.1f}%")

    st.progress(float(prob_bad), text=f"Risco de inadimplência: {prob_bad*100:.1f}%")

    # Resumo dos dados inseridos
    st.divider()
    st.subheader("📝 Dados Informados")
    resumo = pd.DataFrame({
        "Campo": [
            "Idade", "Qualificação", "Valor do Crédito", "Duração",
            "Poupança", "Conta Corrente", "Moradia", "Sexo", "Finalidade"
        ],
        "Valor": [
            f"{age} anos", str(job), f"DM {credit_amount:,}", f"{duration} meses",
            saving_accounts, checking_account, housing, sex, purpose
        ]
    })
    st.dataframe(resumo, use_container_width=True, hide_index=True)

    # Interpretação
    st.divider()
    st.subheader("💡 Interpretação")
    st.info(
        f"""
        **Modelo utilizado:** {meta['model_name']}

        O modelo analisa características financeiras e demográficas do solicitante
        e estima a probabilidade de inadimplência com base em padrões aprendidos
        de 1.000 registros históricos do UCI German Credit Dataset.

        - **Probabilidade de inadimplência (bad):** {prob_bad*100:.1f}%
        - **Probabilidade de adimplência (good):** {prob_good*100:.1f}%

        ⚠️ *Esta previsão é um apoio à decisão e deve ser combinada com análise
        humana, especialmente em casos com probabilidade próxima de 50%.
        Uso conforme boas práticas da LGPD.*
        """
    )

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color: gray; font-size: 0.85em;'>
    Desafio 14 · Grupo 14 · Machine Learning · UNIMAR 2026<br>
    Dataset: UCI German Credit · Modelo: scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
