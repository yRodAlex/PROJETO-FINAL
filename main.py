import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

st.set_page_config(page_title='SecureTrust Project', layout='wide')
st.title("SecureTrust")



if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Secure-Trust"

tab = st.radio("Navegue pelas seções:", ["Secure-Trust", "Documentação Técnica", "Protótipo", "Sobre Nós"], horizontal=True)

if st.session_state.active_tab != tab:
    st.session_state.active_tab = tab
    for k in ["graficos_gerados", "data", "modelo"]:
        st.session_state.pop(k, None)
    st.rerun()

if tab == "Secure-Trust":
    st.header("\U0001F3E6 Cenário: Projeto SecureTrust")
    st.markdown("""📘 **Pedido de Projeto Completo com sua Empresa de Dados**

🏢 **Sobre nós:**  
Somos uma fintech chamada **SecureTrust**, especializada em microcrédito digital para pessoas não-bancarizadas em países emergentes.  
Atuamos 100% online, liberando crédito de forma automatizada por meio de app e web, com forte crescimento em regiões com baixa bancarização — especialmente na África e América Latina.

🔍 **Nosso problema: fraude bancária digital**  
Nosso processo de análise de crédito é quase instantâneo. Porém, isso nos torna altamente vulneráveis a fraudes digitais, como:
- Roubo de identidade  
- Abertura de contas falsas  
- Solicitações massivas com dados gerados artificialmente  

Em 2024, estimamos perdas superiores a **US$ 1 milhão** por fraudes. Isso nos fez buscar uma consultoria especializada — e é por isso que procuramos sua empresa.

🎯 **Nosso pedido: solução ponta a ponta**  
Entregamos a vocês a base pública do Kaggle, que simula um conjunto de dados realista sobre solicitações bancárias, com a coluna `fraud_bool` identificando fraudes confirmadas:  
🔗 *Bank Account Fraud Dataset — NeurIPS 2022*

**Sua missão:**  
Desenvolver uma solução completa de engenharia e ciência de dados, que nos ajude a detectar, visualizar e prevenir fraudes no futuro.

📦 **Escopo técnico desejado**

✅ **Engenharia de Dados**
- Tratar valores faltantes e outliers  
- Criar variáveis derivadas (ex: tempo entre solicitações, padrões suspeitos)  
- Construir uma pipeline reutilizável de ingestão e transformação  
- Armazenar dados limpos em ambiente estruturado (idealmente pronto para BI)

✅ **Ciência de Dados**
- Exploração estatística: entender padrões comuns em fraudes  
- Identificar variáveis mais correlacionadas com fraude  
- Treinar modelos de classificação (Árvore de Decisão, Random Forest, XGBoost)  
- Avaliar curva ROC, matriz de confusão, F1-score  
- Criar um modelo interpretável (ex: SHAP ou Feature Importance)

✅ **Dashboard final (Power BI, Tableau ou Streamlit)**  
Visão geral com KPIs:
- Total de solicitações  
- % de fraudes detectadas  
- Prejuízo estimado por fraudes  
- Gráfico temporal de fraudes por mês  
- Heatmap por tipo de dispositivo, região ou sistema operacional  
- Filtros por tipo de pagamento, origem do usuário, idade  
- Visualização de "riscos em tempo real" por categoria de cliente  
- Ranking de variáveis mais indicativas de fraude

🧠 **Queremos insights como:**
- Qual perfil mais comum de fraude?  
- Existe um padrão de dispositivo ou sistema operacional que favorece fraude?  
- Usuários com nome e e-mail muito semelhantes (alta similaridade) são mais confiáveis?  
- Quanto tempo uma conta costuma permanecer ativa antes da fraude?  
- Vale a pena mudar nossa regra de concessão de crédito?

💼 **Entregáveis esperados**

| Entregável                  | Descrição                                                       |
|----------------------------|------------------------------------------------------------------|
| 📊 Dashboard interativo     | KPIs e visualizações com insights acionáveis                    |
| 🧪 Modelo de detecção       | Algoritmo treinado e documentado, pronto para produção          |
| 📁 Base tratada             | Arquivo com todos os campos limpos e transformações aplicadas   |
| 🔄 Pipeline                 | Script automatizado (preferencialmente Python + Pandas/PySpark) |
| 📝 Apresentação executiva   | Storytelling dos resultados para diretoria não técnica          |

🧩 **Como diferencial:**  
Queremos que sua empresa sugira melhorias estratégicas baseadas em ciência de dados aplicada ao negócio — como por exemplo:
- Alterar limite de crédito de acordo com risco do perfil  
- Priorizar análise manual de clientes com score intermediário  
- Automatizar bloqueio para dispositivos com histórico suspeito
                
""")

elif tab == "Documentação Técnica":
    st.header("\U0001F4C4 Documentação Técnica")
    st.markdown(
        """
        ### Projeto de Engenharia de Dados – Ingestão, Tratamento e Consulta via S3, Glue e Athena

        ---

        #### 🧭 Visão Geral
        Este documento descreve a estrutura e implementação de um pipeline de dados utilizando os serviços da AWS: **S3**, **Glue (Python Shell)** e **Athena**.  
        O objetivo é realizar a ingestão de arquivos CSV para o S3, executar tratativas via Glue Job e tornar os dados disponíveis para consulta via Athena.

        ---

        #### 📂 Estrutura do Bucket S3
        **Bucket:** `securetrust-bucket`
                
        ```
        s3://securetrust-bucket/sistemadeorientacaodecredito/
            ├── BRONZE/    # Dados brutos (originais)
            ├── PRATA/     # Dados tratados
            └── OURO/      # Dados finalizados para BI ou modelo
        ```

        ---

        #### 🧪 Glue Job – Tratamento de Dados (Python Shell)

        **Tratativas Realizadas:**

        1. **Substituição de valores inválidos**  
           - Colunas: `prev_address_months_count`, `current_address_months_count`, `device_distinct_emails_8w`, `session_length_in_minutes`  
           - Ação: substituição de valores `-1` por `NaN`

        2. **Remoção de duplicatas**  
           - Ação: `drop_duplicates()`

        3. **Remoção de coluna irrelevante**  
           - Coluna: `intended_balcon_amount`  
           - Ação: removida por inconsistência ou irrelevância

        4. **Criação de coluna auxiliar com nome do mês**  
           - Nova coluna: `month_named`  
           - Baseada no valor de `month` mapeado por dicionário

        ---

        #### 🗃️ Athena – Criação da Tabela de Consulta

        ```sql
        CREATE DATABASE IF NOT EXISTS securetrust_db;

        CREATE EXTERNAL TABLE IF NOT EXISTS securetrust_db.base_tratada (
          fraud_bool INT,
          income DOUBLE,
          name_email_similarity DOUBLE,
          prev_address_months_count INT,
          current_address_months_count INT,
          customer_age INT,
          days_since_request DOUBLE,
          payment_type STRING,
          zip_count_4w INT,
          velocity_6h DOUBLE,
          velocity_24h DOUBLE,
          velocity_4w DOUBLE,
          bank_branch_count_8w INT,
          date_of_birth_distinct_emails_4w INT,
          employment_status STRING,
          credit_risk_score INT,
          email_is_free INT,
          housing_status STRING,
          phone_home_valid INT,
          phone_mobile_valid INT,
          bank_months_count INT,
          has_other_cards INT,
          proposed_credit_limit DOUBLE,
          foreign_request INT,
          source STRING,
          session_length_in_minutes DOUBLE,
          device_os STRING,
          keep_alive_session INT,
          device_distinct_emails_8w INT,
          device_fraud_count INT,
          month INT,
          month_named STRING
        )
        ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
        WITH SERDEPROPERTIES (
          'serialization.format' = ',',
          'field.delim' = ','
        )
        LOCATION 's3://securetrust-bucket/sistemadeorientacaodecredito/PRATA/'
        TBLPROPERTIES ('has_encrypted_data'='false', 'skip.header.line.count'='1');

        SELECT * FROM securetrust_db.base_tratada LIMIT 10;
        ```

        ---

        #### 🧾 Observações Finais
        - O arquivo CSV deve ter delimitador `,` e conter cabeçalho  
        - O Glue Job pode ser agendado para execução automática  
        - O uso de Crawler foi evitado para manter controle manual de schema e tipos  

        ---

        #### 🐍 Código Python para Tratamento dos Dados

        ```python
        import boto3
        import pandas as pd
        import numpy as np
        import io

        bucket = 'securetrust-bucket'
        origem_prefix = 'sistemadeorientacaodecredito/BRONZE/'
        destino_prefix = 'sistemadeorientacaodecredito/PRATA/'

        colunas_com_menos_um = [
            'prev_address_months_count',
            'current_address_months_count',
            'device_distinct_emails_8w',
            'session_length_in_minutes'
        ]

        mapa_meses = {
            0: 'Janeiro', 1: 'Fevereiro', 2: 'Março', 3: 'Abril',
            4: 'Maio', 5: 'Junho', 6: 'Julho', 7: 'Agosto',
            8: 'Setembro', 9: 'Outubro', 10: 'Novembro', 11: 'Dezembro'
        }

        s3 = boto3.client('s3')

        try:
            print("🔁 Listando arquivos na pasta BRONZE...")
            response = s3.list_objects_v2(Bucket=bucket, Prefix=origem_prefix)

            if 'Contents' not in response:
                print("Nenhum arquivo encontrado.")
            else:
                for obj in response['Contents']:
                    origem_key = obj['Key']
                    if origem_key.endswith('/'):
                        continue
                    nome_arquivo = origem_key.split('/')[-1]
                    destino_key = f"{destino_prefix}{nome_arquivo}"

                    print(f"📥 Processando {origem_key}...")
                    csv_obj = s3.get_object(Bucket=bucket, Key=origem_key)
                    df = pd.read_csv(csv_obj['Body'])

                    df[colunas_com_menos_um] = df[colunas_com_menos_um].replace(-1, np.nan)
                    df = df.drop_duplicates()

                    if 'intended_balcon_amount' in df.columns:
                        df = df.drop(columns=['intended_balcon_amount'])

                    df['month_named'] = df['month'].map(mapa_meses)

                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    s3.put_object(Bucket=bucket, Key=destino_key, Body=csv_buffer.getvalue())

                    print(f"✅ Arquivo tratado e salvo em: {destino_key}")

        except Exception as e:
            print("❌ Erro ao processar arquivos:", str(e))
        ```
        """
    )



elif tab == "Protótipo":
    st.header("🤖 Protótipo de Predição de Fraudes")
    st.markdown("O arquivo `Base.csv` foi carregado automaticamente para detecção de fraudes. Verifique os dados abaixo:")

    try:
        chunk_size = 10**6
        chunks = []
        for chunk in pd.read_csv("data/Base.csv", chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= 2_000_000:
                st.warning("Arquivo muito grande! Apenas parte dos dados foi carregada.")
                break
        data = pd.concat(chunks, ignore_index=True)
        st.success("✅ Arquivo 'Base.csv' carregado com sucesso!")
        st.write("Pré-visualização dos dados:")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo Base.csv: {e}")
        st.stop()

    st.info('Clique no botão abaixo para processar o arquivo')
    processar = st.button("Processar")

    if processar:
        try:
            def codificar_dados(df):
                df = df.copy()
                for col in df.select_dtypes(include='object').columns:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                for col in df.select_dtypes(include='bool').columns:
                    df[col] = df[col].astype(int)
                return df

            def tratar_nulos(X):
                imputer = SimpleImputer(strategy='median')
                return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            def preparar_dados(df):
                df = df.dropna(subset=['fraud_bool'])
                X = df.drop(columns=["fraud_bool"])
                y = df["fraud_bool"]
                X = codificar_dados(X)
                X = tratar_nulos(X)
                return X, y

            def balancear_amostras(X, y):
                smote_enn = SMOTEENN(random_state=42)
                return smote_enn.fit_resample(X, y)

            def treinar_modelo(X, y):
                model = XGBClassifier(
                    scale_pos_weight=10,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X, y)
                return model

            X, y = preparar_dados(data)
            X_bal, y_bal = balancear_amostras(X, y)
            modelo = treinar_modelo(X_bal, y_bal)

            probs = modelo.predict_proba(X)[:, 1]

            st.subheader("⚙️ Thresholds para Avaliação Tabular")
            thr_input = st.text_input("Digite os thresholds desejados (ex: 0.3, 0.5, 0.7):", value="0.3, 0.5, 0.7")

            try:
                thresholds = [float(t.strip()) for t in thr_input.split(",")]
            except:
                st.error("⚠️ Formato inválido. Use números separados por vírgula.")
                thresholds = [0.5]

            results = []
            for thr in thresholds:
                y_pred_thr = (probs >= thr).astype(int)
                cm = confusion_matrix(y, y_pred_thr)
                prec = precision_score(y, y_pred_thr, zero_division=0)
                rec = recall_score(y, y_pred_thr, zero_division=0)
                f1 = f1_score(y, y_pred_thr, zero_division=0)
                acc = accuracy_score(y, y_pred_thr)
                results.append({
                    "threshold": thr,
                    "TN (True Negative)": int(cm[0,0]),
                    "FP (True Positive)": int(cm[0,1]),
                    "FN (Fase Negative)": int(cm[1,0]),
                    "TP (False Negative)": int(cm[1,1]),
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1
                })

            st.subheader("📊 Comparação de Métricas por Threshold")
            df_result = pd.DataFrame(results).set_index("threshold")
            st.dataframe(df_result.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}"
            }))

            melhor_resultado = max(results, key=lambda x: x["F1"])
            melhor_threshold = melhor_resultado["threshold"]
            melhor_f1 = melhor_resultado["F1"]

            st.success(f"🔍 Melhor threshold com base no F1-Score: **{melhor_threshold:.2f}** (F1 = {melhor_f1:.2%})")

            st.subheader("📊 Falsos Positivos vs Falsos Negativos por Threshold")
            fps = [r["FP (True Positive)"] for r in results]
            fns = [r["FN (Fase Negative)"] for r in results]
            thresholds_plot = [f"{r['threshold']:.2f}" for r in results]

            x = np.arange(len(thresholds_plot))
            bar_width = 0.35

            fig4, ax4 = plt.subplots()
            ax4.bar(x - bar_width/2, fps, width=bar_width, label='Falsos Positivos (FP)')
            ax4.bar(x + bar_width/2, fns, width=bar_width, label='Falsos Negativos (FN)')

            ax4.set_xlabel("Threshold")
            ax4.set_ylabel("Quantidade")
            ax4.set_title("Comparação: Falsos Positivos vs Falsos Negativos")
            ax4.set_xticks(x)
            ax4.set_xticklabels(thresholds_plot)
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig4)

            st.subheader("📈 Relação entre Precision e F1-Score por Threshold")
            precisions = [r["Precision"] for r in results]
            f1_scores = [r["F1"] for r in results]
            thresholds_plot = [r["threshold"] for r in results]

            fig3, ax3 = plt.subplots()
            ax3.plot(precisions, f1_scores, marker='o')
            for i, txt in enumerate(thresholds_plot):
                ax3.annotate(f"{txt:.2f}", (precisions[i], f1_scores[i]), fontsize=8, xytext=(5, 2), textcoords='offset points')
            ax3.set_xlabel("Precision")
            ax3.set_ylabel("F1-Score")
            ax3.set_title("Precision vs F1-Score")
            ax3.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig3)

            y_pred_final = (probs >= melhor_threshold).astype(int)
            data['predicted_fraud'] = y_pred_final
            st.success(f"Predições com threshold **{melhor_threshold:.2f}** salvas na coluna `predicted_fraud`.")

            st.subheader("⬇️ Download dos Resultados")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Baixar arquivo com predições", data=csv, file_name='dados_com_predicoes.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Erro no processamento: {e}")

elif tab == "Sobre Nós":
    st.header("👥 Sobre a Equipe")
    st.markdown("""
    Este projeto foi desenvolvido por estudantes do MBA em Engenharia de Dados.

    **Integrantes do grupo:**
    - 👤 Rodrigo Alex do Nascimento – Engenhario de Dados
    - 👤 Davi Sasso – Engenheiro de Dados
    - 👤 Victor Barradas – Engenheiro de Dados

    **Contato e LinkedIn:**
    - 💼 https://www.linkedin.com/in/rodrigo-alex-nasc/
    - 💼 https://www.linkedin.com/in/davi-sasso-14a706232/   
    - 💼 https://www.linkedin.com/in/victor-barradas/  
                           


    **Mentoria/Professores:** Gustavo Ferreira
    """)
