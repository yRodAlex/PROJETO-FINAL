
# SecureTrust - Detecção de Fraudes Bancárias com Engenharia e Ciência de Dados

🚀 Projeto acadêmico focado em uma solução ponta a ponta para prevenção de fraudes digitais em uma fintech global de microcrédito.

---

## 🏢 Contexto do Projeto

A **SecureTrust** atua no setor de microcrédito digital em países emergentes, com operações 100% online. Devido à rapidez no processo de concessão de crédito, a empresa enfrenta sérios desafios com fraudes digitais, como:

- Roubo de identidade  
- Criação de contas falsas  
- Solicitações fraudulentas em massa  

As perdas estimadas ultrapassam US$ 1 milhão ao ano, motivando o desenvolvimento desta solução de detecção e prevenção de fraudes baseada em dados.

---

## 🎯 Solução Proposta

Este projeto simula uma entrega real para a SecureTrust, incluindo:

✅ Pipeline de dados em ambiente AWS  
✅ Tratamento e estruturação dos dados no S3 (camadas BRONZE, PRATA e OURO)  
✅ Modelo preditivo de detecção de fraudes (XGBoost)  
✅ Dashboard interativo via Streamlit para análise de métricas e visualizações  

---

## 🗂️ Estrutura de Dados - Data Lake

**Bucket S3:** `securetrust-bucket`

```
s3://securetrust-bucket/sistemadeorientacaodecredito/
    ├── BRONZE/    # Dados brutos originais
    ├── PRATA/     # Dados tratados e limpos
    └── OURO/      # Dados finais para BI e modelos
```

Tabela consultável no Athena:

```sql
CREATE DATABASE IF NOT EXISTS securetrust_db;

CREATE EXTERNAL TABLE IF NOT EXISTS securetrust_db.base_tratada (... campos ...)
LOCATION 's3://securetrust-bucket/sistemadeorientacaodecredito/PRATA/';
```

---

## ⚙️ Tecnologias Utilizadas

- **AWS S3, Glue e Athena**  
- **Python, Pandas, Numpy, Scikit-Learn, Imblearn, XGBoost**  
- **Streamlit** para interface e visualização  
- **SMOTEENN** para balanceamento de classes  

---

## 🛠️ Funcionalidades do Protótipo

✅ Upload e pré-visualização dos dados  
✅ Pipeline de preparação (codificação, tratamento de nulos)  
✅ Balanceamento das classes com SMOTEENN  
✅ Treinamento do modelo XGBoost  
✅ Avaliação por Threshold (Precision, Recall, F1-Score, Accuracy)  
✅ Comparação visual: Falsos Positivos x Falsos Negativos  
✅ Visualização da relação entre Precision e F1-Score  
✅ Download do arquivo final com predições  

---

## 📊 Exemplo de Insights Esperados

- Perfis mais comuns de fraude  
- Dispositivos ou sistemas operacionais com maior incidência de fraudes  
- Relação entre similaridade de nome/e-mail e risco de fraude  
- Tempo médio até a ocorrência da fraude após criação da conta  

---

## 👥 Equipe

Projeto desenvolvido no MBA em Engenharia de Dados:

- **Rodrigo Alex do Nascimento** – [LinkedIn](https://www.linkedin.com/in/rodrigo-alex-nasc/)  
- **Davi Sasso** – [LinkedIn](https://www.linkedin.com/in/davi-sasso-14a706232/)  
- **Victor Barradas** – [LinkedIn](https://www.linkedin.com/in/victor-barradas/)  

**Mentoria:** Gustavo Ferreira

---

## 📁 Requisitos para Execução

```bash
pip install -r requirements.txt
```

**Bibliotecas principais:**  
`streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imblearn`, `boto3`, `matplotlib`

Execute o app com:

```bash
streamlit run app.py
```

---

## 📦 Próximos Passos (Evolução Recomendada)

- Integração direta com o S3 para leitura e gravação de dados  
- Versionamento dos datasets no Data Lake (controle BRONZE → PRATA → OURO)  
- Monitoramento em tempo real via Streamlit ou BI (ex: Tableau)  
- Implementação de explicabilidade do modelo com SHAP  
