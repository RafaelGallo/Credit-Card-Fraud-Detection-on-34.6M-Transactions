# Credit-Card-Fraud-Detection-on-34.6M-Transactions
Projeto ciencias de dados fraude em cartão de credito 34.6m em transações

## App Streamlit

App que pontua transações de cartão de crédito com o modelo final (Random Forest, limiar de 0,85 escolhido por custo):

- **Transação única:** preencha os 9 campos e veja o score de risco, a decisão (alerta ou liberar) e como o valor da compra muda o score.
- **Análise em lote:** envie um CSV, veja alertas, distribuição dos scores e baixe o resultado. Com a coluna `is_fraud`, o app mede recall, precisão e custo.
- **Sobre o modelo:** problema, método, resultados no teste e limitações.

### Rodar localmente

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> O `requirements.txt` fixa o `scikit-learn==1.6.1`, a mesma versão que salvou o modelo. Com outra versão o `joblib` carrega o modelo com aviso e os resultados podem divergir.

### Deploy no Streamlit Community Cloud

1. Suba para o GitHub apenas o que o app precisa (o `.gitignore` já bloqueia o CSV de ~10 GB e os modelos pesados):
   `app/streamlit_app.py`, `src/modelo.py`, `src/mapa.py`, `requirements.txt`, `.streamlit/config.toml`, `models/modelo_final.joblib` e `output/transacoes_novas.csv`.
2. Em [share.streamlit.io](https://share.streamlit.io), entre com o GitHub e clique em **Create app**.
3. Escolha o repositório, a branch `main` e o arquivo principal `app/streamlit_app.py`.
4. Em **Advanced settings**, selecione **Python 3.12** (o `scikit-learn==1.6.1` não tem pacote para o 3.14, e a versão do Python só muda recriando o app).
5. Clique em **Deploy**. Cada novo `git push` na branch atualiza o app.

### Arquivos do app

| Arquivo | Função |
|---|---|
| `app/streamlit_app.py` | Interface (3 abas) |
| `src/modelo.py` | Carga do modelo, validação, predição e métricas (sem Streamlit) |
| `src/mapa.py` | Mapa de bolhas por local do cliente (adaptado do notebook): score 0 a 1 ou fraude × falso alarme |
| `models/modelo_final.joblib` | Modelo, limiar, codificador e colunas (gerado no notebook de modelagem) |
| `output/transacoes_novas.csv` | 5.000 transações de exemplo para a análise em lote |
| `.streamlit/config.toml` | Tema e limite de upload (50 MB) |
