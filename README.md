# Detecção de fraude em cartão de crédito em 34,6 milhões de transações

Projeto de ciência de dados de ponta a ponta: análise exploratória, modelo de classificação, app Streamlit e uma
esteira de MLOps com Airflow, MLflow e Docker. O objetivo é **pontuar cada transação na hora da autorização** para
que o time de fraude revise só as de maior risco.

> **Os dados são sintéticos.** Há sinais claros disso (nenhuma fraude acima de ~US$ 1.250, clientes de 16 e 17 anos com
> cartão, ciclo semanal de compras idêntico em novembro). Os números deste projeto **não** são estimativas do mundo real.

## Sumário

1. [O problema](#1-o-problema)
2. [Resultados em resumo](#2-resultados-em-resumo)
3. [Os dados](#3-os-dados)
4. [Insights da análise exploratória](#4-insights-da-análise-exploratória)
5. [Modelagem](#5-modelagem)
6. [App Streamlit](#6-app-streamlit)
7. [MLOps: Airflow, MLflow e Docker](#7-mlops-airflow-mlflow-e-docker)
8. [Como executar](#8-como-executar)
9. [Estrutura do projeto](#9-estrutura-do-projeto)
10. [Limitações e próximos passos](#10-limitações-e-próximos-passos)
11. [Tecnologias e licença](#11-tecnologias-e-licença)

---

## 1. O problema

Fraude em cartão causa perda direta (estornos e reembolsos), custo de revisão manual e perda de confiança do cliente.
É um evento **raro** (cerca de 0,55% das transações), então regras fixas e revisão manual de tudo não escalam.

O negócio precisa de um modelo que dê uma nota a cada transação e equilibre dois custos:

- **Falso negativo:** uma fraude que passa custa o valor inteiro da compra.
- **Falso positivo:** uma compra legítima bloqueada ou enviada para revisão custa atrito, atendimento e risco de perder o cliente.

**Métrica principal: PR-AUC.** Prever "não é fraude" para tudo já dá ~99,5% de acurácia, então acurácia e ROC-AUC sozinhas
enganam. A decisão final usa o **custo**: valor das fraudes que passaram + US$ 10 por falso alarme (suposição).

## 2. Resultados em resumo

Modelo atual: **Random Forest** (200 árvores, profundidade 6) com limiar de decisão **0,395**, escolhido por custo.
Teste: 1.358.891 transações (14/08 a 23/12/2020), 7.042 fraudes (0,52%).

| Cenário no teste | Fraudes detectadas | Perdidas | Falsos alarmes | Custo (US$) |
|---|---:|---:|---:|---:|
| Sem modelo | 0 | 7.042 | 0 | 3.810.073 |
| Random Forest original, corte 0,5 | 4.985 | 2.057 | 120.011 | 1.446.207 |
| Ajustado, corte 0,5 | 4.843 | 2.199 | 102.030 | 1.290.984 |
| **Ajustado, corte por custo (0,395)** | **4.997** | **2.045** | **108.357** | **1.298.802** |

- Detecta cerca de **71%** das fraudes, com precisão de **~4%** (cerca de 22 falsos alarmes por fraude detectada).
- Reduz o custo em **~66%** frente a não ter modelo.
- No teste, o corte por custo (escolhido na validação) ficou 0,6% acima do custo do corte 0,5: a diferença é pequena.
- O **valor da compra carrega quase todo o sinal**: usar só `amt` como nota já dá PR-AUC de 0,1526 no teste,
  contra 0,2835 do Random Forest, e 0,3366 contra 0,3625 na validação.

## 3. Os dados

- **Base:** `credit_card_fraud.csv`, ~10,7 GB, **34.636.378 transações** de dezembro de 2018 a dezembro de 2020. O arquivo
  fica em `input/` e não vai para o git.
- **Taxa de fraude:** 0,548%. A perda total soma perto de US$ 100 milhões.
- **Rótulos incompletos no fim:** o último dia com fraude registrada é 25/12/2020, então a análise e o modelo usam só dados até
  23/12/2020 (33.974.423 transações). Sem esse corte, o teste ficaria com rótulos faltando.
- **Privacidade:** a base tem colunas pessoais (`ssn`, `cc_num`, nome, endereço). O modelo e o app **nunca as leem**.
- **Variáveis do modelo (9):** `amt`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long` (numéricas) e `category`,
  `gender`, `state` (categóricas). Não há variáveis criadas.

## 4. Insights da análise exploratória

Notebook: [`notebook/Credit_Card_Fraud_Mega___Exploratory_Data_Analysis.ipynb`](notebook/Credit_Card_Fraud_Mega___Exploratory_Data_Analysis.ipynb)
(a versão exportada está em [`py/`](py/)).

### 4.1 A fraude é rara em quantidade, mas pesa em dinheiro

![Fatia de fraude: quantidade versus valor](img/output.png)

São **0,55% das transações**, mas **4,15% do valor** movimentado: o ticket médio da fraude é cerca de **7,5 vezes** o ticket geral.
Perder pouca quantidade ainda significa perder muito dinheiro.

### 4.2 O valor da compra é o sinal mais forte

![Valores extremos: parcela das linhas versus parcela das fraudes](img/output2.png)

Só ~6,5% das linhas têm valor extremo em `amt`, mas elas reúnem **~76% das fraudes**. Em população da cidade os dois lados
são parecidos, e em latitude e longitude não há diferença. Por faixa de valor, a taxa de fraude é:

| Valor da compra | Taxa de fraude |
|---|---:|
| Abaixo de US$ 250 | perto de zero |
| US$ 250 a 500 | ~7% |
| US$ 500 a 1.000 | ~23% |
| US$ 1.000 a 5.000 | ~21% |

Essas três faixas reúnem **3% das transações e cerca de três quartos das fraudes**.

![Valores extremos antes e depois do tratamento](img/output3.png)

A mediana da fraude fica perto de US$ 450, contra ~US$ 50 nas compras legítimas. O gráfico mostra o antes e depois do tratamento
de valores extremos: só `amt` e `city_pop` foram tratadas (escala log); as coordenadas não precisam.

### 4.3 A perda está concentrada em poucos comerciantes

![Concentração da perda por comerciante](img/output5.png)

Entre 693 comerciantes, os **10% do topo respondem por 41% da perda**, e cerca de 28% deles respondem por 82%. Acompanhar
os comerciantes do topo da curva cobre a maior parte do dinheiro perdido.

### 4.4 Categorias: perda e risco não coincidem

- **Quatro categorias concentram ~80% da perda:** `shopping_net`, `shopping_pos`, `misc_net` e `grocery_pos`.
- O **maior risco** está em `shopping_net` (1,7%), `misc_net` (1,6%) e `grocery_pos` (1,3%).
- `shopping_pos` é a segunda em perda, mas tem risco de 0,7%: o dinheiro perdido ali vem do **volume** de compras, não de uma chance maior.

### 4.5 Quem: idade importa, gênero quase não

![Taxa de fraude por faixa etária e gênero](img/output1.png)

Clientes de **45 anos ou mais** têm taxa cerca de **duas vezes maior** (0,7% a 0,8%, contra 0,35% a 0,44% entre 25 e 44 anos).
A diferença entre gêneros é menor que 0,1 ponto percentual. A idade pode refletir o perfil de compra e não ser o fator em si.

### 4.6 Quando: a fraude tem ritmo próprio e não acompanha o volume

![Fraude e volume de compras ao longo do tempo](img/output6.png)

São de 7.300 a 8.400 fraudes por mês, enquanto o volume de compras varia de 0,9 a 2,6 milhões por mês e **dobra em dezembro**.
A taxa de fraude cai em dezembro (~0,3%) e sobe em janeiro (~0,8%) **só porque o denominador muda**, sem que a fraude tenha mudado.
Por isso a taxa deve ser lida junto com o volume, e o acompanhamento deve usar a quantidade e o valor das fraudes.

![Compras e fraudes por dia, novembro e dezembro de 2020](img/output7.png)

No dia a dia, as compras têm um ciclo semanal idêntico (de ~27 mil na segunda a ~60 mil nos fins de semana, e o ciclo dobra em dezembro),
enquanto as fraudes ficam em torno de **250 por dia** em qualquer dia. Também aparece o **fim dos rótulos em 25/12/2020**:
depois disso as compras seguem, mas a contagem de fraudes cai a zero. É o motivo do corte em 23/12.

### 4.7 Outros achados

- **Janela de risco no fim do dia:** a taxa fica perto de 0,1% da hora 0 à 15, chega a ~2,5% na hora 17 e volta a 0,1% na hora 23.
  O padrão não vem do volume. As horas estão em UTC e ainda falta confirmar contra o horário original.
- **A distância cliente–comerciante não separa fraude de compra legítima:** a mediana é de ~79 km nos dois grupos.
- **Latitude e longitude** do cliente e do comerciante têm correlação de 0,99 a 1,00 entre si.

### 4.8 Recomendações para o time de fraude

1. **Priorizar a revisão por valor.** Revisar toda compra entre US$ 250 e 5.000 cobriria ~75% das fraudes, mas com precisão de ~13%
   (1 fraude a cada 8 compras revisadas). A regra por valor serve para ordenar a fila, e o modelo refina quais revisar primeiro.
2. **Monitorar quantidade e valor de fraudes**, e usar a taxa só junto com o volume.
3. **Dimensionar a equipe pela carga estável** (~8 mil casos por mês), não pelo calendário de compras.
4. **Acompanhar os comerciantes do topo** da curva de concentração.
5. **Não investir em regras de distância:** neste dataset ela não distingue fraude de compra legítima.

## 5. Modelagem

Notebook: [`notebook/Credit_Card_Fraud_Mega___Model_ML.ipynb`](notebook/Credit_Card_Fraud_Mega___Model_ML.ipynb).

### 5.1 Método

1. **Amostra de 20%:** uma a cada 5 transações (por hash), o que mantém a taxa real de fraude.
2. **Divisão cronológica, sem embaralhar:** teste = últimos 20% da amostra; do restante, os últimos 20% viram validação.
   Assim o teste vem depois do treino no tempo, como seria em produção. (Um mesmo cliente pode aparecer em várias partes.)
3. **Subamostragem 1:10 só no ajuste** (10 legítimas para cada fraude, 9,1% de fraude). Validação e teste mantêm a taxa real.
4. **10 algoritmos comparados** com parâmetros fixos (Regressão Logística, Árvore, Random Forest, Extra Trees, LightGBM,
   HistGradientBoosting, XGBoost, XGBoost RF, CatBoost, Naive Bayes).
5. **Busca aleatória** de hiperparâmetros (nota = PR-AUC na validação) e **limiar por custo** na validação.
6. **Avaliação no teste**, que não participa de nenhuma escolha.

### 5.2 O que os modelos mostraram

- O **Random Forest liderou** a PR-AUC no teste (**0,2835**, cerca de 54 vezes a linha aleatória de 0,0052). Naive Bayes (0,1523) e
  Regressão Logística (0,1512) vieram em seguida.
- **O valor sozinho como nota já dá 0,1526**: os modelos simples só repetem essa regra. Qualquer modelo novo precisa superar esse número.
- **O corte importa tanto quanto o modelo.** Com o corte padrão de 0,5 e o treino com subamostragem, as notas ficam deslocadas e o
  número de alertas muda muito entre modelos. Por isso o limiar vem do custo, não de 0,5.
- **Coordenadas e população importam pouco** para o Random Forest e mais para os modelos de boosting.

### 5.3 SMOTENC foi testado e não adotado

![Tratamento da classe alvo com SMOTENC](img/output4.png)

O SMOTENC elevou a fraude de 9,1% para 33,3% do treino (143 mil reais + 574 mil sintéticas), e a distribuição de valores das sintéticas imita
a das reais. Mas a PR-AUC na validação foi de apenas **0,0091**, perto da linha aleatória (0,0053). O modelo final usa a
subamostragem 1:10.

### 5.4 Limiar por custo

O limiar sai do menor custo na validação (US$ 10 por falso alarme): **0,395**. Mudar esse custo muda o limiar e a economia, e o
app tem um painel lateral para testar outros valores.

## 6. App Streamlit

```bash
streamlit run app/streamlit_app.py
```

| Aba | O que faz |
|---|---|
| **Transação única** | Preencha os 9 campos e veja o score de risco, a decisão (alerta ou liberar), o medidor com o limiar e como o valor da compra muda o score |
| **Análise em lote (CSV)** | Envie um CSV (até 300.000 linhas) e veja alertas, custo de revisão, o **mapa por local do cliente**, a distribuição dos scores e as transações de maior risco. Com a coluna `is_fraud`, mede recall, precisão e custo |
| **Sobre o modelo** | Problema, método, resultados no teste e limitações |

- **Mapa:** dois modos. *Score de risco (0 a 1)* funciona com qualquer CSV. *Resultado: fraude × falso alarme* (adaptado do notebook)
  exige `is_fraud`: laranja = fraude detectada, cinza = falso alarme, e o tamanho da bolha é o número de alertas no local.
- **Base local:** se `input/credit_card_fraud.csv` existe, um botão sorteia N transações da base (só as colunas do modelo e `is_fraud`).
- **Validação do upload:** linhas inválidas são descartadas e listadas; o app avisa sobre categorias que o modelo não conhece.
- **Deploy:** funciona no Streamlit Community Cloud (arquivo principal `app/streamlit_app.py`, Python 3.12). Detalhes em [doc/app-streamlit.md](doc/app-streamlit.md).

> O score **não é uma probabilidade calibrada**: o treino usou subamostragem. Ele serve para ordenar o risco.

## 7. MLOps: Airflow, MLflow e Docker

```
 input/credit_card_fraud.csv
        │
        ▼
 Airflow: validar_base → converter_parquet → preparar_amostra → dividir_dados → treinar_avaliar → promover
                                                                                 │                  │
                                                                                 ▼                  ▼
                                                                    MLflow (métricas, modelo)   mlflow/producao/modelo_final.joblib
                                                                    registro + alias champion            │
                                                                                                         ▼
                                                                                                   App Streamlit
```

- **DAG** `fraude_cartao_pipeline` ([`airflow/dags/fraude_pipeline.py`](airflow/dags/fraude_pipeline.py)): reproduz o método do notebook, é disparada
  manualmente e tem parâmetros (iterações da busca, custo do falso alarme, tamanho da amostra, regra de promoção).
- **MLflow:** registra parâmetros, métricas de validação e de teste, o modelo e um pacote no formato do app. Cada versão vai para o
  registro `fraude_cartao_random_forest`, e o alias `champion` aponta para a versão em produção.
- **Promoção:** o novo modelo só vira campeão se tiver **menor custo no teste** que o atual. Quando promove, publica o pacote em
  `mlflow/producao/` e o app passa a usá-lo.
- **Docker Compose:** Postgres, MLflow, Airflow (webserver e scheduler) e o app, em [`docker/`](docker/).

Portas: Airflow `8080`, MLflow `5000`, app `8501`. Passo a passo em [doc/guia-docker.md](doc/guia-docker.md).

## 8. Como executar

### 8.1 Só o app

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

`requirements.txt` fixa o `scikit-learn==1.5.2`, a série que salvou o modelo. Outra versão carrega o modelo com aviso e pode divergir.

### 8.2 Os notebooks

Coloque `credit_card_fraud.csv` em `input/` e abra os notebooks em `notebook/`. Eles usam caminhos relativos à raiz do projeto.
**Atenção:** rodar o notebook de modelagem inteiro **sobrescreve** os arquivos em `models/`, inclusive `modelo_final.joblib`, que o app usa.
Para retreinar sem tocar em `models/`, use o executor `executar_notebook.py`, que grava as saídas em `retreino/`.

### 8.3 Airflow, MLflow e app no Docker

```bash
cd docker
cp .env.example .env              # edite as senhas e a AIRFLOW_FERNET_KEY
docker compose up --build
```

Depois abra o Airflow em http://localhost:8080, dispare a DAG `fraude_cartao_pipeline` e acompanhe em http://localhost:5000.

### 8.4 Deploy do app no Streamlit Community Cloud

1. Suba para o GitHub o que o app precisa (o `.gitignore` já bloqueia o CSV de ~10 GB e os modelos pesados).
2. Em [share.streamlit.io](https://share.streamlit.io), crie o app com a branch `main` e o arquivo principal `app/streamlit_app.py`.
3. Em **Advanced settings**, selecione **Python 3.12**.

## 9. Estrutura do projeto

```
.
├── app/streamlit_app.py            # interface (3 abas)
├── src/
│   ├── modelo.py                   # carga do modelo, validação de lote, predição, métricas
│   ├── mapa.py                     # mapas de bolhas por local do cliente
│   └── pipeline_treino.py          # passos do pipeline de treino (usados pela DAG)
├── airflow/dags/fraude_pipeline.py # DAG do Airflow
├── docker/                         # docker-compose, imagens do Airflow e do app, .env.example
├── mlflow/                         # imagem do servidor MLflow e producao/ (modelo campeão)
├── notebook/                       # EDA e modelagem
├── py/                             # EDA exportado como .py
├── models/modelo_final.joblib      # modelo final (os demais arquivos de models/ ficam só na sua máquina)
├── output/transacoes_novas.csv     # 5.000 transações de exemplo para o app
├── input/                          # base de dados (não vai para o git)
├── img/                            # gráficos usados neste README
├── doc/                            # documentação detalhada
└── requirements.txt
```

Documentação detalhada: [doc/README.md](doc/README.md).

## 10. Limitações e próximos passos

**Limitações**

- **Dados sintéticos:** os padrões são mais nítidos que no mundo real, e os números não valem como estimativa real.
- **Uma única divisão dos dados e uma semente fixa**, com busca de só 8 combinações. Modelos retreinados variam: uma versão
  anterior (treinada no Colab, 300 árvores, limiar 0,85) teve 31.314 falsos alarmes contra 108.357 do atual, com quantidade parecida
  de fraudes detectadas.
- **O teste foi consultado mais de uma vez** na comparação dos modelos.
- **US$ 10 por falso alarme é uma suposição**, e limiar e economia dependem dela.
- **Os dados mostram associação, não causa.**

**Próximos passos**

- Confirmar o fuso das horas (UTC) e testar a hora e a faixa de valor como variáveis do modelo.
- Validação cruzada no tempo e busca de hiperparâmetros maior.
- Calibrar as notas para obter probabilidades.
- Agendar a DAG e monitorar a deriva dos dados.
- Testar a esteira completa no Docker e publicar as imagens.

## 11. Tecnologias e licença

Python 3.12, polars, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn, Plotly, Streamlit, Apache Airflow,
MLflow, Docker e PostgreSQL.

Licença [MIT](LICENSE). Autor: Rafael Gallo.
