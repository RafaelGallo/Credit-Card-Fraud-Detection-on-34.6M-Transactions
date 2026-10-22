"""Executa o notebook de modelagem localmente, adaptado do Colab.

O notebook original (notebook/) e os modelos atuais (models/) não são alterados:
- caminhos do Colab (/content, Drive) viram caminhos locais;
- cells de pip install, download do Kaggle, unzip e montagem do Drive são ignoradas;
- tudo que o notebook grava vai para retreino/ (dados, modelos, resultados, modelo_final).

Uso (na raiz do repositório):  python retreino/executar_notebook.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "retreino"
ORIGINAL = RAIZ / "notebook" / "Credit_Card_Fraud_Mega___Model_ML.ipynb"
EXECUTADO = SAIDA / "Model_ML_executado.ipynb"
LOG = SAIDA / "progresso.log"

CSV = (RAIZ / "input" / "credit_card_fraud.csv").as_posix()
SUBSTITUICOES = [
    ("/content/data/credit_card_fraud.csv", CSV),
    ("/content/credit_card_fraud_full.parquet", (SAIDA / "dados" / "credit_card_fraud_full.parquet").as_posix()),
    ("/content/drive/MyDrive/fraude_cartao/modelos", (SAIDA / "modelos").as_posix()),
    ("/content/drive/MyDrive/fraude_cartao/resultados", (SAIDA / "resultados").as_posix()),
    ("/content/drive/MyDrive/fraude_cartao/modelo_final", (SAIDA / "modelo_final").as_posix()),
    ("/content/transacoes_novas.csv", (SAIDA / "transacoes_novas.csv").as_posix()),
    ('shutil.disk_usage(r"/content")', f'shutil.disk_usage(r"{SAIDA.as_posix()}")'),
    ("shutil.disk_usage(r'/content')", f"shutil.disk_usage(r'{SAIDA.as_posix()}')"),
    ("/content/data", (SAIDA / "dados").as_posix()),
]
# linhas que dependem do Colab ou instalam pacotes
LINHAS_IGNORADAS = re.compile(r"google\.colab|drive\.mount|userdata\.get|\"-m\", \"pip\", \"install\"")
CELULAS_IGNORADAS_POR_CONTEUDO = ("kaggle", "datasets", "download")  # cell do download do Kaggle
UNZIP = "zipfile.ZipFile(zip_path)"


def adaptar(codigo: str) -> str:
    if all(t in codigo for t in CELULAS_IGNORADAS_POR_CONTEUDO) or UNZIP in codigo:
        return "pass  # ignorado no local: o CSV já está em input/"
    for antigo, novo in SUBSTITUICOES:
        codigo = codigo.replace(antigo, novo)
    linhas = [("pass  # ignorado no local" if LINHAS_IGNORADAS.search(l) else l) for l in codigo.splitlines()]
    return "\n".join(linhas)


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def main() -> int:
    for pasta in ("dados", "modelos", "resultados", "modelo_final"):
        (SAIDA / pasta).mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(ORIGINAL, as_version=4)
    for celula in nb.cells:
        if celula.cell_type == "code":
            celula.source = adaptar(celula.source)
            celula.outputs, celula.execution_count = [], None
    total = len(nb.cells)
    inicio = time.time()
    log(f"inicio: {total} celulas")

    def ao_terminar(cell, cell_index, execute_reply):  # chamado após cada célula
        log(f"celula {cell_index + 1}/{total} ok ({(time.time() - inicio) / 60:.1f} min)")
        if cell_index % 5 == 0:
            nbformat.write(nb, EXECUTADO)

    cliente = NotebookClient(
        nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}},
        on_cell_executed=ao_terminar,
    )
    try:
        cliente.execute()
        log("FIM: todas as celulas executadas")
        return 0
    except Exception as erro:  # a célula que falhou fica com o erro gravado no notebook
        log(f"ERRO: {type(erro).__name__}: {str(erro)[:600]}")
        return 1
    finally:
        nbformat.write(nb, EXECUTADO)


if __name__ == "__main__":
    sys.exit(main())
