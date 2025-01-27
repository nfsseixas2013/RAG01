import pandas as pd
import re
from tqdm.auto import tqdm

def chunks_e_paginas(data: list[dict]) -> list[dict]:
    """
    A ideia aqui é agrupar cada chunk por página
    """
    paginas_e_chunks = []
    for item in tqdm(data):
        for sentences in item["chunk_sentença"]:
            chunk = {}
            chunk["pagina_do_chunk"] = item["numero_pagina"]

            # Juntar as sentenças como parágrafos
            paragrafo = "".join(sentences).replace("  ", " ").strip()
            chunk['paragrafo_chunk'] = paragrafo

            ## Vamos adicionar um número aproximados de tokens por chunks:
            chunk['numero_de_tokens'] = len(paragrafo)/4

            paginas_e_chunks.append(chunk)
    return paginas_e_chunks

if __name__ == "__main__":
    # Carregando o dataset em pickle
    dataset = pd.read_pickle("Dataset/chuncked_dataset.pkl")
    
    # convertendo para lista de dicionários para operações
    dataset = dataset.to_dict(orient="records")

    # Criando o dataset de chunks e páginas

    chunks_paginados = chunks_e_paginas(data=dataset)

    # Salvando o dataset final

    final_data = pd.DataFrame(chunks_paginados)
    final_data.to_csv("Dataset/chunks_e_paginas.csv", index = False)

    