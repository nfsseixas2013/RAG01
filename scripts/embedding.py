from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd

def embedding_chunks(input: list[dict],
                     model: SentenceTransformer) -> list[dict]:
    for item in tqdm(input):
        item["embedding"] = model.encode(item["paragrafo_chunk"],
                                          batch_size = 32,
                                          convert_to_tensor=True)
    dataset = pd.DataFrame(input)
    dataset.to_pickle("Dataset/Embedded_chunks.pkl")
    return input



if __name__ == "__main__":
    # Baixando o modelo de embedding
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")
    
    # Carregando o dataset checkpoint
    dataset = pd.read_csv("Dataset/chunks_e_paginas.csv")

    # Convertendo para lista de dicionários
    dataset = dataset.to_dict(orient="records")

    # Realizando os embeddings e salvando
    _ = embedding_chunks(dataset,embedding_model )
