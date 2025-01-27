import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import util, SentenceTransformer
import torch
import textwrap


def busca_contexto_relevante(query: str,
                             embeddings: torch.tensor,
                             model: SentenceTransformer,
                             numero_de_contextos: int = 10,
                             ):
    ## embed o query:
    query_embedded = model.encode(query, convert_to_tensor=True)

    ## retornar o produto escalar
    dot_scores = util.dot_score(query_embedded,embeddings)[0]

    scores, indices = torch.topk(dot_scores, k=numero_de_contextos)

    return scores, indices

def print_wrapped(text, wrap_length = 80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def imprime_resultados(query: str,
                       embeddings: torch.tensor,
                       embedded_chunks: list[dict],
                       model: SentenceTransformer,
                       resources_to_return: int):
    
    # Busca pelas passagens relevantes
    scores, indices = busca_contexto_relevante(query=query,
                                               embeddings=embeddings,
                                               model=model,
                                               numero_de_contextos=resources_to_return)
    print(f"A pergunta foi: \n {query}")
    
    for score,idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Texto")
        print_wrapped(embedded_chunks[idx]["paragrafo_chunk"])
        print(f"Número da página: {embedded_chunks[idx]['pagina_do_chunk']}")
        print("\n")

    
## Para teste

if __name__ == "__main__":
    ## carregando dataset
    dataset = pd.read_pickle("Dataset/Embedded_chunks.pkl")
    
    ## carregando o modelo
    model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")

    ## convertendo nossa lista de embeddings para tensor
    embeddings = torch.stack(list(dataset["embedding"]))

    ## Convertendo o dataset para lista de dicionários
    dataset = dataset.to_dict(orient="records")

    ## Testando::
    imprime_resultados(query="What are the V's of big data?",
                       embeddings=embeddings,
                       embedded_chunks=dataset,
                       model=model,
                       resources_to_return=10)




    

    
