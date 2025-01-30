import pandas as pd
from spacy.lang.en import English
from tqdm.auto import tqdm

def separador_sentencas(data: list[dict]) -> list[dict]:
   nlp = English()
   nlp.add_pipe("sentencizer")
   """
   De cada linha, separar por sentenças
   """
   for item in tqdm(data):
       item["sentenças"] = list(nlp(item['texto']).sents)
    # certificar que não vai haver caractere especial 
       item['sentenças'] = [str(sentença) for sentença in item['sentenças']]
       item['contador_sentenças'] = len(item['sentenças'])
   return data

def particiona_item(input_list: list[str], 
                  tamanho_particao: int)-> list[list[str]]:
    """
    Retorna sequencias de sentenças, ex. sentença[1] até sentença[10],
    sentença[11] até sentença[20], etc. Dependendo do tamanho da partição
    """

    return [input_list[i:i + tamanho_particao] for i in range(0, len(input_list), tamanho_particao)]

def particionador(data: list[dict], 
                  tamanho_particao: int) -> list[dict]:
    for item in tqdm(data):
        item["chunk_sentença"] = particiona_item(input_list=item["sentenças"],
                                                 tamanho_particao=tamanho_particao)
        item["contador_chunks"] = len(item["chunk_sentença"])

    return data
    
   

if __name__ == "__main__": 
    ## Carregamento em conversão para lista de dicionários
    dataset = pd.read_csv("Dataset/raw_dataset.csv")
    dataset = dataset.to_dict(orient='records')

    ## Separação de sentenças
    dataset = separador_sentencas(data=dataset)
    
    ## Particao de sentenças em chunks
    dataset = particionador(data=dataset,tamanho_particao=10)
    
    ## Salvando o progresso:
    dataset = pd.DataFrame(dataset)
    dataset.to_pickle("Dataset/chuncked_dataset.pkl")