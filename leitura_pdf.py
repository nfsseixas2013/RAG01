import sys
import fitz
from tqdm.auto import tqdm
import pandas as pd

def formatador_texto(text: str) -> str:

    """
    Removendo excessos de quebra de linha e espaços no fim
    e no começo de cada string resultante
    """
    texto_limpo = text.replace("\n", " ").strip()

    return texto_limpo

def abre_e_ler(pdf_path: str) -> list[dict]:
    try:
        documento = fitz.open(pdf_path)
        paginas_e_texto = []

        """
        Aqui faremos uma leitura por página
        """
        for num_pagina, pagina in tqdm(enumerate(documento)):
            texto = pagina.get_text()
            texto = formatador_texto(text=texto)
            paginas_e_texto.append({
                "numero_pagina": num_pagina + 1,
                "contador_caracteres": len(texto),
                "contador_palavras": len(texto.split(" ")),
                "contador de sentenças": len(texto.split(". ")),
                "contador aprox. token": len(texto)/4, # 1 token aprox 4 caracteres
                "texto": texto
            })
        return paginas_e_texto
    except:
        print("Arquivo não encontrado")
        return [{}]

if __name__ == "__main__":
    dados = abre_e_ler(pdf_path=sys.argv[1])
    dataset = pd.DataFrame(dados)
    dataset.to_csv("Dataset/raw_dataset.csv", index = False)
    


