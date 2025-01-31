from fastapi import FastAPI, HTTPException, APIRouter, status
from pydantic import BaseModel
from app.config import settings
from app.models.input import input_rag
from app.models.output import rag_output
import logging
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import util, SentenceTransformer
import torch
import app.api.v1.rag.retrieval as retrieval
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM



logger = logging.getLogger(__name__)

logger.info(" Carregando os módulos e datasets")
dataset = pd.read_pickle("./app/api/v1/rag/Embedded_chunks.pkl")
    
## carregando o modelo
model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")

## convertendo nossa lista de embeddings para tensor
embeddings = torch.stack(list(dataset["embedding"]))

## Convertendo o dataset para lista de dicionários
dataset = dataset.to_dict(orient="records")

logger.info("Carregando/baixando o modelo LLM")

quantizations_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.float16)
attn_implementation = "sdpa"
model_checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)
use_quantization_config = True
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_checkpoint,
                                                 quantization_config=quantizations_config,
                                                 attn_implementation=attn_implementation,
                                                 )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)



app = APIRouter(tags=["rag"])

def formatador_prompt(query: str,
                      context_items: list[dict])-> str:
    context = "- "+ "\n- ".join(item["paragrafo_chunk"] for item in context_items)

    prompt = f"""

    Based on the following context items, please answer the query:
    Context items:
    {context}

    Query: {query}

    Answer:

    """
    return prompt

def prompting(query: str):
    _, indices = retrieval.busca_contexto_relevante(query=query,
                                               embeddings=embeddings,
                                               model=model,
                                               numero_de_contextos=10)
    context_items = [dataset[i] for i in indices]
    return formatador_prompt(query=query,context_items=context_items)

@app.post("/rag", response_model = rag_output)
async def rag_posting(msg: input_rag):
    logger.info("criando prompt")
    prompt = prompting(query=msg.input)

    logger.info("")
    inputs_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(**inputs_ids,
                             temperature = 0.5,
                             do_sample= True,
                             max_new_tokens = 512)
    output_text = tokenizer.decode(outputs[0])
    return rag_output(output = output_text)

@app.post("/search", response_model = rag_output)
async def rag_search(msg: input_rag):

    logger.info("Processando resposta")
    inputs_ids = tokenizer(msg.input, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(**inputs_ids,
                             temperature = 0.5,
                             do_sample= True,
                             max_new_tokens = 1024)
    
    output_text = tokenizer.decode(outputs[0])
    return rag_output(output = output_text)



