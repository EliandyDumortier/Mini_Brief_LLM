#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import chromadb
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

# ── CONFIG ────────────────────────────────────────────────────────────────
PDF_DIR        = "data"                      # dossier contenant Menu.pdf et Liste_allergenes.pdf
CHROMA_DB_DIR  = "./chroma_db"
COLLECTION     = "bella_napoli_pizza"
MODEL_NAME     = "mistral:latest"
os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

# ── (Re)création du dossier ChromaDB pour éviter les collisions d'embed dims ─
if os.path.exists(CHROMA_DB_DIR):
    shutil.rmtree(CHROMA_DB_DIR)

# ── 1) Chargement des PDF avec metadata « source » ───────────────────────────
menu_loader      = PyPDFLoader(os.path.join(PDF_DIR, "Menu.pdf"))
allergy_loader   = PyPDFLoader(os.path.join(PDF_DIR, "Liste_allergenes.pdf"))

docs_menu    = menu_loader.load()    # liste de Document(page_content, metadata)
docs_allergy = allergy_loader.load()

# On tagge chaque document avec la source pour pouvoir citer si besoin
for doc in docs_menu:
    doc.metadata["source"] = "menu"

for doc in docs_allergy:
    doc.metadata["source"] = "allergenes"

# Combine les deux jeux de docs
docs = docs_menu + docs_allergy

# ── 2) Découpage en chunks pour la recherche ────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks   = splitter.split_documents(docs)

# ── 3) Index vectoriel ChromaDB ─────────────────────────────────────────────
client     = chromadb.PersistentClient(path=CHROMA_DB_DIR)
embeddings = OllamaEmbeddings(model=MODEL_NAME)

vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    client=client,
    collection_name=COLLECTION
)

# ── 4) PromptTemplate intelligent ────────────────────────────────────────────
template = """
Vous êtes l’assistant de Bella Napoli.
Vous disposez de deux sources :
- le MENU (source="menu") contenant les noms de pizzas et leurs ingrédients,
- la LISTE_ALLERGENES (source="allergenes") listant chaque allergène et son numéro.

1. Si la question porte sur les **ingrédients** d’une pizza X, répondez :
   “Ingrédients de la pizza X : ….”

2. Si la question porte sur les **allergènes** d’une pizza X, répondez :
   “Allergènes de la pizza X (codes) : ….”
   Si possible, donnez aussi le nom de l’allergène entre parenthèses.

3. Si la pizza X n’existe pas dans le MENU, répondez :
   “D’après la documentation Bella Napoli, il n’existe pas de pizza « X » dans le menu fourni.”

4. N’ajoutez pas d’informations non présentes dans les documents.

Contexte (extraits les plus pertinents) :
{context}

Question :
{question}

Réponse :
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template.strip()
)

# ── 5) Construction de la chaîne RAG ────────────────────────────────────────
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm       = ChatOllama(model=MODEL_NAME)
qa_chain  = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

# ── 6) Fonction d’interrogation Gradio ─────────────────────────────────────
def answer_question(full_question: str) -> str:
    # On extrait le nom de la pizza
    m = re.search(r"pizza\s+(.+?)[\?\.]?$", full_question, re.IGNORECASE)
    pizza_name = m.group(1).strip() if m else full_question.strip()
    # On construit la question pour le LLM
    query = full_question.replace(m.group(1), pizza_name) if m else full_question
    # On interroge la RAG
    result = qa_chain.invoke({"query": query})
    return result["result"]

# ── 7) Lancement Gradio ─────────────────────────────────────────────────────
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ex : Quels ingrédients contient la pizza Margherita ?"),
    outputs="text",
    title="🍕 Assistant Bella Napoli",
    description="Posez vos questions sur les ingrédients ou les allergènes des pizzas."
)

if __name__ == "__main__":
    iface.launch(share=True)
