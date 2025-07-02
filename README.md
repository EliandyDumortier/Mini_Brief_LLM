# 🍕 Bella Napoli AI Assistant

Un assistant IA simple basé sur une architecture RAG (Retrieval-Augmented Generation) et LangChain, pour répondre aux questions sur les pizzas de la pizzeria **Bella Napoli** : ingrédients, allergènes, etc. L’interface interactive est servie via Gradio et le LLM Mistral tourne localement via Ollama.

---

## 📂 Structure du projet

```
Mini_Brief_LLM/
├── data/
│ ├── Menu.pdf
│ └── Liste_allergenes.pdf
├── chroma_db/ # (auto-généré) base ChromaDB persistante
├── llm.py # script principal
├── requirements.txt # dépendances Python
└── README.md # ce fichier

```
---

## ⚙️ Prérequis

1. **Python ≥ 3.8**  
2. **Ollama** installé et configuré :  
   ```bash
   brew install ollama      # ou selon votre OS
   ollama pull mistral:latest
   ollama pull mxbai-embed-large:latest

    Un virtualenv (recommandé) :

    python -m venv .venv
    source .venv/bin/activate

📥 Installation

    Clonez ce dépôt et entrez dans le dossier :

git clone <votre-repo-url>
cd Mini_Brief_LLM

Installez les dépendances Python :

    pip install --upgrade pip
    pip install -r requirements.txt

🛠️ Configuration

    Dossier data/
    Placez vos fichiers PDF :

        Menu.pdf : liste des pizzas et leurs ingrédients.

        Liste_allergenes.pdf : codes et libellés des allergènes.

    Variables d’environnement
    (optionnels, si Ollama tourne sur un host différent)

export OLLAMA_HOST="http://localhost:11434"
export CHROMA_ENABLE_TELEMETRY="false"

Paramètres dans llm.py
Vous pouvez modifier :

    PDF_DIR       = "data"
    CHROMA_DB_DIR = "./chroma_db"
    COLLECTION    = "bella_napoli_pizza"
    MODEL_NAME    = "mistral:latest"            # ou un autre modèle Ollama

🚀 Lancement

    Assurez-vous d’avoir pullé les modèles Ollama :

ollama list
# → doit afficher "mistral:latest" et "mxbai-embed-large:latest"

Lancez l’application :

    python llm.py

    Ouvrez l’URL affichée (par défaut http://127.0.0.1:7860) ou utilisez le lien public si share=True est activé.

🎯 Utilisation

    Questions ingrédients

        Quels ingrédients contient la pizza Margherita ?
        → liste des ingrédients.

    Questions allergènes

        Quels sont les allergènes de la pizza Reine ?
        → codes et noms des allergènes correspondants.

    Pizza inconnue

        Quels ingrédients contient la pizza Truffe d’hiver ?
        → message :

    D’après la documentation Bella Napoli, il n’existe pas de pizza « Truffe d’hiver » dans le menu fourni.

📦 Dépendances principales

langchain
langchain-ollama
langchain-chroma
chromadb
gradio

🔄 Recréation de la base vectorielle

Si vous changez de modèle d’embeddings (dimension différente), supprimez le dossier chroma_db/ avant de relancer :

rm -rf chroma_db/
python llm.py

🔧 Personnalisation

    Changer le modèle LLM : modifiez MODEL_NAME pour un autre tag Ollama.

    Ajuster k (nombre de documents RAG) : dans la création du retriever :

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    PromptTemplate : dans llm.py, éditez template pour guider le style de réponse.

📝 Licence

Projet libre, à adapter et réutiliser selon vos besoins pédagogiques ou professionnels.