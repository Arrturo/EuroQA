from langchain_community.embeddings import SentenceTransformerEmbeddings


def embeddings():
    return SentenceTransformerEmbeddings(model_name="ipipan/silver-retriever-base-v1.1", 
                                         model_kwargs={"device": "cpu"})