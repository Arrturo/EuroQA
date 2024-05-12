from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

from embedding import embeddings
from load_data import load_documents, split_documents
from path import DB_PATH


def add_documents_to_db(documents: list[Document]):
    chunks = add_chunks_ids(documents)
    chunk_ids = [chunk.metadata["id"] for chunk in chunks]
    Chroma.from_documents(chunks, ids=chunk_ids, embedding=embeddings(), persist_directory=DB_PATH)

def add_chunks_ids(chunks):
    _ = ''
    chunk_num = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page = f"{source}:{page}"

        if current_page != page:
            chunk_num += 1
        else:
            chunk_num = 0

        current_id = f"{current_page}:{chunk_num}"
        _ = current_page

        chunk.metadata["id"] = current_id

    return chunks


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_documents_to_db(chunks)

if __name__ == '__main__':
    main()