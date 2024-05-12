import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

from embedding import embeddings
from path import DB_PATH


PROMPT_TEMPLATE = """
        Odpowiedz na pytania na podstawie poniższego tekstu w Context, jeśli nie znalazłeś odpowiedzi na
        pytanie w tekście, napisz "Nie posiadam wiedzy na ten temat." Odpowiadaj tylko w języku Polskim, a nie w języku angielskim. 
        Nie pisz gdzie znalazłeś daną informację.

        Context: {context}

        Question: {question}
"""

examples = [
            ["Skąd wzięła się nazwa Europa?"],
            ["Co jest stolicą Polski?"], 
            ["Jak nazywa się hymn Francji?"],
            ["Jaką powierzchnię ma San Marino?"],
            ["Gdzie leżą Czechy?"],
            ["Wymień 5 popularnych muzyków pochodzących z Austrii"],
            ["Do jakich języków należy język litewski?"],
            ["Historia Finlandii"]
            ]

def query(query_text: str) -> str:
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings())
    results = db.similarity_search_with_score(query_text, k=10)

    context = [doc.page_content for doc, _ in results]
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context, question=query_text)

    model = Ollama(model="gemma")

    return model.invoke(prompt)

def main():
    demo = gr.Interface(fn=query, 
                        inputs="text", 
                        outputs="text",
                        title="EuroQA",
                        description="""
                                        EuroQA to system odpowiadania na pytania, który wykorzystuje model językowy 
                                        do odpowiadania na pytania w oparciu o dany kontekst.
                                    """,
                        examples=examples
                        )
    demo.launch(share=False)

if __name__ == '__main__':
    main()