import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma/books"

PROMPT_TEMPLATE = """
Answer the question based only on the following context: 

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Please enter your question.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the db
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search the db
    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )
    if len(results) == 0: #or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    llm = ChatOpenAI()
    response_text = llm.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print("====================================================")
    print(formatted_response)
    print("====================================================")

if __name__ == "__main__":
    main()