from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#
from groq import Groq
from langchain_groq import ChatGroq
#
import joblib
import os
import nest_asyncio  # noqa: E402
import streamlit as st
nest_asyncio.apply()

llamaparse_api_key = os.environ.get("llamaparse_api_key")
groq_api_key = os.environ.get("groq_api_key")
def set_custom_prompt():
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt
custom_prompt_template = """Utilisez les éléments d'information suivants pour répondre à la question de l'utilisateur.
        Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

        Contexte : {context}
        Question : {question}

        N'affichez que la réponse utile ci-dessous et rien d'autre.
        Réponse utile :
        """
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chat_model = ChatGroq(temperature=0,
                            model_name="llama3-70b-8192",
                            api_key=groq_api_key)

vectorstore = Chroma(embedding_function=embed_model,
                    persist_directory="./chroma_db_llamaparse1",
                    collection_name="rag")
retriever=vectorstore.as_retriever(search_kwargs={'k': 3})


prompt = set_custom_prompt()
qa = RetrievalQA.from_chain_type(llm=chat_model,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": prompt})


def main():
    q = st.text_input("Enter your question")
    b = st.button("Answer")
    if b:
        response = qa.invoke({"query": q})
        st.write(response["result"])
    
if __name__ == '__main__':

    main()
