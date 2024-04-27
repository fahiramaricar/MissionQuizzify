# embedding_client.py
import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings
from vertexai.language_models import TextEmbeddingModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.

    The EmbeddingClient class should be capable of initializing an embedding client with specific configurations
    for model name, project, and location. Your task is to implement the __init__ method based on the provided
    parameters. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Steps:
    1. Implement the __init__ method to accept 'model_name', 'project', and 'location' parameters.
       These parameters are crucial for setting up the connection to the VertexAIEmbeddings service.

    2. Within the __init__ method, initialize the 'self.client' attribute as an instance of VertexAIEmbeddings
       using the provided parameters. This attribute will be used to embed queries.

    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.

    Instructions:
    - Carefully initialize the 'self.client' with VertexAIEmbeddings in the __init__ method using the parameters.
    - Pay attention to how each parameter is used to configure the embedding client.

    Note: The 'embed_query' method has been provided for you. Focus on correctly initializing the class.
    """
    
    def __init__(self, model_name, project, location):
        # Initialize the VertexAIEmbeddings client with the given parameters
        # Read about the VertexAIEmbeddings wrapper from Langchain here
        # https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai
        self.client = VertexAIEmbeddings(model_name=model_name, project=project, location=location)
        #self.client.embed_documents()
        # self.client= TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        # self.client = GoogleGenerativeAIEmbeddings(model=model_name,project= project,location=location)

        
    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.

        :param query: The text query to embed.
        :return: The embeddings for the query or None if the operation fails.
        """
        vectors = self.client.embed_query(query)
        return vectors
    
    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.

        :param documents: A list of text documents to embed.
        :return: A list of embeddings for the given documents.
        """
        try:
            # Ensure documents is always treated as a list
            # if not isinstance(documents, list):
            #  documents = [documents]
            vectors = self.client.embed_documents(documents)
            return vectors
        except AttributeError:
            #print("Method embed_documents not defined for the client.")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    # model_name = "models/embedding-001"
    project = "x-signifier-421304"
    location = "us-central1"

    embedding_client = EmbeddingClient(model_name, project, location)
    text = ["hello","hi"]
    #vectors = embedding_client.embed_query("Hello world")
    vectors = embedding_client.embed_documents(["Hello World!", "First document"])
    if vectors:
        st.write(vectors)
        print(vectors)
        print("Successfully used the embedding client!")
