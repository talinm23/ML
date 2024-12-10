Semantic search system using RAG

- This project is a semantic search system using Retrieval-Augmented Generation (RAG) to analyze and interact with PDF-formatted books. 
- The goal is to enable efficient analysis, exploration, and conversation with the text data 
- It takes the pdf of the book, splits it into sentences in the most basic way (using PYPDF2).
- The sentences get passed into the "thenlper/gte-large" model using the SentenceTransformer library.
- We get the vector embeddings back and save them in a MongoDB collection. 
- Then using the same model above, we embed a query about the pdf (e.g. what is the author's name).
- It searches the MongoDB collection using a vector search pipeline based on the user's query and returns a list of matchig documents.
- Then the outputs along with the original query about the book are passed into Google's Gemma model (google/gemma-2b-it) and it responds with the answers. 


