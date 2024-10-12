This project uses Retrieval Augmented Generation with LLMs to answer questions about a document. It involves:

* Document Processing: A PDF document is converted into images, and Optical Character Recognition (OCR) is used to extract text.
* Data Preparation: The extracted text is structured, with relevant metadata added for context.
* LLM Integration: A pre-trained LLM from Hugging Face is used for summarization and question answering.
* Vector Database: A ChromaDB is created to store the document chunks and their embeddings for efficient retrieval.
* Answer Generation: The retrieved context and the user's input query are fed to the LLM to generate a concise answer.
