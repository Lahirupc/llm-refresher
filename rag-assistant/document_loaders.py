import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader, PyPDFLoader)



from dotenv import load_dotenv
load_dotenv()


def load_text_file():
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write("Hello, this is a test file.")
        temp_file_path = temp_file.name

    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print("Document content:")
            print(doc)
            print(doc.page_content)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def pdf_loader():
    loader = PyPDFLoader("./docs/example.pdf")
    documents = loader.load()    
    for doc in documents:
        print("Document content:")
        print(doc)
        print(doc.page_content)


if __name__ == "__main__":
    pdf_loader()
