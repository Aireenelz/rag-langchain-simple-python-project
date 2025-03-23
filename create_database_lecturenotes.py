from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH_LECTURENOTES = "data/lecture_notes/iot"

def load_documents_lecturenotes():
    loader = DirectoryLoader(DATA_PATH_LECTURENOTES, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#lecture_notes_documents = load_documents_lecturenotes()
#lecture_note_page = lecture_notes_documents[8] # access last page of lecture week 1
#print("LECTURE NOTE: \n", lecture_note_page, "\n")