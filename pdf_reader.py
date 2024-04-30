import spacy
import re
import streamlit as st
from spacypdfreader.spacypdfreader import pdf_reader
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

@st.cache_resource
def create_document_store(pdf_path: str) -> InMemoryDocumentStore:

    nlp = spacy.load('en_core_web_sm')
    spacy_doc = pdf_reader(pdf_path, nlp)
    RE_EXCESS_NEWLINE = re.compile(r"\n(?=[a-z])") # A \n followed by a lowercase character

    docs = []
    for p in range(1, spacy_doc._.page_range[1] + 1):
        user_type = ('HCP'
                if p >= 4 and p <= 26
                else 'PATIENT'
                if p >= 27 and p <= 32
                else 'NULL')
        
        docs.append(Document(
            content=RE_EXCESS_NEWLINE.sub("", spacy_doc._.page(p).text), 
            meta={
                'source': spacy_doc._.pdf_file_name,
                'page': p, 
                'drug-name': 'Tremfya',
                'material-type': 'PM', # can use marketing material, etc
                'user-type': user_type
                }))


    document_store = InMemoryDocumentStore()
    splitter = DocumentSplitter(split_by='passage', split_length= 4, split_overlap=1) # initial 2 and 0 
    final_docs = splitter.run(docs)
    print("Initial number of pages: ", len(docs))
    print("Final number of chunks: ", len(final_docs['documents']))
    # Add document embeddings
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(final_docs['documents'])
    document_store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.SKIP)
    return document_store

def rag_pipeline_generator(document_store: InMemoryDocumentStore, user_inputs: dict, approved_claims: list[str]) -> Pipeline:
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=5)

    llm_system_prompt = '''You are a marketer working for a pharmaceutical company.
    Your job is to generate core marketing claims based of a Product monograph.
    The facts stated in the product monograph are dry material and must be turned into marketing 
    content while maintaining regulatory compliance and key factual information.'''

    template = """

    Product Monograph:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Based of the Product Monograph (PM) stated Above. Extract at least three marketing claim for that drug.

    Example:
    {% for claim in approved_claims %}
        {{ loop.index }}. {{ claim }}
    {% endfor %}

    Marketing Claims:
    """

    prompt_builder = PromptBuilder(template=template)
    generator = OpenAIGenerator(model="gpt-3.5-turbo",system_prompt=llm_system_prompt, generation_kwargs={'temperature': 0.2})


    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")
    basic_rag_pipeline.warm_up()
    return basic_rag_pipeline

def reference_finder_pipeline_generator(document_store: InMemoryDocumentStore)-> Pipeline:
    find_ref_pipeline = Pipeline()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=3)
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    find_ref_pipeline.add_component("text_embedder", text_embedder)
    find_ref_pipeline.add_component("retriever", retriever)

    find_ref_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    find_ref_pipeline.warm_up()
    return find_ref_pipeline

