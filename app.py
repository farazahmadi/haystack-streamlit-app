import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np
import random
import os
from pdf_reader import create_document_store, rag_pipeline_generator, reference_finder_pipeline_generator

REF_COLOR = "#cc3df5"
#### test case
approved_claims = [
    'When your patient presents with moderate-to-severe plaque psoriasis, SAY TREMFYA®',
    'TREMFYA®/TREMFYA ONE-PRESS® (guselkumab injection) is indicated for the treatment of adult patients with moderate-to-severe plaque psoriasis who are candidates for systemic therapy or phototherapy.',
    'TREMFYA® demonstrated a superior PASI 90 response vs. COSENTYX at Week 48 (ITT population)',
    'Indication not previously mentioned and clinical use:TREMFYA®/TREMFYA ONE-PRESS® is also indicated for the treatment of adult patients with active psoriatic arthritis. TREMFYA®/TREMFYA ONE-PRESS® can be used alone or in combination with a conventional disease-modifying antirheumatic drug (cDMARD) (e.g., methotrexate).'
]


user_inputs = {
    'brand': 'Tremfya',
    'email_title': 'PLACEHOLDER_TITLE',
    'region': 'NORTH AMERICA',
    'user_type': 'HCP', # or PATIENT
    'brand_voice': 'FORMAL', # or INFORMAL
    'email_goal': 'EDUCATIONAL', # or PROMOTIONAL, AWARNESS, Call to Action
}

output_template = {
    'title': user_inputs['email_title'],
    'body': None,
    'reference': None, # source names of the files used in RAG or filtered based on user_inputs
    'safety': None,
    'footer': None
}

question = str()
document_store = None

def show_generated_email(claim_with_ref):
    for claim, refs in claim_with_ref:
        annotated_text(
            claim, " ",
            [(f"p. {ref[1]}", f"{ref[0]}", REF_COLOR) for ref in refs]
        )
    pass

def get_clean_claims_wRef(claims, user_inputs, find_ref_pipeline):
    claims =  [claim.strip() for claim in claims]
    # drop empty claims
    new_claims = [claim for claim in claims if len(claim) > 0]
    print(new_claims)
    claim_with_ref = [(claim, claim_reference_finder(find_ref_pipeline, claim, user_inputs)) for claim in new_claims]
    return claim_with_ref

def claim_reference_finder(pipeline, claim:str, user_inputs:dict):
    response = pipeline.run({"text_embedder": {"text": claim},
                             "retriever": {"filters": {"field": "user-type", "operator": "in","value": [user_inputs['user_type']]}, "top_k": 3}})
    references = [(doc.meta['source'], doc.meta['page']) for doc in response['retriever']['documents']]
    if len(references) == 0:
        references = [('PM', 40)]
    return set(references)


def get_safety_info():
    df = pd.read_excel("./static/safety.xlsx")
    return df['claim'].values

def get_static_claims(n_random=3):
    df = pd.read_excel("./static/claims.xlsx")
    out = list(zip(df['text'], df['reference']))
    # randomly return 3 claims
    return random.sample(out, n_random)

#### app
with st.sidebar:
    st.title('Generation Menu')
    openai_api_key = st.text_input("OpenAI API Key", key="file_qa_api_key", type="password")
    st.title("Library information:")
    user_inputs['brand'] = st.text_input("Brand name", key="brand_name", value='Tremfya')
    user_inputs['email_title'] = st.text_input("Email Title", key="email_title", value='PLACEHOLDER_TITLE')
    user_inputs['region'] = st.selectbox("Region", key="region", options=['NORTH AMERICA', 'EUROPE', 'ASIA'])
    user_inputs['user_type'] = st.selectbox("User Type", key="user_type", options=['HCP', 'PATIENT'], 
                                            help="Select the user type")
    user_inputs['brand_voice'] = st.selectbox("Brand Voice", key="brand_voice", options=['FORMAL', 'INFORMAL'])
    user_inputs['email_goal'] = st.selectbox("Email Goal", key="email_goal", options=['EDUCATIONAL', 'PROMOTIONAL', 'AWARENESS', 'Call to Action'])

                                       
st.title('Generate Marketing Content from Product Monograph')
uploaded_file = st.file_uploader("Drop a PDF file", type="pdf")
if uploaded_file is not None:
    output_template['reference'] = uploaded_file.name
    # Save file to a temporary location
    with open('temp.pdf', 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF file
    st.write("Loading PDF file...")
    document_store = create_document_store('temp.pdf')
    st.write("PDF file loaded successfully!")

    question = st.text_input(
        "Use a prompt to generate marketing content from the product monograph",
        placeholder=f"Indications and Usage for {user_inputs['brand']}",
        disabled=not uploaded_file,
    )

if question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🚨")

if document_store and question and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # st.write("Number of documents in the doc store", document_store.count_documents())
    print(type(user_inputs['user_type']))
    basic_rag_pipeline = rag_pipeline_generator(document_store, user_inputs, approved_claims)
    find_ref_pipeline = reference_finder_pipeline_generator(document_store)
    response = basic_rag_pipeline.run({"text_embedder": {"text": question}, 
                                   "prompt_builder": {'approved_claims': approved_claims},
                                   "retriever": {"filters": {"field": "user-type", "operator": "in","value": [user_inputs['user_type']]}, "top_k": 5}
                                    })
    # get claims
    new_claims = response['llm']['replies'][0].split('\n')
    # get references
    claims_with_ref = get_clean_claims_wRef(new_claims, user_inputs, find_ref_pipeline)

    # printing out the email
    st.title('Generated Email')
    st.markdown(f"### {user_inputs['email_title']}")
    st.write(f"Dear {user_inputs['user_type']},")
    annotated_text(
        "I hope you are doing well. \
         Below is some information that I thought you may find interesting\
          about the results of a head-to-head trial of TREMFYA® 100 mg vs. COSENTYX 300 mg\
          for the treatment of moderate-to-severe plaque psoriasis in adults. ",
        ("p. 3A","ECLIPSE", REF_COLOR),
        "Feel free to contact me if you have any questions"
    )
    show_generated_email(claims_with_ref)
    output_template['body'] = get_static_claims()
    st.markdown("#### Approved Claims from Library")
    for claim, ref in output_template['body']:
        annotated_text(
            claim, " ",
            [(ref, f"{user_inputs['brand']}_approved", REF_COLOR)]
        )
    st.markdown("### Safety Information")
    output_template['safety'] = get_safety_info()
    for info in output_template['safety']:
        st.markdown(f"- {info}")
    st.markdown("### References")
    st.markdown(f"1- {output_template['reference']}")
    pass

