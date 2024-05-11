# Streamlit App: Automated Claim Generation from PM Files

This repository contains a Streamlit application that automates the generation of marketing claims from product information documents (PM files) in PDF format.

## What it does:

- **Parses PM files:** The app reads and parses PM files uploaded as PDFs, extracting relevant information in a structured format suitable for information retrieval.
- **Haystack Integration:** Haystack, a powerful library for Natural Language Processing (NLP), is used to build a vector database with sentence embeddings. This optimizes the efficiency of searching and retrieving relevant information from parsed PM documents.
- **RAG-based Claim Generation with OpenAI API:** The app leverages Retrieval-Augmented Generation (RAG) and the OpenAI API (GPT-3.5) to automate claim generation. Here's how it works:
  1. The user submits a query.
  2. The app retrieves the top documents from the Haystack vector database based on semantic similarity to the query.
  3. Using RAG with GPT-3.5, the app generates claims based on the retrieved documents, fine-tuned to convert technical content into audience-specific marketing claims (HCPs vs. Patients).
- **Safety Information Integration:** Finally, approved claims and safety information are added to the generated content from a pre-defined library.

## Benefits:

- **Increased Efficiency:** Automates claim generation, saving time and resources.
- **Improved Accuracy:** Haystack ensures high-precision information retrieval for claim generation.
- **Tailored Content:** Fine-tuned GPT-3.5 generates claims that resonate with specific audiences (HCPs vs. Patients).
- **Streamlined Workflow:** Streamlit provides a user-friendly interface for interacting with the claim generation process.

## Getting Started:

1. Clone this repository.
2. Ensure you have the required dependencies installed (`pip install -r requirements.txt`).
3. Run the app using `streamlit run app.py`.
4. Upload your PM file (PDF format) and enter your query.
5. The generated claims will be displayed on the screen.

**Note:** This application utilizes the OpenAI API, which may require a separate subscription.

## Future Developments:

- Integrate additional functionalities such as claim editing and approval workflows.
- Enhance claim generation customization options for a wider range of use cases.

Feel free to reach out with any questions or suggestions!