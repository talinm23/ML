

According to the paper, https://www.sciencedirect.com/science/article/pii/S2214750021001268 (Vaccines and sudden infant death: An analysis of the VAERS database 1990–2019 and review of the medical literature), an online search of the VAERS database was conducted. 
The database was filtered to only include reports with a vaccination date from 1990 through 2019, of infants (children < 1 year of age) who died within 60 days post-vaccination.
In this project, we reconstruct the study and check its statistics. 

Please review the analysis in VAERS_data_MiniLM.ipynb. 
A static render is available here: 
<https://nbviewer.org/github/talinm23/ML/blob/main/Projects/VAERS_database/VAERS_data_MiniLM.ipynb>

In this project, we also fed the (csv) data into an LLM model to answer some specific questions (files rag_with_csvs_data_processing.py and rag_with_csvs.py). 
In order to run these notebooks, you need the VAERS_DATA.csv file, which is the output of the data_processing code. 
The original three files from the CDC website are used in the data_processing code to filter and prepare the data to be saved as a combined csv file.  

In the second code (rag_with_csvs.py) then we process the csv files in chunks. A FAISS vector store is created from these embeddings for efficient similarity search.
And we also use the OPENAI embeddings (you need your OPENAI API key). Then we retrieve and query the processed csvs and get responds from the OPENAI prompt.

Here's an example that exists in the code: 

    system_prompt = (
          "You are an assistant for question-answering tasks. "
          "Use the following pieces of retrieved context to answer "
          "the question. If you don't know the answer, say that you "
          "don't know. Use three sentences maximum and keep the "
          "answer concise.") 


    answer= rag_chain.invoke({"input": "Which vaccine has the most symptoms?"
                               "Which vaccine causes the most deaths?"
                               "What duration is the most frequent that the death occurred?"
                               "Share some interesting statistics."
                               "Can you conclude that the deaths were related to the vaccine?"
                               "Explain how many records you are looking at."})


Output: 

    The vaccine with the most symptoms reported is the DTAPHEPBIP, HIBV, PNC13, RV5 combination, 
    which includes symptoms like death, sudden infant death syndrome, and others. 
    The vaccine associated with the most deaths is the DTAPHEPBIP, HIBV, PNC13, RV5 combination. 
    The most frequent duration for death occurrence is 1 day after vaccination. 
    Some interesting statistics include the frequent mention of sudden infant death syndrome 
    (SIDS) as a cause of death and the commonality of deaths occurring within a few days 
    post-vaccination. However, it is important to note that these reports do not establish 
    a causal relationship between the vaccines and the deaths. The records reviewed include 
    multiple cases, but the exact number is not specified in the provided context.


