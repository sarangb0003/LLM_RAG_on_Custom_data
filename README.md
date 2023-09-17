#  Fine-Tuned-LLM-for-Custom-Data  
```
Created a Fine-tuned LLM model for question answering on custom data.   
```

## Pre-requisites
- Python
- Account with Streamlit
- HuggingFace API key
- Langchain package and dependencies
- chromadb
- HuggingFace model "google/flan-t5-xxl"

## Demo
[streamlit-streamlit_app-2023-09-16-23-09-12.webm](https://github.com/sarangb0003/Fine-Tuned-LLM-for-Custom-Data/assets/61322867/478c4498-1937-46db-8179-8550004ec6ad)



## Obsrevations

Founds two interesting answers from model output:
1)  For the question "Which city is the capital of England ?"
--> Model answered it correctly as "London".
   
2) For question "write python code to Combine two lists and remove duplicates without using set function"
--> provided the code with no use of set() python code as expected.

## Future Work

-- Fine tune the model with custom data for specific use case
-- Improve accuracy 
