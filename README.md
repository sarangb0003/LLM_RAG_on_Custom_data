#  Fine-Tuned-LLM-for-Custom-Data  
```
Created a Fine-tuned LLM model for question answering on custom data.   
```

## Pre-requisites:
- Python
- Account with Streamlit
- HuggingFace API key
- Langchain package and dependencies
- chromadb
- HuggingFace model "google/flan-t5-xxl"
<br>

## Demo:
[Demo Fine Tuned LLM for Custom Data](https://github.com/sarangb0003/Fine-Tuned-LLM-for-Custom-Data/assets/61322867/478c4498-1937-46db-8179-8550004ec6ad)

## Obsrevations:
1) for the question "Which city is the capital of England ?" <br>
--> Model answered it correctly as "London".
2) On which river it stands ? <br>
--> Here, I have asked the question by giving the input as "it" and the bot was correctly able to correlate that i was referring to "London" city. So by considering that, model was correctly able to answer it as "Thames". Same behavior observed for questions asked later in demo (check in demo video).

## Conclusion:
--> LLM model was able to answer the questions correctly for custom/unknown data provided. <br> 
--> LLM model was able to answer the questions correctly for custom/unknown data provided. It was able to memorize the context of conversation and answered the follow up questions by taking reference of previous question asked.

## Note:
--> Generated Results are short due to token size limit of models. <br>
--> Model may generate inaccurate or false content.

## Future Work:
--> Fine tune the model for more accurate results and can give answer human like text <br>
--> Bulid model which can perform operations on CSV data
