# Quora Question Answer - Generative AI

I propose a novel approach for Quora Answer Generation using Generative AI utilizing Question answer pair training set and state of the art LLMs. This approach comprises of RAG based architecture approach.
Before delving into the approach, it's essential to grasp the knowledge of dataset utilized. We've leveraged an open-source dataset containing question answer pair.
Here is the data source : toughdata/quora-question-answer-dataset

**This repository consists of:**
1.	quora_answer_generation_notebook (Jupyter Notebook) : This Notebook consists of testing multiple LLM, retrieval techniques, prompting and evaluation on Test Data.
2.	question_answer.py (Python File) : This python code file consists of backend - helper functions, vectorstore, LLM for Chatbot.
3.	app_rag.py (Python Files) : This python code consist of streamlit app codes to be utilized as chatbot, app_rag consist of RAG based chatbot.
4.	test_evaluate.csv (Worksheet): The worksheet consists of test questions and their reference answers, question and answer pair from Retrieval, RAG based answer generation, and its respective BLEU, ROUGE and F1 Scores.

**Note:** Before running codes in your environment please make sure to install required libraries using requirements.txt (pip install -r requirements.txt).
**Note:** If you want to launch the chatbot, please use these commands in your command prompt in the same directory of the project folder. streamlit run app_rag.py for RAG based chatbot. 
