# Quora Question Answer - Generative AI

1.	quora_answer_generation_notebook (Jupyter Notebook) : This Notebook consists of testing multiple LLM, retrieval techniques, prompting and evaluation on Test Data.
2.	question_answer.py (Python File) : This python code file consists of backend - helper functions, vectorstore, LLM for Chatbot.
3.	app_rag.py (Python Files) : This python code consist of streamlit app codes to be utilized as chatbot, app_rag consist of RAG based chatbot.
4.	test_evaluate.csv (Worksheet): The worksheet consists of test questions and their reference answers, question and answer pair from Retrieval, RAG based answer generation, and its respective BLEU, ROUGE and F1 Scores.

Before running codes in your environment please make sure to install required libraries using requirements.txt (pip install -r requirements.txt).
If you want to launch the chatbot, please use these commands in your command prompt in the same directory of the project folder. streamlit run app_rag.py for RAG based chatbot. 
