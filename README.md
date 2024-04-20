# Poemify

An Image to Poem Generation Application, designed to create poetry from images. This project was created by Surabhi Kulkarni and Hrisha Yagnik for the LLMs Course at Northeastern University.

## Running Instructions
In order to run the main application, ensure streamlit version 1.24.0 is installed, and use the command streamlit run Poem_Gen_Interface.py. The final trained Microsoft GIT model must also be in the same directory as the application.

## Trained Models
Two models (Salesforce BLIP2 and Microsoft GIT) were fine tuned on the MultiM poem dataset and Microsoft GIT was chosen for the final user interface. The models are too large, so the google drive folders containing the model and the optimizers are provided below:
- Microsoft GIT: https://drive.google.com/drive/folders/1peaEiiru6ixIAMa6LwSxqZfM1lp3v3Rd?usp=sharing
- Salesforce BLIP2: https://drive.google.com/drive/folders/1JnyFf7nmhlIBePM8hIysXW_X1c4hp_XG?usp=sharing

## Files in Repo
- Data_Formatting.ipynb: Initial data structuring and reorganization to ensure compatibility with HuggingFace library
- Microsoft_Git_finetnue_poem.ipynb: Fine tuning code for Microsoft GIT
- BLIP2_finetune_poem.ipynb: Fine tuning code for BLIP2
- BLIP_Prompt_Eng_and_MS_Poem_Generation.ipynb: Experimentation with BLIP2 prompt engineering and MS GIT poem generation
- Poem_Gen_Interface.py: Interactive application integrating Microsoft GIT model
