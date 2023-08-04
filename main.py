import streamlit as st
import os
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import shutil
from langchain.memory import StreamlitChatMessageHistory

history = StreamlitChatMessageHistory("foo")



load_dotenv()
st.set_page_config(page_title='Ask your csv ')
st.header('ask your csv')

# Create a temporary directory
temp_dir = 'temp_folder'
os.makedirs(temp_dir, exist_ok=True)

file = st.file_uploader('Upload CSV', type='csv')

if file:
    temp_file_path = os.path.join(temp_dir, "uploaded_file.csv")

    # Save the uploaded CSV file to the temporary directory
    with open(temp_file_path, 'wb') as f:
        f.write(file.getvalue())

    llm = OpenAI(temperature=0)
    user_input = st.text_input("Question here:")

    agent = create_csv_agent(llm, temp_file_path, verbose=True)

    if user_input:
        response = agent.run(user_input)
        history.add_user_message(user_input)
        st.write(response)
        history.add_ai_message(response)
# Delete the temporary directory and its contents after code execution
shutil.rmtree(temp_dir, ignore_errors=True)

 


