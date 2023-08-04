
import pandas as pd
import streamlit as st
import requests
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
serp_api_key = os.getenv('SERPAPI_API_KEY')
alpha_key = os.getenv('alpha_vintage_api_key')

data_prompt = PromptTemplate(
    input_variables=['query'],
    template='{query} '
)

query_prompt = PromptTemplate(
    input_variables=['query'],
    template='you have the day wise  data of the stock execute the following user command and give the desired results for the query : "{query}" '
)

temp_dir = 'temp_folder'
os.makedirs(temp_dir, exist_ok=True)

def fetch_stock_data(company_code):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}&datatype=json&outputsize=full'.format(
        company_code, alpha_key)
    response = requests.get(url)
    data = response.json()
    extracted_data = []
    for date, values in data['Time Series (Daily)'].items():
        extracted_data.append({
            'Date': date,
            'Open': float(values['1. open']),
            'High': float(values['2. high']),
            'Low': float(values['3. low']),
            'Close': float(values['4. close']),
            'Volume': int(values['5. volume']),
            "company name": data['Meta Data']['2. Symbol']
        })
    return extracted_data

def create_chatbot_response(query):
    try:
        tool_chain = load_tools(tool_names=['serpapi', 'llm-math'], llm=OpenAI(api_key=OPENAI_API_KEY))
        agent_decider = initialize_agent(tool_chain, OpenAI(api_key=OPENAI_API_KEY), agent='zero-shot-react-description', verbose=True)
        reply = agent_decider.run(data_prompt.format(query=query))
        return reply
    except Exception as e:
        st.warning(f"An error occurred: {e}")
        company_code = st.text_input("Enter the company code:")
        if company_code:
            extracted_data = fetch_stock_data(company_code)
            df = pd.DataFrame(extracted_data)
            temp_file_path = os.path.join(temp_dir, f'{company_code}.csv')
            df.to_csv(temp_file_path, index=False)
            agent = create_csv_agent(OpenAI(api_key=OPENAI_API_KEY), temp_file_path, verbose=True)
            response = agent.run(query_prompt.format(query=query))
            os.remove(temp_file_path)
            return response

def main():
    st.title("Stock Data Chatbot")
    query = st.text_input("Enter your query:")
    if query:
        response = create_chatbot_response(query)
        if response:
            st.success(response)

if __name__ == "__main__":
    main()
