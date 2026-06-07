import streamlit as st
import pandas as pd
from llm import query_llama
from evaluator import evaluate

st.set_page_config(page_title = "Llama Sandbox Attack", layout = "wide")
st.title("Llama Prompt Injection Sandbox")
st.write("Test how a local Llama model behaves under adversarial prompts")

category = st.sidebar.selectbox(
    "Test Category",
    ["normal", "roleplay", "instruction_conflict", "multi_turn", "custom"]
)

user_input = st.text_area ("Enter test prompt")

run = st.button("Run test")

if run and user_input:
    with st.spinner("Querying Llama.."):
        response = query_llama(user_input)
        
    label, score = evaluate(user_input, response)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prompt")
        st.write(user_input)
        
        st.subheader("Evaluation")
        st.json(score)
        
        st.success(label)
        
    with col2:
        st.subheader("Llama response")
        st.write (response)
        
    row = pd.DataFrame([{
        "category": category,
        "prompt": user_input,
        "response": response,
        "label": label
    }])
    
    file_exists = os.path.exists("results.csv")
    row.to_csv("results.csv", mode="a", header=not file_exists, index=False)
    
    st.divider()
    st.subheader("Collected results")
    
    if os.path.exists("results.csv"):
        df = pd.read_csv("results.csv")
        st.dataframe(df)
        
    else:
        st.info("No results yet")
        