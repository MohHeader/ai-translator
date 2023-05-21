import streamlit as st
import time
import os

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

#@st.cache_resource
def getLLM(openai_api_key=None, model_name='gpt-3.5-turbo'):
  return ChatOpenAI(temperature=0,
                 model_name=model_name,
                 request_timeout=1200,
                 openai_api_key=openai_api_key
                )

with st.expander("Advanced Options"):
  gpt_model = st.selectbox(
    'Please Select GPT Model?',
    ('gpt-3.5-turbo', 'gpt-4'))

  openai_api_key_input = st.text_input("OpenAI API Key", "sk-")
  
  chunk_size_input = st.slider(
      "Chunk characters limit :",
      value=200,
      max_value=10000,
      min_value=50
  )
  
  system_prompt_input = st.text_area("System Instructions", """Instructions:
Act as a language translator, user will give you english strings (each made of one word or more), each string in a new line

And you need to translate them all to all the following languages:
{languages}

Respond in a csv format including the original english string, so each line will be converted to a only one single line with the translations.

If you see anything that starts with < and ends with > leave it don't translate it, that's just a tag, however, translate the words in between normally, as if these tags didn't exist.

# Example 1:
The next example contains only 4 strings
## Example Input -- Start:
Languages: FR	JA	DE	ES	IT
Loading...
Pass Level 10
Chest
Gift
## Example Input -- End

## Example Output -- Start:
"Loading...","Chargement...","ロード中...","Wird geladen...","Cargando...","Caricamento in corso..."
"Pass Level 10","Dépasse le niveau 10","ステージ10を突破","Bestehe Level 10","Completa el nivel 10","Supera il livello 10"
"Chest","Coffre","宝箱","Truhe","Cofre","Forziere"
"Gift","Cadeau","報酬","Geschenk","Regalo","Regalo"
## Example Output -- End

# Example 2:
The next example contains only 1 string
## Example Input -- Start:
Languages: FR	JA
Gift
## Example Input -- End

## Example Output -- Start:
"Gift","Cadeau","報酬"
## Example Output -- End


#Example 3:
The next example contains only 1 multi-line string
## Example Input -- Start:
Languages: Deutsch, Turkish, Arabic
"The Piggy Bank is full!
It's time to collect your earnings!"
## Example Input -- End

## Example Output -- Start:
"The Piggy Bank is full!
It's time to collect your earnings!","Das Sparschwein ist voll!
Es ist Zeit, Ihre Einnahmen zu sammeln!","Kumbara dolu!
Kazancınızı toplama zamanı!","الخنزير المتواضع ممتلئ!
حان الوقت لجمع أرباحك!"
## Example Output -- End

User could send you a bigger list, you need to translate them all to the languages specified above and not the ones in the examples""")

keywords = st.text_area("The list to Translate")
languages = st.text_input("Target Translations", "Deutsch, Turkish, Arabic")

translate_button = st.button("Translate")
if translate_button:
  
  if len(openai_api_key_input) < 5:
    openai_api_key = os.environ['OPENAI_API_KEY']
  else:
    openai_api_key = openai_api_key_input

  llm = getLLM(openai_api_key, gpt_model);
  systemPrompt = SystemMessagePromptTemplate.from_template(system_prompt_input)
  
  humanPrompt = HumanMessagePromptTemplate.from_template("""
  Languages: {languages}
  {keywords}""")

  chat_prompt = ChatPromptTemplate.from_messages([systemPrompt, humanPrompt])
  
  
  text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_input,
            separators=["\n"],
            chunk_overlap=0,
        )
  chunks = text_splitter.split_text(keywords)
  result = ""
  st.write(f"Split into {len(chunks)} parts")
  for i, chunk in enumerate(chunks):
    start_time = time.time()
    messages = chat_prompt.format_prompt(keywords=chunk, languages=languages).to_messages()
    response = llm(messages).content
    result += response.replace("\\n", "\n") + "\n"
    st.write(f"{i}, took: {time.time() - start_time} seconds")
    st.write("-----")
    time.sleep(0.4)
    
  st.code(result)
  st.download_button('Download CSV', result, f"{languages.replace(' ', '-').lower()}-translations.csv")