import os
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import glob

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[]
)

def api_needed(prompt: str) -> bool:
    meta_prompt = f"Does the following prompt need data from an weather api? Answer with only 'yes' or 'no'! \n Prompt: {prompt}"
    temp_session = model.start_chat(history=[])
    answer = temp_session.send_message(meta_prompt).text.strip().lower()
    return answer == "yes"

def use_weather_api(prompt: str) -> str:
    url_meta_prompt = f"""
    Answer only with a valid request url for the api.open-meteo.com api endpoint. 
    The url should fetch all necessary data to answer the following prompt: {prompt}
    """
    temp_session = model.start_chat(history=chat_session.history)
    answer = temp_session.send_message(url_meta_prompt).text.strip().lower().replace("\n", "").replace("```", "")
    response = requests.get(answer).json()

    meta_prompt = f"""
    The api.open-meteo.com api endpoint responds with the following data: 
    {response}
    
    Use this data to answer the following prompt: 
    {prompt}
    """
    return chat_session.send_message(meta_prompt).text

def file_needed(prompt: str) -> str:
    file_names =glob.glob("./files/*")
    meta_prompt = f"Does the following prompt need data from a file on the users system? Answer with only the filename and path or 'no'! \n Prompt: {prompt} \n File names: {file_names}"
    temp_session = model.start_chat(history=chat_session.history)
    answer = temp_session.send_message(meta_prompt).text.strip().lower()
    if answer in file_names:
        return answer
    return ""

def use_file(prompt: str, file_name: str) -> str:
    file = genai.upload_file(file_name)
    answer = model.generate_content([file, prompt]).text.strip().lower()
    file.delete()
    return answer


def rag(prompt: str) -> str:
    if api_needed(prompt):
        return use_weather_api(prompt)
    if file_name := file_needed(prompt):
        print("file needed")
        return use_file(prompt, file_name)
    return chat_session.send_message(prompt).text


if __name__ == "__main__":
    while user_input := input("Input a message: "):
        print(rag(user_input))
