# RAG

This repo is a simple chatbot with RAG (Retrieval Augmented Generation).
It can answer questions based on information in txt and pdf files as well as weather information from [Open-Meteo](https://open-meteo.com/).

## How to get started
### 1. Install dependencies
With pip:
```bash
pip install .
```
or with poetry:
```bash
poetry install
```

## 2. Create .env
Create a new file with the name `.env` and write
```.env
GEMINI_API_KEY=<your-api-key>
```
in it.

## 3. Setup the vector database
Just run
```bash
docker compose up
```
in the project root.

## 4. Add files
Add files with information the AI should have in the `files` directory in the project root.

## 5. Run the rag.py file
```bash
python rag.py
```
When running it for the first time the information has to be but in the vector db so answer `y` when asked if the information should be indexed.

# Example prompts


> What is the company Satellytes?
> 
> Answer: Satellytes is a technology company that is always looking for passionate developers. 

> What options for a carer does Satellyets provide?
> 
> Answer: Satellyets is looking for passionate developers who want to take on new challenges and continually develop themselves. 

> What is the secret number?
> 
> Answer: The secret number is 54321. 

> How is the weather in Munich?
> 
> Answer: I can tell you that the weather in Munich is currently 12.7°C with a relative humidity of 83%. 
> 
> And tomorrow?
> 
> Answer: The weather in Munich tomorrow will be 12.0°C with a relative humidity of 95%.
> 
> Will it rain tomorrow?
> 
> Answer: Yes, it will rain tomorrow in Munich. 

> What is Satellyets?
> 
> Answer: Satellytes is a technology company that focuses on creating high-quality software solutions. 