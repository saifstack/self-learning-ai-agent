import json
from openai import OpenAI 

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") 

EXTRACT_PROMPT = """You extract durable facts about the user from their message. Durable = names, preferences, goals, constraints, projects, skills. NOT durable = small talk, one-off questions, anything about today only. Return JSON exactly like this: {"facts": ["fact one", "fact two"]} Return {"facts": []} if nothing is worth keeping. Write each fact as a short third-person sentence starting with "User". """ 
def extract_facts(user_message): 
    r = client.chat.completions.create( model="llama3.2", messages=[ {"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": user_message}, ], response_format={"type": "json_object"}, ) 
    try: return json.loads(r.choices[0].message.content)["facts"] 
    except Exception: return []