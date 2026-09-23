from openai import OpenAI
from vector_memory import add_memory, search
from extractor import extract_facts

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def build_system_prompt(user_message): 
    mems = search(user_message, k=5) 
    if not mems: 
        return "You are a blunt, helpful assistant." 
    block = "\n".join(f"- {m}" for m in mems) 
    return ( "You are a blunt, helpful assistant.\n\n" "Possibly relevant things you know about this user:\n" f"{block}\n\n" "Use them only if relevant. Do not announce that you remembered." )

messages = [{"role": "system", "content": "You are a blunt, helpful assistant."}]

while True:
    user = input("\nyou: ")
    if user.strip() in {"quit", "exit"}:
        break

    for fact in extract_facts(user):
        add_memory(fact)
        print(f"  [learned] {fact}")

    messages[0] = {"role": "system", "content": build_system_prompt(user)}
    messages.append({"role": "user", "content": user})

    r = client.chat.completions.create(model="llama3.2", messages=messages)
    reply = r.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(f"\nbot: {reply}")