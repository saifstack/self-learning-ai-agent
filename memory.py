import json
import os

MEM_FILE = "memories.json"

def load_memories():
    if not os.path.exists(MEM_FILE):
        return []
    with open(MEM_FILE) as f:
        return json.load(f)

def save_memories(mems):
    with open(MEM_FILE, "w") as f:
        json.dump(mems, f, indent=2)

def add_memory(fact):
    mems = load_memories()
    if fact not in mems:
        mems.append(fact)
        save_memories(mems)