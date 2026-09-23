import sqlite3 
import numpy as np 
from openai import OpenAI 

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") 

DB = "memories.db" 
def _conn(): 
    c = sqlite3.connect(DB) 
    c.execute("CREATE TABLE IF NOT EXISTS mem (fact TEXT PRIMARY KEY, vec BLOB)") 
    return c 
def embed(text): 
    r = client.embeddings.create(model="nomic-embed-text", input=text) 
    v = np.array(r.data[0].embedding, dtype=np.float32) 
    return v / np.linalg.norm(v) 
def _insert(fact): 
    c = _conn() 
    c.execute("INSERT OR IGNORE INTO mem VALUES (?, ?)", (fact, embed(fact).tobytes())) 
    c.commit() 
    c.close() 
def _delete(fact): 
    c = _conn() 
    c.execute("DELETE FROM mem WHERE fact = ?", (fact,)) 
    c.commit() 
    c.close() 
def _find_closest(fact): 
    c = _conn() 
    rows = c.execute("SELECT fact, vec FROM mem").fetchall() 
    c.close() 
    if not rows: 
        return None, 0.0 
    q = embed(fact) 
    best_fact, best_score = None, -1.0 
    for f, blob in rows: 
        v = np.frombuffer(blob, dtype=np.float32) 
        score = float(q @ v) 
        if score > best_score: 
            best_fact, best_score = f, score 
    return best_fact, best_score 

def _resolve(old_fact, new_fact): 
    prompt = ( 
        "Two facts about the same user might conflict.\n\n" 
        f'OLD: "{old_fact}"\n' 
        f'NEW: "{new_fact}"\n\n' 
        "If NEW replaces OLD (an update, a correction, a change over time), reply exactly: new\n" 
        "If OLD is still true and NEW is a different, unrelated fact, reply exactly: both\n" 
        "If NEW is basically the same as OLD, reply exactly: old\n" 
        "Reply with one word only: new, both, or old." 
    ) 
    r = client.chat.completions.create( 
        model="llama3.2", 
        messages=[{"role": "user", "content": prompt}], 
    ) 
    return r.choices[0].message.content.strip().lower() 

def add_memory(new_fact): 
    closest, score = _find_closest(new_fact) 
    if closest is None or score < 0.85: 
        _insert(new_fact) 
        return 
    if closest == new_fact: 
        return 
    decision = _resolve(closest, new_fact) 
    if decision == "new": 
        _delete(closest) 
        _insert(new_fact) 
        print(f" [replaced] \"{closest}\" -> \"{new_fact}\"") 
    elif decision == "both": 
        _insert(new_fact) 
def search(query, k=5):
    c = _conn()
    rows = c.execute("SELECT fact, vec FROM mem").fetchall()
    c.close()
    if not rows:
        return []

    q = embed(query)
    scored = []
    for fact, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        scored.append((float(q @ v), fact))

    scored.sort(reverse=True)
    return [fact for score, fact in scored[:k] if score > 0.3]