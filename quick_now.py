import requests, random
U="http://127.0.0.1:8080/v1/chat/completions"
def ask(p, mt, t=0.0):
    r=requests.post(U,json={"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":t,"top_p":1.0,"stream":False},timeout=600)
    r.raise_for_status(); return r.json()["usage"]
p1="Write a 200-word essay on why the sky is blue at noon and red at sunset."
u1=ask(p1,256); u2=ask(p1,256)
print(f"decode fresh: {u1['completion_tokens']}t | repeat: {u2['completion_tokens']}t")
words=("harbor lantern sediment glacier magnet orbit quiver thicket cobalt ember raster palindrome "
       "viaduct monsoon kestrel azimuth trellis fathom bramble zephyr anvil cascade dovetail ").split()
rng=random.Random(77)
filler=" ".join(rng.choice(words) for _ in range(20000))
u3=ask(filler+"\n\nReply with only: done", 8)
print(f"prefill: {u3['prompt_tokens']} tokens")
