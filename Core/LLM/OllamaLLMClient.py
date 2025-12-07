
import subprocess, json
class OllamaLLMClient:
    def __init__(self, model="phi3"):
        self.model=model
    def generate(self,prompt):
        proc=subprocess.Popen(["ollama","run",self.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out,_=proc.communicate(prompt)
        txt=""
        for line in out.splitlines():
            try:
                obj=json.loads(line)
                if "response" in obj: txt+=obj["response"]
            except:
                txt+=line
        return txt.strip()
