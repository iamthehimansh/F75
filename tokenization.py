

import json


DATA_PATH="data/shakespeare.txt"

text=open(DATA_PATH,"r").read()

class CharacterTokenization():
    def __init__(self,setup=False):
        if setup:
            self.setup()
        
    def setup(self):
        vocab=sorted(set(text))
        self.stoi={s:i for i,s in enumerate(vocab)}
        self.itos={i:s for i,s in enumerate(vocab)}

    def encode(self,text):
        return [self.stoi[i] for i in text]

    def decode(self,token):
        return [self.itos[i] for i in token]
    
    def export(self,path="tokenizer.json"):
        with open(path,"w") as f:
            json.dump(self.itos,f)

    def import_token(self,path="tokenizer.json"):
        with open(path) as f:
            self.itos=json.load(f)
            self.itos={int(k):v for k,v in self.itos.items()}
            self.stoi={self.itos[k]:k for k in self.itos.keys()}
            self.itos
            print(self.stoi)
    


if __name__ =="__main__":
    tokenizer=CharacterTokenization()
    tokenizer.import_token()
    print(tokenizer.encode("hello boy"))
    print(tokenizer.decode([12,13,14]))