from flask import Flask


import random

import pandas as pd
import torch
from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 


dataset = pd.read_csv('data/keyword.csv')
keywords = dataset['keyword'].to_list()

class KoGPT2Chat(LightningModule):
    def __init__(self, hparams):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        print("Success setting AI")

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def chat(self, sent='0'):
        tok = TOKENIZER
        tok.tokenize(sent)
        with torch.no_grad():
            choice = random.sample(keywords, 3)
            q = str(choice[0] + ', ' + choice[1] + ', ' + choice[2]).strip()
            a = ''
            gen = ''
            while gen == EOS:
                a += gen.replace('▁', ' ')
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
            return a.strip()

model = KoGPT2Chat.load_from_checkpoint('data/model_-last.ckpt')


app = Flask(__name__)

@app.route("/epitagram")
def get_epitagram():
    chat = model.chat()
    print(chat)
    return {"status" : "success", "code" : "DP000", "data" : { "epitagram" : chat}}