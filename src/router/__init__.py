from fastapi import APIRouter

import random

import pandas as pd
import torch
from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

import tracemalloc

from ..DTO import GetEpitagramResponse

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


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

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
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')
            return a.strip()

model = KoGPT2Chat.load_from_checkpoint('src/data/model_-last.ckpt')
dataset = pd.read_csv('src/data/keyword.csv')
keywords = dataset['keyword'].to_list()

router = APIRouter()
@router.get("/", response_model=GetEpitagramResponse)
def get_epitagram():
    chat = model.chat()
    print(chat)
    return {"status" : "success", "code" : "DP000", "data" : { "epitagram" : chat}}


router_list = [{"router": router, "prefix": "/epitagram"}]