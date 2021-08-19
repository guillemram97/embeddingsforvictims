import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from transformers import BertModel
from tqdm import tqdm
import logging
import time
import transformers
#logging.basicConfig(level=logging.INFO)
#import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

el=np.load('lis.npy', allow_pickle=True)
nume=np.load('num_no_cont.npy', allow_pickle=True)

# Load pre-trained model (weights)
tt=time.time()

model = BertModel.from_pretrained('bert/', output_hidden_states = True)
print("Time "+str(time.time()-tt))

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

aa=int(len(el)/100)+1
maxims=np.linspace(0, 0+(100*aa),aa,endpoint=False)[1:]
maxim_idx=0


for idx, maxim in enumerate(maxims[:100]):
    maxim=int(maxim)
    idx_together=[]
    idx_numerals=[]
    tokenized_text_vec=[]
    indexed_tokens_vec=[]
    mask_vec=[]
    fail=0
    segments_ids_vec = [] #[1] * len(tokenized_text[0])
    max_len=0
    print("_____TOKENIZING____")
    tt=time.time()
    if idx==0: inici=int(0)
    else: inici=int(maxims[idx-1])
    for idx_2, txt in tqdm(enumerate(el[inici:maxim])):
        idx_bo=inici+idx_2
        if idx_bo==0 or el[idx_bo]!=el[idx_bo-1]:
            old_txt=txt
            try:
                idx_numerals.append([txt.index('numnumnum'), len(txt)-txt.index('numnumnum')-1])
                new_idx_together=[]
                txt=str(txt)[1:-1].replace('\'', '')
                txt=txt.replace(',', '')
                txt=txt.replace('.', '')
                s=nume[idx_bo].lower()
                if s[-2:]==',0' or s[-2:]=='.0':
                    s=s[:-2]
                s=s.replace(',', '')
                s=s.replace('.', '')
                if s.replace(' ', '').isnumeric():
                    s=s.replace(' ', '')
                print(s)
                txt=txt.replace('numnumnum', s)
                # Add the special tokens.
                marked_text = "[CLS] " + txt + " [SEP]"
                tokenized_text = tokenizer.tokenize(marked_text)
                max_len=max(len(tokenized_text), max_len)
                mask=np.concatenate((np.ones(len(tokenized_text)), np.zeros(int(100-len(tokenized_text)))))
                while len(tokenized_text)<100: tokenized_text.append("[PAD]")
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids=[1]*len(tokenized_text)
                for idx_3, segment in enumerate(tokenized_text):
                    if segment[:2]=='##': new_idx_together.append(idx_3)
                idx_together.append(new_idx_together)
                tokenized_text_vec.append(tokenized_text)
                indexed_tokens_vec.append(indexed_tokens)
                segments_ids_vec.append(segments_ids)     
                mask_vec.append(mask.tolist())
            except:
                fail+=1
    tokens_tensor = torch.tensor([indexed_tokens_vec])
    segments_tensors = torch.tensor([segments_ids_vec])
    attention_mask = torch.tensor([mask_vec])
    print("Time "+str(time.time()-tt))
    print(max_len)

    print("_____BERT IS THINKING_____")
    tt=time.time()
    with torch.no_grad():
        outputs = model(tokens_tensor[0], attention_mask[0], segments_tensors[0])
        hidden_states = outputs[2]
    print("Time: "+str(time.time()-tt))
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = token_embeddings.permute(1, 2, 0, 3)

    embeds=[]
    for idx_4, sentence in enumerate(token_embeddings):
        token_vecs_sum = []
        mask=attention_mask[0][idx_4]
        idx_tog=idx_together[idx_4]
        primer=True 
    embeds=[]
    for idx_4, sentence in enumerate(token_embeddings):
        token_vecs_sum = []
        mask=attention_mask[0][idx_4]
        idx_tog=idx_together[idx_4]
        primer=True 
        for idx_2, token in enumerate(sentence):
            new_idx=99-idx_2
            token=sentence[new_idx]
            if not primer and new_idx>0:
                new_cont+=1
                sum_vec += torch.sum(token[-4:], dim=0)
                if not new_idx in idx_tog:
                    sum_vec=sum_vec/new_cont
                    token_vecs_sum.append(sum_vec)
                    sum_vec=torch.zeros(768)
                    new_cont=0
                else:
                    assert tokenized_text_vec[idx_4][new_idx][:2]=='##'
            if primer and mask[new_idx]==1:
                primer=False
                sum_vec=torch.zeros(768)
                new_cont=0
        if idx_numerals[idx_4][0]+idx_numerals[idx_4][1]+1==len(token_vecs_sum):
            token_vecs_sum=torch.stack(token_vecs_sum)
            embeds.append(token_vecs_sum)
        else:
            if idx_numerals[idx_4][0]==0:
                aux=token_vecs_sum[idx_numerals[idx_4][1]:]
                B = torch.stack(aux).mean(axis=0).reshape(1, 768)
                if idx_numerals[idx_4][1]==0:
                    token_vecs_sum=torch.cat([B], axis=-2)
                else:
                    A = torch.stack(token_vecs_sum[:idx_numerals[idx_4][1]], dim=0).reshape(idx_numerals[idx_4][1], 768)
                    token_vecs_sum=torch.cat([A, B], axis=-2)
            else: 
                aux=token_vecs_sum[idx_numerals[idx_4][1]:-idx_numerals[idx_4][0]]
                C = torch.stack(token_vecs_sum[-idx_numerals[idx_4][0]:], dim=0).reshape(idx_numerals[idx_4][0], 768)
                B = torch.stack(aux).mean(axis=0).reshape(1, 768)
                if idx_numerals[idx_4][1]==0:
                    token_vecs_sum=torch.cat([B, C], axis=-2)
                else:
                    A = torch.stack(token_vecs_sum[:idx_numerals[idx_4][1]], dim=0).reshape(idx_numerals[idx_4][1], 768)
                    token_vecs_sum=torch.cat([A, B, C], axis=-2)
            embeds.append(token_vecs_sum)
            assert idx_numerals[idx_4][0]+idx_numerals[idx_4][1]+1==token_vecs_sum.shape[0]
    print("_____LAST TRANSFORMATION. OBTAINING X_____")  
    X=np.zeros((1, 768*12))
    for idx_5, embed in enumerate(embeds):
        aux=np.zeros((1, 768*12))
        for idx_2, word in enumerate(embed):
            n=5-idx_2+idx_numerals[idx_5][1]
            aux[0, n*768:(n+1)*768]=word
        X=np.concatenate([X, aux], axis=0)
    X=X[1:, :]
    np.save('bert/no/X_no_'+str(maxim), X)