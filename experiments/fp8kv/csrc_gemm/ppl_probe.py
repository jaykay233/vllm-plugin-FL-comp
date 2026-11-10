import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL='/mnt/moark-models/Qwen3-0.6B'
dev='cuda'
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                       trust_remote_code=True).to(dev).eval()
W0=m.lm_head.weight.detach()

# corpus: fineweb-edu shard, a few dozen documents
import pyarrow.parquet as pq
pf=pq.ParquetFile('/mnt/moark-models/fineweb-edu-100b-shuffle/shard_00000.parquet')
col=pf.schema_arrow.names
txtcol=[c for c in col if c.lower() in ('text','content','raw_content')][0]
docs=[]
for i,row in enumerate(pf.iter_batches(batch_size=64, columns=[txtcol])):
    for t in row.column(0).to_pylist():
        if t and len(t)>400: docs.append(t)
    if len(docs)>=24: break
print("docs",len(docs),"col",txtcol, flush=True)

WINF=1024  # compute ppl over a fixed token budget
ids=tok("\n\n".join(docs), return_tensors='pt', truncation=True, max_length=WINF)['input_ids'].to(dev)
print("eval tokens",ids.shape[1], flush=True)

def quant(kind):
    if kind=='bf16': return W0
    if kind=='fp8_tensor':
        s=(W0.abs().amax().float()/448.0)
        return (W0.float()/s).to(torch.float8_e4m3fn).float()*s
def ppl_with(kind):
    if kind=='bf16': m.lm_head.weight.data=W0
    else: m.lm_head.weight.data=quant(kind).to(torch.bfloat16)
    with torch.no_grad():
        out=m(ids, labels=ids)
    return float(out.loss.exp())

for kind in ('bf16','fp8_tensor'):
    p=ppl_with(kind)
    print("%-12s ppl = %.4f"%(kind,p), flush=True)
