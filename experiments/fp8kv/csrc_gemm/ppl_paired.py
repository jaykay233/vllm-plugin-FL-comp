import torch, statistics
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyarrow.parquet as pq
MODEL='/mnt/moark-models/Qwen3-0.6B'; dev='cuda'
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                       trust_remote_code=True).to(dev).eval()
W0=m.lm_head.weight.detach().clone()

pf=pq.ParquetFile('/mnt/moark-models/fineweb-edu-100b-shuffle/shard_00000.parquet')
docs=[]
for row in pf.iter_batches(batch_size=128, columns=['text']):
    for t in row.column(0).to_pylist():
        if t and len(t)>400: docs.append(t)
    if len(docs)>=256: break
allids=tok("\n\n".join(docs), return_tensors='pt')['input_ids'][0]
print("total tokens available", allids.numel(), flush=True)

WIN=1024; NW=40
starts=[i*WIN for i in range(NW) if (i+1)*WIN <= allids.numel()]
wins=[allids[s:s+WIN].unsqueeze(0).to(dev) for s in starts]
print("windows",len(wins),"of",WIN,"tokens", flush=True)

s=(W0.abs().amax().float()/448.0)
WQ=((W0.float()/s).to(torch.float8_e4m3fn).float()*s).to(torch.bfloat16)

def ppl(w):
    with torch.no_grad(): return float(m(w, labels=w).loss.exp())

res={'bf16':[], 'fp8':[]}
for i,w in enumerate(wins):
    m.lm_head.weight.data=W0;  a=ppl(w)
    m.lm_head.weight.data=WQ;  b=ppl(w)
    res['bf16'].append(a); res['fp8'].append(b)
    print("win %2d  bf16 %.4f  fp8 %.4f  delta %+.4f (%+.3f%%)"%(i,a,b,b-a,(b/a-1)*100), flush=True)

d=[b-a for a,b in zip(res['bf16'],res['fp8'])]
rel=[(b/a-1)*100 for a,b in zip(res['bf16'],res['fp8'])]
print()
print("bf16 mean ppl %.4f  (sd %.4f)"%(statistics.mean(res['bf16']),statistics.stdev(res['bf16'])))
print("fp8  mean ppl %.4f  (sd %.4f)"%(statistics.mean(res['fp8']),statistics.stdev(res['fp8'])))
print("paired delta mean %+.4f   sd %+.4f   sem %+.4f"%(statistics.mean(d),statistics.stdev(d),
      statistics.stdev(d)/len(d)**0.5))
mu=statistics.mean(rel); sd=statistics.stdev(rel); n=len(rel); sem=sd/n**0.5
import math
try:
    from scipy import stats as st
    tcrit=st.t.ppf(0.975, n-1)
except Exception:
    tcrit=2.0
print("paired relative  mean %+.4f%%  sd %+.4f%%  sem %+.4f%%"%(mu,sd,sem))
print("95%% CI  [%+.4f%%, %+.4f%%]  (t=%.2f, df=%d)"%(mu-tcrit*sem, mu+tcrit*sem, mu/sem, n-1))
print("windows where fp8 helped: %d / %d"%(sum(1 for r in rel if r<0), n))
