import torch, os
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL='/mnt/moark-models/Qwen3-0.6B'
dev='cuda'
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                       trust_remote_code=True).to(dev).eval()
W=m.lm_head.weight.detach()            # [V, H] = [151936, 1024]
print("lm_head", tuple(W.shape), W.dtype)
print("row max/min spread: %.3f"%(W.abs().amax(dim=1).max()/W.abs().amax(dim=1).median()),
      " (per-tensor quant ignores this)")
# real activations: run a prompt, take hidden states entering lm_head
texts=["The capital of France is","def fibonacci(n):\n    ","1, 2, 3, 5, 8, 13,","The quick brown fox jumps over the lazy dog. "*8,
       "In machine learning, quantization refers to"]
acts=[]
for t in texts:
    ids=tok(t,return_tensors='pt').to(dev)
    out=m(**ids, output_hidden_states=True)
    acts.append(out.hidden_states[-1][0,-1].float())   # last token hidden
X=torch.stack(acts)                                     # [T, H]
print("acts", tuple(X.shape), "amax %.2f"%X.abs().max())
def q(scale_kind):
    if scale_kind=='tensor':
        s=(W.abs().amax().float()/448.0)
    elif scale_kind=='row':
        s=(W.abs().amax(dim=1).float().clamp(min=1e-12)/448.0).unsqueeze(1)
    elif scale_kind=='group128':
        wf=W.float().view(W.shape[0], -1, 128)
        s=(wf.abs().amax(dim=2).clamp(min=1e-12)/448.0)
        wq=(wf/s.unsqueeze(2)).to(torch.float8_e4m3fn)
        return (wq.float().view(W.shape[0],-1)*s.repeat_interleave(128,dim=1))
    wq=(W.float()/s).to(torch.float8_e4m3fn)
    return (wq.float()*s)
ref=(X@W.float().t())
print("\n%-10s %10s %10s %10s %10s"%("scheme","rel Fro","max|dlogit|","top1 agree","cos"))
for kind in ('tensor','row','group128'):
    Wd=q(kind)
    got=X@Wd.t()
    rel=((Wd-W.float()).norm()/W.float().norm()).item()
    d=(got-ref).abs().max().item()
    a=ref.argmax(-1); b=got.argmax(-1)
    cos=torch.nn.functional.cosine_similarity(got,ref,dim=-1).mean().item()
    print("%-10s %10.5f %10.3f %10s %10.6f"%(kind,rel,d,
          "%d/%d"%(int((a==b).sum()),len(a)),cos))
