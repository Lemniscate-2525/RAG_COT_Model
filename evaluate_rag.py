import argparse, json, math
import numpy as np, requests, faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def clean(s): return " ".join(BeautifulSoup(s or "", "html.parser").get_text(" ").split())
def split(s, size=900, overlap=150):
    s=" ".join(s.split()); out=[]; start=0
    while start<len(s):
        end=min(start+size,len(s)); out.append(s[start:end].strip())
        if end==len(s): break
        start=end-overlap
    return [x for x in out if x]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--site",required=True); p.add_argument("--k",type=int,default=3); p.add_argument("--pages",type=int,default=3); a=p.parse_args()
    cases=json.load(open(a.dataset,encoding="utf-8")); docs=[]
    for page in range(1,a.pages+1):
        r=requests.get(a.site.rstrip("/")+"/wp-json/wp/v2/posts",params={"page":page,"per_page":100,"_fields":"id,title,content"},timeout=20)
        if r.status_code==400 and page>1: break
        r.raise_for_status(); batch=r.json()
        for post in batch:
            text=clean(post.get("title",{}).get("rendered",""))+" "+clean(post.get("content",{}).get("rendered",""))
            docs += [{"id":int(post["id"]),"text":c} for c in split(text)]
        if len(batch)<100: break
    if not docs: raise SystemExit("No posts fetched.")
    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec=np.asarray(model.encode([d["text"] for d in docs],normalize_embeddings=True),dtype="float32")
    index=faiss.IndexFlatIP(vec.shape[1]); index.add(vec); totals=[0.,0.,0.,0.]; count=0
    for case in cases:
        relevant=set(map(int,case["relevant_post_ids"]))
        if not relevant: continue
        q=np.asarray(model.encode([case["question"]],normalize_embeddings=True),dtype="float32")
        _,ids=index.search(q,min(max(1,a.k),index.ntotal)); ranked=[docs[int(i)]["id"] for i in ids[0] if 0<=i<len(docs)]
        hit=float(any(x in relevant for x in ranked)); recall=len(set(ranked)&relevant)/len(relevant)
        rr=next((1/(i+1) for i,x in enumerate(ranked) if x in relevant),0.)
        dcg=sum(1/math.log2(i+2) for i,x in enumerate(ranked) if x in relevant); ideal=sum(1/math.log2(i+2) for i in range(min(len(relevant),a.k)))
        ndcg=dcg/ideal if ideal else 0.; vals=[hit,recall,rr,ndcg]
        totals=[x+y for x,y in zip(totals,vals)]; count+=1
        print(case["question"], dict(zip(["hit_rate","recall","mrr","ndcg"],[round(x,3) for x in vals])))
    if not count: raise SystemExit("No labelled evaluation cases.")
    print("Macro averages @",a.k,dict(zip(["hit_rate","recall","mrr","ndcg"],[round(x/count,4) for x in totals])))
    print("Retrieval metrics only; they do not measure generation faithfulness or reasoning correctness.")
if __name__=="__main__": main()
