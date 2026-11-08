"""Compare mem_growth_bytes_per_s of query v1 (as exported in each run's queries.json) with the
current v2 (live scrape targets only) over every lab run still in Prometheus retention.
Usage (repo root): python loadlab/runner/check_query_v2.py   (needs 127.0.0.1:9090)"""
import json, sys, glob, os, math
import requests
sys.path.insert(0, 'loadlab/runner')
import promexport as p
rows_tot=rows_diff=rows_big=0; per=[]
for d in sorted(glob.glob('loadlab/runs/*/')):
    if not os.path.exists(d+'queries.json'): continue
    q1=json.load(open(d+'queries.json'))['deploy']['mem_growth_bytes_per_s']
    sc=json.load(open(d+'scenario.json'))['_run']; ns=sc['namespace']; a,b=sc['export_window']
    q2=p.deploy_queries(ns)['mem_growth_bytes_per_s']
    def get(q):
        r=requests.get('http://127.0.0.1:9090/api/v1/query_range',params={'query':q,'start':a,'end':b,'step':5},timeout=60).json()['data']['result']
        return {round(float(t)):float(v) for t,v in (r[0]['values'] if r else [])}
    m1,m2=get(q1),get(q2)
    if not m1: per.append((os.path.basename(d[:-1]),'NO DATA')); continue
    keys=set(m1)|set(m2)
    diff=[k for k in keys if (m1.get(k) is None)!=(m2.get(k) is None) or (m1.get(k) is not None and m2.get(k) is not None and abs(m1[k]-m2[k])>1)]
    big=[k for k in diff if abs((m1.get(k) or 0)-(m2.get(k) or 0))>100e3]
    rows_tot+=len(m1); rows_diff+=len(diff); rows_big+=len(big)
    per.append((os.path.basename(d[:-1]),len(m1),len(diff),len(big)))
for x in per: print(x)
print('TOTAL points',rows_tot,'differ',rows_diff,'differ>100KB/s',rows_big)
