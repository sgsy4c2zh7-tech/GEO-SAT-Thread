#!/usr/bin/env python3
"""Blend SWIFT solar-wind forecast with the official NOAA WSA-ENLIL L1 time series.

This is deliberately a *verification-weighted ensemble*, not a replacement of
SWIFT by ENLIL.  The weight is lead-dependent and, when previous verification
exists, reduced when ENLIL has recently performed poorly.

The script edits docs/data/swift-wind/latest.json in place and updates the
current run in forecast-archive.json so later verification scores the forecast
that users actually saw.
"""
from __future__ import annotations

import json, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs'/'data'
SWIFT=DATA/'swift-wind'/'latest.json'
ARCH=DATA/'swift-wind'/'forecast-archive.json'
ENLIL=DATA/'enlil'/'latest.json'
VALID=DATA/'validation'/'latest.json'


def load(p:Path,d=None):
    try:return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d
    except:return d

def save(p:Path,o:Any):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def parse(v):
    if not v:return None
    s=str(v).strip()
    if s.endswith('Z'):s=s[:-1]+'+00:00'
    try:
        d=datetime.fromisoformat(s)
        if d.tzinfo is None:d=d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except:return None
def num(v,d=None):
    try:
        x=float(v);return x if math.isfinite(x) else d
    except:return d
def clamp(x,a,b):return max(a,min(b,x))

def nearest(rows,t,max_h=2.0):
    best=None;bd=1e99
    for r in rows:
        rt=parse(r.get('time') or r.get('target_time'))
        if not rt:continue
        dd=abs((rt-t).total_seconds())
        if dd<bd:best,bd=r,dd
    return best if best and bd<=max_h*3600 else None

def enlil_skill_factor():
    v=load(VALID,{}) or {}
    e=((v.get('models') or {}).get('enlil_wind') or {}).get('last_30d') or {}
    mae=num(e.get('mae'))
    n=int(e.get('count',0) or 0)
    if mae is None or n<20:return 0.55
    # 40 km/s -> very useful, 180+ km/s -> small influence.
    return clamp((190-mae)/160,0.15,0.9)

def base_weight(lead_h):
    if lead_h<=12:return .24
    if lead_h<=24:return .34
    if lead_h<=48:return .46
    return .18

def main():
    swift=load(SWIFT,{}) or {}; enlil=load(ENLIL,{}) or {}
    fc=swift.get('forecast') or []
    erows=enlil.get('forecast') or enlil.get('records') or []
    if not fc or not erows:
        print('No SWIFT or ENLIL forecast rows; leaving SWIFT unchanged')
        return
    sf=enlil_skill_factor(); now=datetime.now(timezone.utc)
    changed={}
    for r in fc:
        t=parse(r.get('time') or r.get('target_time'))
        if not t:continue
        er=nearest(erows,t,2.0)
        sv=num(r.get('speed') or r.get('swift_cme_enhanced_speed') or r.get('predicted_speed'))
        ev=num(er.get('v_r') if er else None)
        if sv is None or ev is None:continue
        lead=max(0,(t-now).total_seconds()/3600)
        w=clamp(base_weight(lead)*sf,0.05,0.55)
        ens=(1-w)*sv+w*ev
        r['swift_only_speed']=round(sv,2)
        r['enlil_speed']=round(ev,2)
        r['enlil_weight']=round(w,3)
        r['ensemble_speed']=round(ens,2)
        # Downstream Bz/Kp loaders prefer `speed`, then swift_cme_enhanced_speed.
        r['speed']=round(ens,2)
        r['swift_cme_enhanced_speed']=round(ens,2)
        r['source']='SWIFT Wind AI + NOAA WSA-ENLIL ensemble'
        changed[t.isoformat()]=r

    # Keep records synchronized for future UI readers that prefer records.
    for r in swift.get('records') or []:
        t=parse(r.get('time'))
        if not t:continue
        q=changed.get(t.isoformat())
        if q and str(r.get('kind','')).lower()=='forecast':
            for k in ('swift_only_speed','enlil_speed','enlil_weight','ensemble_speed','speed','swift_cme_enhanced_speed','source'):
                if k in q:r[k]=q[k]

    if fc:swift['current']=fc[0]
    swift['model']='SWIFT-Wind-AI-v0.7-ENLIL-ensemble'
    swift['ensemble']={
        'official_source':'https://services.swpc.noaa.gov/json/enlil_time_series.json',
        'enlil_skill_factor':round(sf,3),
        'method':'lead-dependent linear ensemble; ENLIL weight scaled by previous 30d ENLIL verification',
    }
    save(SWIFT,swift)

    # Update only the most recently issued archive entries, preserving original SWIFT speed.
    a=load(ARCH,{'items':[]}) or {'items':[]}; items=a.get('items',[])
    issue_times=[parse(x.get('issued_at')) for x in items if parse(x.get('issued_at'))]
    latest_issue=max(issue_times) if issue_times else None
    if latest_issue and abs((now-latest_issue).total_seconds())<=6*3600:
        for x in items:
            it=parse(x.get('issued_at'));tt=parse(x.get('target_time'))
            if not it or not tt or abs((it-latest_issue).total_seconds())>60:continue
            q=changed.get(tt.isoformat())
            if not q:continue
            old=num(x.get('predicted_speed'))
            if old is not None:x['swift_only_speed']=old
            x['enlil_speed']=q.get('enlil_speed');x['enlil_weight']=q.get('enlil_weight')
            x['predicted_speed']=q.get('ensemble_speed');x['model']='SWIFT-Wind-AI-v0.7-ENLIL-ensemble'
    a['updated_at']=now.replace(microsecond=0).isoformat().replace('+00:00','Z')
    save(ARCH,a)
    print(json.dumps({'model':swift['model'],'forecast_rows':len(fc),'blended_rows':len(changed),'enlil_skill_factor':sf},indent=2))

if __name__=='__main__':main()
