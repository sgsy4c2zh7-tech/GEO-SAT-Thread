#!/usr/bin/env python3
"""Build research-grade forecast verification products for SWIFT.

Scores *issued forecasts* after observations arrive.  This is intentionally
separate from model hindcasts so future papers can distinguish operational
forecast skill from in-sample fit quality.

Outputs
-------
docs/data/validation/latest.json
docs/data/validation/ui_validation_latest.json
docs/data/validation/history.json
docs/data/validation/cme-forecast-archive.json
docs/data/cme-calibration/model.json
docs/data/validation/monthly/YYYY-MM.json

Primary definitions
-------------------
Wind error = forecast - observed [km/s].
Kp error   = forecast - observed [Kp].
Bz-min error = forecast 3-h minimum - observed 3-h minimum [nT].
Brier score evaluates probability of any southward Bz in the target 3-h bin.
CME arrival verification uses an explicitly labelled NOAA solar-wind shock
proxy unless an independently observed arrival is available in the event data.
"""
from __future__ import annotations

import json, math, statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs'/'data'
OUT=DATA/'validation'; OUT.mkdir(parents=True,exist_ok=True)
MONTHLY=OUT/'monthly'; MONTHLY.mkdir(parents=True,exist_ok=True)
CME_CAL=DATA/'cme-calibration'; CME_CAL.mkdir(parents=True,exist_ok=True)

LATEST=OUT/'latest.json'; UI=OUT/'ui_validation_latest.json'; HISTORY=OUT/'history.json'
CME_ARCH=OUT/'cme-forecast-archive.json'; CME_MODEL=CME_CAL/'model.json'
MODEL_RETENTION_DAYS=730
NOMINAL_LEAD_HOURS=(24,48,72)
NOMINAL_LEAD_TOLERANCE_HOURS=4.5
SNAPSHOT_AGE_TOLERANCE_HOURS=10.0


def now_utc():return datetime.now(timezone.utc).replace(microsecond=0)
def iso(dt):
    if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
def parse(v):
    if not v:return None
    s=str(v).strip()
    if s.endswith('Z'):s=s[:-1]+'+00:00'
    if len(s)==19 and s[10]==' ':s=s.replace(' ','T')+'+00:00'
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
def load(p,d=None):
    try:return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d
    except:return d
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(obj,keys=('records','items','data','history','forecast','arrivals','events')):
    if isinstance(obj,list):return [x for x in obj if isinstance(x,dict)]
    if isinstance(obj,dict):
        for k in keys:
            if isinstance(obj.get(k),list):return [x for x in obj[k] if isinstance(x,dict)]
    return []
def pct(a,q,default=None):
    a=sorted(float(x) for x in a if x is not None and math.isfinite(float(x)))
    if not a:return default
    if len(a)==1:return a[0]
    pos=(len(a)-1)*q;lo=int(math.floor(pos));hi=int(math.ceil(pos));f=pos-lo
    return a[lo]*(1-f)+a[hi]*f
def mean(a,d=None):return sum(a)/len(a) if a else d
def rmse(a):return math.sqrt(sum(x*x for x in a)/len(a)) if a else None
def pearson(xs,ys):
    if len(xs)<3 or len(xs)!=len(ys):return None
    mx,my=mean(xs),mean(ys);dx=[x-mx for x in xs];dy=[y-my for y in ys]
    den=math.sqrt(sum(x*x for x in dx)*sum(y*y for y in dy))
    return sum(a*b for a,b in zip(dx,dy))/den if den>0 else None

def recs(paths):
    out=[]
    for p in paths:
        out.extend(rows(load(p,{}) or {}))
    return out

def load_wind_obs():
    by={}
    for r in recs([DATA/'noaa'/'wind_history.json',DATA/'noaa-wind'/'history.json',DATA/'noaa_wind_history.json']):
        t=parse(r.get('time') or r.get('time_tag'));v=num(r.get('speed') or r.get('v') or r.get('proton_speed'));n=num(r.get('density') or r.get('proton_density'))
        if t and v is not None:by[t.replace(second=0,microsecond=0)]={'time':t,'speed':v,'density':n}
    return [by[k] for k in sorted(by)]
def load_mag_obs():
    by={}
    for r in recs([DATA/'noaa'/'mag_history.json',DATA/'noaa-imf'/'history.json',DATA/'noaa_imf_history.json']):
        t=parse(r.get('time') or r.get('time_tag'));bz=num(r.get('bz') if r.get('bz') is not None else r.get('bz_gsm'));bt=num(r.get('bt'))
        if t and bz is not None:by[t.replace(second=0,microsecond=0)]={'time':t,'bz':bz,'bt':bt}
    return [by[k] for k in sorted(by)]
def load_kp_obs():
    by={}
    for r in recs([DATA/'noaa'/'kp_history.json',DATA/'noaa-kp'/'history.json',DATA/'noaa_kp_history.json']):
        t=parse(r.get('time') or r.get('time_tag'));k=num(r.get('kp') if r.get('kp') is not None else r.get('kp_raw'))
        if t and k is not None:by[t.replace(second=0,microsecond=0)]={'time':t,'kp':k}
    return [by[k] for k in sorted(by)]

def nearest(obs,t,key,max_min):
    if not obs:return None
    q=min(obs,key=lambda r:abs((r['time']-t).total_seconds()))
    return q.get(key) if abs((q['time']-t).total_seconds())<=max_min*60 else None
def window(obs,t,hours):return [r for r in obs if abs((r['time']-t).total_seconds())<=hours*3600]

def score_wind_archive(path,obs,pred_key='predicted_speed',model_default='SWIFT'):
    out=[]
    for a in rows(load(path,{}) or {}):
        tt=parse(a.get('target_time') or a.get('time'));it=parse(a.get('issued_at'))
        p=num(a.get(pred_key))
        if not tt or not it or p is None:continue
        o=nearest(obs,tt,'speed',50)
        if o is None:continue
        e=p-o
        out.append({'issued_at':iso(it),'target_time':iso(tt),'lead_hours':num(a.get('lead_hours'),(tt-it).total_seconds()/3600),'predicted':round(p,3),'observed':round(o,3),'error':round(e,3),'abs_error':round(abs(e),3),'model':a.get('model') or model_default})
    return out

def score_kp(obs):
    out=[]
    for a in rows(load(DATA/'swift-kp'/'forecast-archive.json',{}) or {}):
        tt=parse(a.get('target_time'));it=parse(a.get('issued_at'));p=num(a.get('predicted_kp'))
        if not tt or not it or p is None:continue
        o=nearest(obs,tt,'kp',100)
        if o is None:continue
        e=p-o
        out.append({'issued_at':iso(it),'target_time':iso(tt),'lead_hours':num(a.get('lead_hours'),(tt-it).total_seconds()/3600),'predicted':round(p,3),'observed':round(o,3),'error':round(e,3),'abs_error':round(abs(e),3),'model':a.get('model') or 'SWIFT-Kp','within_067':abs(e)<=.67,'within_1':abs(e)<=1.0})
    return out

def score_bz(obs):
    out=[]
    for a in rows(load(DATA/'swift-bz'/'forecast-archive.json',{}) or {}):
        tt=parse(a.get('target_time'));it=parse(a.get('issued_at'));pmin=num(a.get('bz_min_forecast'));prob=num(a.get('southward_bz_probability'))
        if not tt or not it or pmin is None or prob is None:continue
        w=window(obs,tt,1.5);vals=[r['bz'] for r in w if r.get('bz') is not None]
        if not vals:continue
        omin=min(vals);south=any(x<0 for x in vals);pr=clamp(prob/100,0,1);e=pmin-omin
        out.append({'issued_at':iso(it),'target_time':iso(tt),'lead_hours':num(a.get('lead_hours'),(tt-it).total_seconds()/3600),'predicted_bz_min':round(pmin,3),'observed_bz_min':round(omin,3),'error_bz_min':round(e,3),'abs_error_bz_min':round(abs(e),3),'southward_probability':round(prob,2),'observed_southward':south,'brier':round((pr-(1 if south else 0))**2,5)})
    return out

def filter_window(pairs,now,days):
    cut=now-timedelta(days=days);return [x for x in pairs if parse(x.get('target_time')) and parse(x['target_time'])>=cut]
def lead_day(x):return max(1,min(3,int(max(0,num(x.get('lead_hours'),0))//24)+1))
def wind_metrics(rr):
    if not rr:return {'count':0,'status':'learning'}
    e=[x['error'] for x in rr];ae=[abs(x) for x in e];p=[x['predicted'] for x in rr];o=[x['observed'] for x in rr]
    return {'count':len(rr),'mae':round(mean(ae),2),'median_ae':round(statistics.median(ae),2),'rmse':round(rmse(e),2),'bias':round(mean(e),2),'p90_ae':round(pct(ae,.90),2),'hit_rate_50':round(100*sum(x<=50 for x in ae)/len(ae),1),'hit_rate_75':round(100*sum(x<=75 for x in ae)/len(ae),1),'hit_rate_100':round(100*sum(x<=100 for x in ae)/len(ae),1),'pearson_r':round(pearson(p,o),3) if pearson(p,o) is not None else None}
def kp_metrics(rr):
    if not rr:return {'count':0,'status':'learning'}
    e=[x['error'] for x in rr];ae=[abs(x) for x in e];p=[x['predicted'] for x in rr];o=[x['observed'] for x in rr]
    return {'count':len(rr),'mae':round(mean(ae),3),'median_ae':round(statistics.median(ae),3),'rmse':round(rmse(e),3),'bias':round(mean(e),3),'p90_ae':round(pct(ae,.90),3),'hit_rate_067':round(100*sum(x<=.67 for x in ae)/len(ae),1),'hit_rate_1':round(100*sum(x<=1 for x in ae)/len(ae),1),'pearson_r':round(pearson(p,o),3) if pearson(p,o) is not None else None}
def bz_metrics(rr):
    if not rr:return {'count':0,'status':'learning'}
    e=[x['error_bz_min'] for x in rr];ae=[abs(x) for x in e];b=[x['brier'] for x in rr]
    hit=sum(((x['southward_probability']>=60)==bool(x['observed_southward'])) for x in rr)
    return {
        'count':len(rr),
        'bz_min_mae':round(mean(ae),3),
        'bz_min_bias':round(mean(e),3),
        'bz_min_rmse':round(rmse(e),3),
        'hit_rate_2nt':round(100*sum(x<=2.0 for x in ae)/len(ae),1),
        'hit_rate_3nt':round(100*sum(x<=3.0 for x in ae)/len(ae),1),
        'brier_score':round(mean(b),4),
        'direction_hit_rate_threshold60':round(100*hit/len(rr),1),
    }

def nominal_lead_pairs(rr,nominal_h,now,days=30):
    """One operational forecast per target, nearest the requested lead.

    SWIFT is issued every few hours, so a target can have many archived forecasts.
    We keep the forecast whose actual lead is closest to 24/48/72 h, within
    +/- 4.5 h, to avoid double counting a target in lead-specific skill.
    """
    cut=now-timedelta(days=days);best={}
    for x in rr:
        tt=parse(x.get('target_time'));it=parse(x.get('issued_at'));lh=num(x.get('lead_hours'))
        if not tt or not it or lh is None or tt<cut or tt>now:continue
        delta=abs(lh-nominal_h)
        if delta>NOMINAL_LEAD_TOLERANCE_HOURS:continue
        key=iso(tt)
        prev=best.get(key)
        score=(delta,-it.timestamp())
        if prev is None or score<prev[0]:best[key]=(score,x)
    return [best[k][1] for k in sorted(best)]

def kp_observed_band(v):
    """Requested non-overlapping Kp bands.

    The user's labels 1-4, 5-6, 6-7, 8-9 overlap at exactly Kp=6.
    To avoid double counting, use half-open operational bins:
      1 <= Kp < 5, 5 <= Kp < 6, 6 <= Kp < 8, 8 <= Kp <= 9.
    """
    x=num(v)
    if x is None:return None
    if 1.0<=x<5.0:return 'Kp 1-4'
    if 5.0<=x<6.0:return 'Kp 5-<6'
    if 6.0<=x<8.0:return 'Kp 6-7'
    if 8.0<=x<=9.0:return 'Kp 8-9'
    return None

def kp_range_skill(kp_pairs,now):
    rows=[]
    bands=['Kp 1-4','Kp 5-<6','Kp 6-7','Kp 8-9']
    selected_by_lead={}
    for h in NOMINAL_LEAD_HOURS:
        selected=nominal_lead_pairs(kp_pairs,h,now,30);selected_by_lead[h]=selected
        for band in bands:
            rr=[x for x in selected if kp_observed_band(x.get('observed'))==band]
            rows.append({'observed_range':band,'nominal_lead_hours':h,**kp_metrics(rr)})
    for band in bands:
        merged=[]
        for h in NOMINAL_LEAD_HOURS:merged.extend([x for x in selected_by_lead[h] if kp_observed_band(x.get('observed'))==band])
        rows.append({'observed_range':band,'nominal_lead_hours':'all',**kp_metrics(merged)})
    return rows

def _snapshot_group(rows_raw,now,age_h):
    groups={}
    for a in rows_raw:
        it=parse(a.get('issued_at'));tt=parse(a.get('target_time') or a.get('time'))
        if not it or not tt:continue
        groups.setdefault(iso(it),[]).append((it,tt,a))
    if not groups:return None
    want=now-timedelta(hours=age_h);candidates=[]
    for k,g in groups.items():
        it=g[0][0];d=abs((it-want).total_seconds())/3600
        candidates.append((d,-it.timestamp(),k,g))
    candidates.sort(key=lambda z:(z[0],z[1]))
    d,_,k,g=candidates[0]
    if d>SNAPSHOT_AGE_TOLERANCE_HOURS:return None
    g=[q for q in g if 0<=((q[1]-q[0]).total_seconds()/3600)<=72.5]
    return {'age_hours':age_h,'issued_at':k,'distance_from_nominal_age_h':round(d,2),'group':g}

def build_past_snapshots(now,kp_obs,mag_obs):
    kp_raw=rows(load(DATA/'swift-kp'/'forecast-archive.json',{}) or {})
    bz_raw=rows(load(DATA/'swift-bz'/'forecast-archive.json',{}) or {})
    out={'kp':[],'bz':[]}
    for age in NOMINAL_LEAD_HOURS:
        snap=_snapshot_group(kp_raw,now,age)
        if snap:
            rows_out=[];scored=[]
            for it,tt,a in snap['group']:
                p=num(a.get('predicted_kp'));
                if p is None:continue
                o=nearest(kp_obs,tt,'kp',100) if tt<=now else None
                q={'time':iso(tt),'lead_hours':round((tt-it).total_seconds()/3600,2),'predicted_kp':round(p,3),'model':a.get('model')}
                if o is not None:
                    e=p-o;q.update({'observed_kp':round(o,3),'error':round(e,3),'within_067':abs(e)<=.67,'within_1':abs(e)<=1.0});scored.append({'predicted':p,'observed':o,'error':e})
                rows_out.append(q)
            met=kp_metrics(scored) if scored else {'count':0,'status':'learning'}
            out['kp'].append({k:v for k,v in snap.items() if k!='group'}|{'skill':met,'rows':rows_out})
        snap=_snapshot_group(bz_raw,now,age)
        if snap:
            rows_out=[];scored=[]
            for it,tt,a in snap['group']:
                pmin=num(a.get('bz_min_forecast'));prob=num(a.get('southward_bz_probability'));bt=num(a.get('bt_forecast'))
                if pmin is None:continue
                w=window(mag_obs,tt,1.5) if tt<=now else [];vals=[r['bz'] for r in w if r.get('bz') is not None]
                q={'time':iso(tt),'lead_hours':round((tt-it).total_seconds()/3600,2),'predicted_bz_min':round(pmin,3),'bt_forecast':bt,'southward_probability':prob}
                if vals:
                    omin=min(vals);south=any(x<0 for x in vals);pr=clamp((prob or 0)/100,0,1);e=pmin-omin
                    q.update({'observed_bz_min':round(omin,3),'observed_southward':south,'error_bz_min':round(e,3),'within_2nt':abs(e)<=2.0,'within_3nt':abs(e)<=3.0})
                    scored.append({'error_bz_min':e,'brier':(pr-(1 if south else 0))**2,'southward_probability':prob or 0,'observed_southward':south})
                rows_out.append(q)
            met=bz_metrics(scored) if scored else {'count':0,'status':'learning'}
            out['bz'].append({k:v for k,v in snap.items() if k!='group'}|{'skill':met,'rows':rows_out})
    return out

def build_nominal_lead_skill(kp_pairs,bz_pairs,now):
    kp_rows=[];bz_rows=[];kp_pair_map={};bz_pair_map={}
    for h in NOMINAL_LEAD_HOURS:
        kr=nominal_lead_pairs(kp_pairs,h,now,30);br=nominal_lead_pairs(bz_pairs,h,now,30)
        kp_rows.append({'nominal_lead_hours':h,**kp_metrics(kr)})
        bz_rows.append({'nominal_lead_hours':h,**bz_metrics(br)})
        kp_pair_map[f'{h}h']=kr
        bz_pair_map[f'{h}h']=br
    return {
        'definition':{
            'lead_selection':f'one forecast per target nearest nominal 24/48/72 h within +/-{NOMINAL_LEAD_TOLERANCE_HOURS} h; target window last 30 d',
            'kp_primary_hit':'abs(predicted Kp - observed Kp) <= 1.0; strict hit <= 0.67',
            'bz_primary_hit':'abs(predicted 3-h Bz minimum - observed 3-h Bz minimum) <= 2 nT; secondary <= 3 nT',
            'bz_direction_hit':'southward probability >= 60% compared with whether any observed Bz < 0 in the target 3-h window',
            'kp_ranges':'non-overlapping: 1<=Kp<5, 5<=Kp<6, 6<=Kp<8, 8<=Kp<=9; Kp<1 excluded from range-stratified table',
        },
        'kp':kp_rows,'bz':bz_rows,'kp_by_observed_range':kp_range_skill(kp_pairs,now),
        'pairs':{'kp':kp_pair_map,'bz':bz_pair_map},
    }

def error_quantiles(rr,error_key):
    e=[num(x.get(error_key)) for x in rr];e=[x for x in e if x is not None]
    if len(e)<8:return {'count':len(e),'status':'learning','p10':None,'p50':None,'p90':None}
    return {'count':len(e),'p10':round(pct(e,.10),3),'p50':round(pct(e,.50),3),'p90':round(pct(e,.90),3)}

def cme_current_records():
    obj=load(DATA/'cme-arrivals'/'latest.json',{}) or {}
    return rows(obj)
def cme_event_id(r):return str(r.get('id') or r.get('activityID') or r.get('cme_id') or r.get('startTime') or r.get('start_time') or 'CME')
def cme_pred_time(r):return parse(r.get('arrival_time') or r.get('predicted_arrival') or r.get('estimated_arrival') or r.get('impact_time'))
def cme_start_time(r):return parse(r.get('time21_5') or r.get('startTime') or r.get('start_time') or r.get('activityStartTime') or r.get('time'))
def cme_speed(r):return num(r.get('effective_speed') or r.get('speed') or r.get('cme_speed') or r.get('speed_kms'))

def update_cme_archive(now):
    old=load(CME_ARCH,{'items':[]}) or {'items':[]};items=[x for x in old.get('items',[]) if isinstance(x,dict)]
    for r in cme_current_records():
        arr=cme_pred_time(r);st=cme_start_time(r);sp=cme_speed(r)
        if not arr:continue
        items.append({'archived_at':iso(now),'event_id':cme_event_id(r),'start_time':iso(st) if st else None,'predicted_arrival':iso(arr),'initial_speed_km_s':sp,'impact_class':r.get('impact_class') or r.get('impactClass'),'source':r.get('source')})
    cutoff=now-timedelta(days=MODEL_RETENTION_DAYS);dedup={}
    for x in items:
        arr=parse(x.get('predicted_arrival'))
        if not arr or arr<cutoff:continue
        dedup[(x.get('event_id'),x.get('predicted_arrival'))]=x
    out=sorted(dedup.values(),key=lambda x:x.get('predicted_arrival',''))
    save(CME_ARCH,{'updated_at':iso(now),'retention_days':MODEL_RETENTION_DAYS,'items':out})
    return out

def shock_proxy(pred,wind):
    base=[r for r in wind if pred-timedelta(hours=12)<=r['time']<=pred-timedelta(hours=3)]
    cand=[r for r in wind if pred-timedelta(hours=12)<=r['time']<=pred+timedelta(hours=12)]
    if not base or not cand:return None
    bv=statistics.median([r['speed'] for r in base]);bd=[r['density'] for r in base if r.get('density') is not None];bn=statistics.median(bd) if bd else None
    best=None
    for r in cand:
        dv=max(0,r['speed']-bv)/90.0;dens=0
        if bn is not None and r.get('density') is not None:dens=max(0,math.log((r['density']+.5)/(bn+.5)))
        score=dv+.55*dens
        if best is None or score>best['score']:best={'time':r['time'],'score':score,'baseline_speed':bv,'observed_speed':r['speed'],'density':r.get('density')}
    return best if best and best['score']>=.55 else None

def score_cme(cme_arch,wind,now):
    out=[]
    for a in cme_arch:
        pred=parse(a.get('predicted_arrival'))
        if not pred or pred>now-timedelta(hours=12):continue
        sh=shock_proxy(pred,wind)
        if not sh:continue
        err=(pred-sh['time']).total_seconds()/3600
        q=dict(a);q.update({'observed_arrival_proxy':iso(sh['time']),'arrival_error_hours':round(err,3),'abs_arrival_error_hours':round(abs(err),3),'shock_proxy_score':round(sh['score'],3),'baseline_speed_km_s':round(sh['baseline_speed'],2),'observed_shock_speed_km_s':round(sh['observed_speed'],2),'verification_method':'NOAA solar-wind speed/density shock proxy; not a manually adjudicated ICME boundary'})
        out.append(q)
    return out

def dbm_arrival_hours(v0,w,gamma,r0=21.5*695700,target=149597870,dt=600):
    if v0 is None:return None
    v=clamp(v0,250,3000);r=r0;s=0
    while r<target and s<10*86400:
        dv=v-w;a=-gamma*dv*abs(dv);v=clamp(v+a*dt,250,3000);r+=v*dt;s+=dt
    return s/3600

def fit_gamma(cme_pairs):
    usable=[]
    for x in cme_pairs:
        st=parse(x.get('start_time'));obs=parse(x.get('observed_arrival_proxy'));v0=num(x.get('initial_speed_km_s'));w=num(x.get('baseline_speed_km_s'))
        if st and obs and v0 is not None and w is not None and obs>st:usable.append((st,obs,v0,w))
    if len(usable)<5:return {'gamma_km_inv':3.0e-8,'count':len(usable),'status':'learning','method':'default DBM gamma until >=5 shock-proxy events'}
    best=None
    for i in range(30):
        g=(.05+i*(1.20-.05)/29)*1e-7;errs=[]
        for st,obs,v0,w in usable:
            h=dbm_arrival_hours(v0,w,g);pred=st+timedelta(hours=h);errs.append(abs((pred-obs).total_seconds()/3600))
        mae=mean(errs)
        if best is None or mae<best[0]:best=(mae,g,statistics.median(errs))
    return {'gamma_km_inv':best[1],'count':len(usable),'mae_hours':round(best[0],3),'median_ae_hours':round(best[2],3),'status':'calibrated','method':'grid-search DBM gamma minimizing shock-proxy arrival MAE'}

def cme_metrics(rr):
    if not rr:return {'count':0,'status':'learning'}
    ae=[x['abs_arrival_error_hours'] for x in rr];e=[x['arrival_error_hours'] for x in rr]
    return {'count':len(rr),'mae_hours':round(mean(ae),2),'median_ae_hours':round(statistics.median(ae),2),'bias_hours':round(mean(e),2),'within_6h':round(100*sum(x<=6 for x in ae)/len(ae),1),'within_12h':round(100*sum(x<=12 for x in ae)/len(ae),1)}


def load_ch_history():
    obj=load(DATA/'coronal-holes'/'history.json',{}) or {}
    out=[]
    for snap in obj.get('items',[]) if isinstance(obj,dict) else []:
        issued=parse(snap.get('time'))
        if not issued:continue
        for e in snap.get('earth_impacts') or []:
            at=parse(e.get('arrival_time'))
            pv=num(e.get('predicted_peak_speed_km_s_prior'))
            if at and pv is not None:
                out.append({'issued_at':issued,'target_time':at,'predicted_peak_speed':pv,'confidence':num(e.get('confidence'),0),
                            'source_id':e.get('source_id'),'earth_facing_score':num(e.get('earth_facing_score')),
                            'model':snap.get('model') or 'SWIFT-CH'})
    dedup={}
    for x in out:
        k=(iso(x['issued_at'])[:13],x.get('source_id'),iso(x['target_time'])[:13])
        dedup[k]=x
    return list(dedup.values())

def score_ch_hss(chrows,wind,now):
    out=[]
    for x in chrows:
        tt=x['target_time'];it=x['issued_at']
        if tt>now-timedelta(hours=20):continue
        pre=[r['speed'] for r in wind if tt-timedelta(hours=24)<=r['time']<=tt-timedelta(hours=6)]
        obs=[r['speed'] for r in wind if tt-timedelta(hours=18)<=r['time']<=tt+timedelta(hours=30)]
        if not pre or not obs:continue
        baseline=statistics.median(pre);peak=max(obs);p=x['predicted_peak_speed'];e=p-peak
        out.append({'issued_at':iso(it),'target_time':iso(tt),'lead_hours':(tt-it).total_seconds()/3600,
                    'predicted_peak_speed':round(p,2),'observed_peak_speed':round(peak,2),
                    'baseline_speed':round(baseline,2),'observed_delta_v':round(peak-baseline,2),
                    'error':round(e,2),'abs_error':round(abs(e),2),'confidence':x.get('confidence'),
                    'earth_facing_score':x.get('earth_facing_score'),'source_id':x.get('source_id')})
    return out

def ch_metrics(rr):
    if not rr:return {'count':0,'status':'learning'}
    e=[x['error'] for x in rr];ae=[abs(x) for x in e]
    return {'count':len(rr),'peak_speed_mae':round(mean(ae),2),'peak_speed_bias':round(mean(e),2),
            'peak_speed_rmse':round(rmse(e),2),'within_75kms':round(100*sum(x<=75 for x in ae)/len(ae),1),
            'within_100kms':round(100*sum(x<=100 for x in ae)/len(ae),1)}

def model_blocks(pairs,metric_fn,now):
    return {'last_7d':metric_fn(filter_window(pairs,now,7)),'last_30d':metric_fn(filter_window(pairs,now,30)),'by_lead_day_30d':[{'lead_day':d,**metric_fn([x for x in filter_window(pairs,now,30) if lead_day(x)==d])} for d in (1,2,3)]}

def main():
    now=now_utc();wind=load_wind_obs();mag=load_mag_obs();kp=load_kp_obs()
    wind_pairs=score_wind_archive(DATA/'swift-wind'/'forecast-archive.json',wind)
    enlil_pairs=score_wind_archive(DATA/'enlil'/'forecast-archive.json',wind,pred_key='v_r',model_default='NOAA WSA-ENLIL L1')
    kp_pairs=score_kp(kp);bz_pairs=score_bz(mag)
    nominal_skill=build_nominal_lead_skill(kp_pairs,bz_pairs,now)
    past_snapshots=build_past_snapshots(now,kp,mag)
    cme_arch=update_cme_archive(now);cme_pairs=score_cme(cme_arch,wind,now);gamma=fit_gamma(cme_pairs);save(CME_MODEL,{'updated_at':iso(now),**gamma})
    ch_pairs=score_ch_hss(load_ch_history(),wind,now)

    models={
        'swift_wind':model_blocks(wind_pairs,wind_metrics,now),
        'coronal_hole_hss':model_blocks(ch_pairs,ch_metrics,now),
        'enlil_wind':model_blocks(enlil_pairs,wind_metrics,now),
        'swift_kp':model_blocks(kp_pairs,kp_metrics,now),
        'swift_bz':model_blocks(bz_pairs,bz_metrics,now),
        'cme_arrival':{'last_30d':cme_metrics(filter_window([{**x,'target_time':x['predicted_arrival']} for x in cme_pairs],now,30)),'all_available':cme_metrics(cme_pairs),'dbm_gamma_fit':gamma},
    }
    unc={
        'wind_by_lead_day':[{'lead_day':d,**error_quantiles([x for x in filter_window(wind_pairs,now,30) if lead_day(x)==d],'error')} for d in (1,2,3)],
        'kp_by_lead_day':[{'lead_day':d,**error_quantiles([x for x in filter_window(kp_pairs,now,30) if lead_day(x)==d],'error')} for d in (1,2,3)],
        'bz_min_by_lead_day':[{'lead_day':d,**error_quantiles([x for x in filter_window(bz_pairs,now,30) if lead_day(x)==d],'error_bz_min')} for d in (1,2,3)],
        'definition':'p10/p50/p90 empirical forecast-minus-observation residuals over previous 30 days; add these residual quantiles to point forecasts for empirical intervals',
    }
    latest={'updated_at':iso(now),'retention_months_excel':24,'models':models,'uncertainty':unc,'nominal_lead_skill':nominal_skill,'past_forecast_snapshots':past_snapshots,'methodology':{'wind':'true issued-forecast verification against NOAA RTSW; error=forecast-observed','kp':'true issued-forecast verification against NOAA planetary K; nominal lead skill uses one forecast per target at ~24/48/72 h','bz':'issued probability/minimum verified against NOAA IMF 3-h target window; nominal lead skill uses one forecast per target at ~24/48/72 h','cme':'shock proxy from NOAA speed/density; clearly not manually adjudicated ICME boundary'},'pairs':{'wind':wind_pairs[-6000:],'enlil_wind':enlil_pairs[-6000:],'kp':kp_pairs[-6000:],'bz':bz_pairs[-6000:],'cme':cme_pairs[-1000:]}}
    save(LATEST,latest)

    ui={'updated_at':iso(now),'models':models,'uncertainty':unc,'nominal_lead_skill':nominal_skill,'past_forecast_snapshots':past_snapshots,'dbm_gamma':gamma,'counts':{'wind_pairs':len(wind_pairs),'enlil_pairs':len(enlil_pairs),'kp_pairs':len(kp_pairs),'bz_pairs':len(bz_pairs),'cme_pairs':len(cme_pairs)}}
    save(UI,ui)

    hist=load(HISTORY,{'history':[]}) or {'history':[]};h=[x for x in hist.get('history',[]) if isinstance(x,dict)]
    h=[x for x in h if parse(x.get('time')) and parse(x['time'])<now-timedelta(hours=2)]
    h.append({'time':iso(now),'swift_wind_30d':models['swift_wind']['last_30d'],'enlil_wind_30d':models['enlil_wind']['last_30d'],'kp_30d':models['swift_kp']['last_30d'],'bz_30d':models['swift_bz']['last_30d'],'cme_all':models['cme_arrival']['all_available'],'dbm_gamma':gamma})
    cutoff=now-timedelta(days=MODEL_RETENTION_DAYS);h=[x for x in h if parse(x.get('time')) and parse(x['time'])>=cutoff]
    save(HISTORY,{'updated_at':iso(now),'retention_days':MODEL_RETENTION_DAYS,'history':h})

    month=now.strftime('%Y-%m');ms=datetime(now.year,now.month,1,tzinfo=timezone.utc);me=(datetime(now.year+(now.month==12),(now.month%12)+1,1,tzinfo=timezone.utc))
    def month_rows(rr,time_key='target_time'):
        return [x for x in rr if parse(x.get(time_key)) and ms<=parse(x[time_key])<me]
    save(MONTHLY/f'{month}.json',{'updated_at':iso(now),'month':month,'models':models,'uncertainty':unc,'nominal_lead_skill':nominal_skill,'past_forecast_snapshots':past_snapshots,'pairs':{'wind':month_rows(wind_pairs),'enlil_wind':month_rows(enlil_pairs),'kp':month_rows(kp_pairs),'bz':month_rows(bz_pairs),'cme':month_rows(cme_pairs,'predicted_arrival')}})
    print(json.dumps({'swift_wind_30d':models['swift_wind']['last_30d'],'kp_30d':models['swift_kp']['last_30d'],'bz_30d':models['swift_bz']['last_30d'],'cme':models['cme_arrival']['all_available'],'gamma':gamma},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
