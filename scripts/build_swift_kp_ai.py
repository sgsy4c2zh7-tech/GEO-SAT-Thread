#!/usr/bin/env python3
"""SWIFT Kp AI v0.6 with lead-time bias correction and Bz-confidence gating.

Key changes:
- Keeps the fitted nonlinear Kp feature model.
- Reduces unverified Bz influence until Bz forecast archive has enough scored cases.
- Applies learned lead-day bias correction from the previous 30-day forecast archive.
- Merges old/new NOAA observed Kp, wind and IMF history paths.
"""
from __future__ import annotations

import json, math, statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = DATA / "swift-kp"
OUT.mkdir(parents=True, exist_ok=True)

LATEST=OUT/"latest.json"; COEF=OUT/"coefficients.json"; VERIF=OUT/"verification.json"
ARCHIVE=OUT/"forecast-archive.json"; LEAD=OUT/"leadtime-skill.json"; HISTORY=OUT/"history.json"
TXT=OUT/"forecast.txt"; INDEX=OUT/"index.json"

NOW=datetime.now(timezone.utc)
FORECAST_DAYS=3
STEP_H=3


def iso(dt): return dt.astimezone(timezone.utc).isoformat()
def parse(v):
    if not v: return None
    s=str(v).strip()
    if s.endswith("Z"): s=s[:-1]+"+00:00"
    if len(s)==19 and s[10]==" ": s=s.replace(" ","T")+"+00:00"
    try:
        d=datetime.fromisoformat(s)
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except: return None
def num(v,d=None):
    try:
        x=float(v); return x if math.isfinite(x) else d
    except: return d
def clamp(x,a,b): return max(a,min(b,x))
def sigmoid(x): return 1/(1+math.exp(-clamp(x,-50,50)))
def softplus(x): return math.log1p(math.exp(clamp(x,-40,40)))
def mean(a,d=0): return sum(a)/len(a) if a else d
def median(a,d=0):
    a=[x for x in a if x is not None and math.isfinite(x)]
    return statistics.median(a) if a else d
def load(p,d=None):
    if not p.exists(): return d
    try:return json.loads(p.read_text(encoding="utf-8"))
    except:return d
def save(p,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
def rows(obj):
    if isinstance(obj,list): return [x for x in obj if isinstance(x,dict)]
    if isinstance(obj,dict):
        for k in ("records","items","data","history","forecast"):
            if isinstance(obj.get(k),list): return [x for x in obj[k] if isinstance(x,dict)]
    return []


def merge_series(paths, value_getter, valid):
    by={}
    for p in paths:
        for r in rows(load(p,{}) or {}):
            t=parse(r.get("time") or r.get("time_tag") or r.get("timestamp"))
            v=value_getter(r)
            if t and v is not None and valid(v):
                by[t.replace(second=0,microsecond=0)]={"time":t,"value":v,"raw":r}
    return [by[k] for k in sorted(by)]


kp_obs=merge_series(
    [DATA/"noaa-kp"/"history.json",DATA/"noaa"/"kp_history.json",DATA/"noaa_kp_history.json"],
    lambda r:num(r.get("kp") if r.get("kp") is not None else r.get("kp_raw")),
    lambda x:0<=x<=9,
)
wind_obs=merge_series(
    [DATA/"noaa-wind"/"history.json",DATA/"noaa"/"wind_history.json",DATA/"noaa_wind_history.json"],
    lambda r:num(r.get("speed") or r.get("proton_speed") or r.get("v")),
    lambda x:200<=x<=1200,
)
imf_raw=[]
for p in [DATA/"noaa-imf"/"history.json",DATA/"noaa"/"mag_history.json",DATA/"noaa_imf_history.json"]:
    for r in rows(load(p,{}) or {}):
        t=parse(r.get("time") or r.get("time_tag"))
        bz=num(r.get("bz") if r.get("bz") is not None else r.get("bz_gsm"))
        bt=num(r.get("bt"))
        if t and bz is not None:
            imf_raw.append({"time":t,"bz":bz,"bt":bt})
imf_by={r["time"].replace(second=0,microsecond=0):r for r in imf_raw}
imf_obs=[imf_by[k] for k in sorted(imf_by)]


def nearest(series,t,hours=1.7):
    if not series:return None
    r=min(series,key=lambda x:abs((x["time"]-t).total_seconds()))
    return r if abs((r["time"]-t).total_seconds())<=hours*3600 else None


def obs_speed(t):
    r=nearest(wind_obs,t,2.0); return r["value"] if r else None


def obs_imf(t):
    rr=[r for r in imf_obs if abs((r["time"]-t).total_seconds())<=1.5*3600]
    if not rr:return {"south_prob":35,"bz_min":-2,"bt":5}
    bz=[r["bz"] for r in rr]; bt=[r["bt"] for r in rr if r["bt"] is not None]
    return {"south_prob":100*sum(x<0 for x in bz)/len(bz),"bz_min":min(bz),"bt":median(bt,5)}


wind_model=load(DATA/"swift-wind"/"latest.json",{}) or {}
wind_fc=[]
for r in wind_model.get("forecast") or wind_model.get("records") or []:
    t=parse(r.get("time") or r.get("target_time") or r.get("start_time"))
    sp=num(r.get("speed") or r.get("swift_cme_enhanced_speed") or r.get("predicted_speed"))
    if t and sp is not None: wind_fc.append({"time":t,"speed":sp})
wind_fc.sort(key=lambda x:x["time"])


def fc_speed(t):
    r=nearest(wind_fc,t,2.2)
    if r:return r["speed"]
    vals=[x["value"] for x in wind_obs if NOW-timedelta(hours=6)<=x["time"]<=NOW]
    return median(vals,400)


bz_model=load(DATA/"swift-bz"/"latest.json",{}) or {}
bz_rows=[]
for r in bz_model.get("forecast") or []:
    t=parse(r.get("time"))
    if t:
        bz_rows.append({
            "time":t,
            "south_prob":num(r.get("southward_bz_probability"),35),
            "bz_min":num(r.get("bz_min_forecast"),-3),
            "bt":num(r.get("bt_forecast"),6),
            "row_conf":num(r.get("confidence"),0.4),
        })
bz_rows.sort(key=lambda x:x["time"])
ready=bz_model.get("readiness") or (bz_model.get("verification") or {}).get("readiness") or {}
bz_model_conf=num(ready.get("kp_input_confidence"),None)
if bz_model_conf is None:
    # Legacy output: do not trust an unverified/persistence-only product too strongly.
    ver=(bz_model.get("verification") or {}).get("last_7d") or {}
    cnt=int(ver.get("count",0) or 0)
    method=str(ver.get("method","")).lower()
    bz_model_conf=0.30 if ("persistence" in method or cnt<32) else clamp(0.25+cnt/200*0.5,0.25,0.8)


def fc_bz(t):
    r=nearest(bz_rows,t,2.2)
    if not r:
        o=obs_imf(NOW); r={"south_prob":o["south_prob"],"bz_min":o["bz_min"],"bt":o["bt"],"row_conf":0.3}
    gate=clamp(bz_model_conf*r["row_conf"],0.10,0.90)
    # Pull uncertain forecasts back toward climatological neutral values.
    south=35 + gate*(r["south_prob"]-35)
    south_bz=gate*max(0,-r["bz_min"])
    bt=5 + gate*(r["bt"]-5)
    return {"south_prob":south,"south_bz":south_bz,"bt":bt,"gate":gate,"raw":r}


def dvdt(t, f):
    a=f(t-timedelta(hours=3)); b=f(t)
    return 0 if a is None or b is None else (b-a)/3


def features(t, forecast=True):
    v=fc_speed(t) if forecast else obs_speed(t)
    v=400 if v is None else v
    if forecast:
        bz=fc_bz(t); south_prob=bz["south_prob"]/100; south_bz=bz["south_bz"]; bt=bz["bt"]; gate=bz["gate"]
    else:
        b=obs_imf(t); south_prob=b["south_prob"]/100; south_bz=max(0,-b["bz_min"]); bt=b["bt"]; gate=1.0
    d=dvdt(t,fc_speed if forecast else obs_speed)
    v_excess=softplus((v-360)/75)
    hss=sigmoid((v-500)/55)
    dv_pos=softplus(d/10)
    eps=(max(0,v)**(4/3))*max(0,south_bz)/10000
    coupling=(v/450)*(1+bt/10)*south_prob
    # CME feature deliberately kept at 0 here; wind/Bz models already carry CME influence.
    return [1.0,v_excess,hss,dv_pos,south_prob,south_bz,bt/10,eps,coupling,0.0], gate


default_coef={
    "version":"SWIFT-Kp-AI-v0.6-bias-bz-gated",
    "feature_names":["bias","v_excess","hss_gate","dv_pos","south_prob","south_bz","bt10","epsilon","coupling","cme"],
    "beta":[0.04718,0.43586,-0.53846,0.32038,0.01652,0.09435,0.02359,1.25555,0.90703,-0.16569],
    "calibration":{"gain":1.2998,"offset":-0.5383},
    "ridge_lambda":0.35,
}
coef=load(COEF,default_coef) or default_coef
coef.setdefault("beta",default_coef["beta"]); coef.setdefault("calibration",default_coef["calibration"])


def recent_kp(hours=9):
    vals=[r["value"] for r in kp_obs if NOW-timedelta(hours=hours)<=r["time"]<=NOW]
    return median(vals,2.0)


prev_skill=load(LEAD,{}) or {}
skill_by_day={int(x.get("lead_day",0)):x for x in prev_skill.get("by_lead_day",[]) if x.get("lead_day")}


def lead_bias_correction(lead_h):
    day=max(1,min(5,int(max(0,lead_h)//24)+1))
    s=skill_by_day.get(day,{})
    n=int(s.get("count",0) or 0); bias=num(s.get("bias"),0)
    if n<30:return 0.0,day,n,bias
    alpha=clamp(0.20+0.45*min(1,n/500),0.20,0.65)
    return alpha*bias,day,n,bias


def raw_kp(t, forecast=True):
    x,gate=features(t,forecast)
    beta=coef["beta"]; y=sum(beta[i]*x[i] for i in range(min(len(beta),len(x))))
    cal=coef.get("calibration",{})
    y=num(cal.get("gain"),1)*y+num(cal.get("offset"),0)
    lead=max(0,(t-NOW).total_seconds()/3600)
    rk=recent_kp(9)
    w=clamp(1-lead/18,0,0.65)
    y=(1-w)*y+w*rk

    corr,day,n,bias=lead_bias_correction(lead)
    y-=corr

    storm=0.38*sigmoid((x[4]-0.55)/0.12)+0.27*sigmoid((x[5]-5)/2)+0.20*sigmoid((x[6]-1)/.35)+0.15*sigmoid((x[8]-1.8)/.7)
    if storm<.35:max_allowed=max(4.6,rk+1.0)
    elif storm<.55:max_allowed=max(5.4,rk+1.6)
    elif storm<.72:max_allowed=max(6.3,rk+2.3)
    else:max_allowed=9
    y=clamp(y,0,max_allowed)
    return y,x,{"bz_gate":round(gate,3),"lead_day":day,"previous_bias":round(bias,3),"bias_correction":round(corr,3),"bias_sample_count":n}


def round_third(k):
    vals=[0,0.33,0.67,1,1.33,1.67,2,2.33,2.67,3,3.33,3.67,4,4.33,4.67,5,5.33,5.67,6,6.33,6.67,7,7.33,7.67,8,8.33,8.67,9]
    return min(vals,key=lambda v:abs(v-k))


def gscale(k):
    return "G5" if k>=9 else "G4" if k>=8 else "G3" if k>=7 else "G2" if k>=6 else "G1" if k>=5 else "G0"


# Forecast
start=NOW.replace(minute=0,second=0,microsecond=0)
start=start.replace(hour=(start.hour//3)*3)
if start<NOW:start+=timedelta(hours=3)
forecast=[]
for i in range(FORECAST_DAYS*8):
    t=start+timedelta(hours=3*i)
    raw,x,diag=raw_kp(t,True)
    kp=round_third(raw)
    br=fc_bz(t)
    forecast.append({
        "time":iso(t),"start_time":iso(t),"end_time":iso(t+timedelta(hours=3)),
        "lead_hours":round((t-NOW).total_seconds()/3600,1),
        "kp":kp,"kp_raw":round(raw,3),"g_scale":gscale(kp),
        "speed":round(fc_speed(t),1),
        "bz_min_effective":round(-br["south_bz"],2),
        "southward_bz_probability_effective":round(br["south_prob"],1),
        "feature_vector":[round(v,4) for v in x],
        "diagnostics":diag,
    })


# Archive + scoring
archive=(load(ARCHIVE,{"items":[]}) or {}).get("items",[])[-12000:]
issued=iso(NOW)
for r in forecast:
    archive.append({"issued_at":issued,"target_time":r["time"],"lead_hours":r["lead_hours"],
                    "lead_day":max(1,min(5,int(r["lead_hours"]//24)+1)),
                    "predicted_kp":r["kp"],"scored":False})

def obs_kp(t):
    r=nearest(kp_obs,t,1.7); return r["value"] if r else None

scored=[]
for a in archive[-14000:]:
    if a.get("scored"): scored.append(a); continue
    tt=parse(a.get("target_time"))
    if not tt or tt>NOW-timedelta(hours=1): scored.append(a); continue
    o=obs_kp(tt)
    if o is None: scored.append(a); continue
    p=num(a.get("predicted_kp"),0); e=p-o
    q=dict(a); q.update({"scored":True,"observed_kp":round(o,2),"error":round(e,3),"abs_error":round(abs(e),3),
                         "hit_within_1kp":abs(e)<=1,"hit_within_067kp":abs(e)<=.67,"g_scale_hit":gscale(p)==gscale(o)})
    scored.append(q)


def skill(rr):
    if not rr:return {"count":0,"status":"learning"}
    e=[num(x.get("error"),0) for x in rr]
    return {"count":len(rr),"mae":round(mean([abs(x) for x in e]),3),"bias":round(mean(e),3),
            "rmse":round(math.sqrt(mean([x*x for x in e])),3),
            "hit_rate_within_1kp":round(100*sum(x.get("hit_within_1kp") for x in rr)/len(rr),1),
            "hit_rate_within_067kp":round(100*sum(x.get("hit_within_067kp") for x in rr)/len(rr),1),
            "g_scale_hit_rate":round(100*sum(x.get("g_scale_hit") for x in rr)/len(rr),1)}

recent=[x for x in scored if x.get("scored") and parse(x.get("target_time")) and parse(x["target_time"])>=NOW-timedelta(days=30)]
by_day=[{"lead_day":d,**skill([x for x in recent if int(x.get("lead_day",0))==d])} for d in range(1,6)]
lead_skill={"updated_at":iso(NOW),"overall":skill(recent),"by_lead_day":by_day}


# Direct hindcast (observed inputs), useful but not a substitute for lead-time skill.
def verify(days):
    rr=[]
    cut=NOW-timedelta(days=days)
    for r in kp_obs:
        if not(cut<=r["time"]<=NOW-timedelta(hours=1)):continue
        if obs_speed(r["time"]) is None:continue
        p,_,_=raw_kp(r["time"],False); pred=round_third(p); e=pred-r["value"]
        rr.append({"time":iso(r["time"]),"predicted_kp":pred,"observed_kp":round(r["value"],2),"error":round(e,3),
                   "abs_error":round(abs(e),3),"hit_within_1kp":abs(e)<=1,"hit_within_067kp":abs(e)<=.67,
                   "g_scale_hit":gscale(pred)==gscale(r["value"])})
    s=skill(rr); s["items"]=rr[-300:]; return s

verification={"updated_at":iso(NOW),"method":"direct_hindcast_against_NOAA_observed_inputs","last_7d":verify(7),"last_30d":verify(30)}

latest={
    "updated_at":iso(NOW),"model":"SWIFT-Kp-AI-v0.6-bias-bz-gated","forecast_days":FORECAST_DAYS,"step_hours":3,
    "forecast":forecast,"current":forecast[0] if forecast else None,
    "max_kp":max(forecast,key=lambda x:x["kp"]) if forecast else None,
    "verification":verification,"leadtime_skill":lead_skill,"coefficients":coef,
    "bz_model_gate":{"model":bz_model.get("model"),"readiness":ready,"kp_input_confidence":round(bz_model_conf,3)},
    "formula":"fitted nonlinear Vsw/Bz features + recent Kp assimilation - learned lead-time bias; Bz features confidence-gated"
}
save(LATEST,latest); save(VERIF,verification); save(LEAD,lead_skill); save(ARCHIVE,{"updated_at":iso(NOW),"items":scored[-14000:]})
save(COEF,coef)
hist=(load(HISTORY,{"items":[]}) or {}).get("items",[])[-720:]
hist.append({"time":iso(NOW),"leadtime_overall":lead_skill["overall"],"max_kp":latest["max_kp"],"bz_gate":latest["bz_model_gate"]})
save(HISTORY,{"updated_at":iso(NOW),"items":hist})
save(INDEX,{"updated_at":iso(NOW),"latest":"latest.json","verification":"verification.json","leadtime_skill":"leadtime-skill.json","history":"history.json"})
TXT.write_text("\n".join([":Product: SWIFT 3-Day Kp Forecast",f":Issued: {NOW.strftime('%Y %b %d %H%M UTC')}"]+
                         [f"{r['time']}  Kp={r['kp']}  {r['g_scale']}" for r in forecast])+"\n",encoding="utf-8")
print(json.dumps({"max_kp":latest["max_kp"],"leadtime_skill":lead_skill["overall"],"bz_gate":latest["bz_model_gate"]},ensure_ascii=False,indent=2))
