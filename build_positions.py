"""東大谷口研・朝日新聞 2026年政治家調査 → 診断アプリ用の位置データを作る。
出力: positions_data.py (アプリが読み込む辞書)  /  politician_positions.csv (確認用)
"""
import json
import numpy as np, pandas as pd

SRC = "/mnt/user-data/uploads/2026UTASP20260220_1.csv"
df = pd.read_csv(SRC, encoding="cp932")
df = df[df.RESPONSE == 1].copy()
PARTY = {1:"自民党",2:"中道改革連合",3:"日本維新の会",4:"国民民主党",5:"れいわ新選組",6:"共産党",
         7:"参政党",8:"日本保守党",9:"社民党",10:"チームみらい",11:"諸派",12:"無所属",20:"減税日本・ゆうこく連合"}
df["party"] = df.PARTY.map(PARTY)

# (列, 向き) 向き=+1: 「賛成/Aに近い」ほど 左側(B/L/P)。R軸は調査に対応設問がないため作らない。
ITEMS = {
 "E": [("Q4_6",-1),("Q3_2",-1),("Q3_3",-1),("Q5_3",-1),("Q4_7",+1)],          # Q4_8(財政出動)は他と逆相関のため除外
 "G": [("Q4_13",-1),("Q4_14",-1),("Q4_15",-1),("Q4_1",-1),("Q4_3",-1),("Q5_13",+1)],
 "V": [("Q4_17",+1),("Q6",+1),("Q5_10",-1),("Q5_11",+1),("Q4_12",+1),("Q4_5",-1)],
}
S = pd.DataFrame({c: (3 - df[c].where(df[c].between(1,5))) * d for it in ITEMS.values() for c, d in it})
for ax, it in ITEMS.items():
    cols = [c for c,_ in it]
    n = S[cols].notna().sum(axis=1)
    df[ax] = np.where(n >= 3, S[cols].mean(axis=1) / 2, np.nan)
    df["n_"+ax] = n
LET = {"E":("B","S"),"G":("L","A"),"V":("P","T")}
ok = df[list("EGV")].notna().all(axis=1)
df["type3"] = None
df.loc[ok, "type3"] = df[ok].apply(lambda r: "".join(LET[a][0] if r[a] >= 0 else LET[a][1] for a in "EGV"), axis=1)

if __name__ == "__main__":
    # 整合性
    def alpha(X):
        X = X.dropna(); k = X.shape[1]
        return k/(k-1)*(1 - X.var(ddof=1).sum()/X.sum(axis=1).var(ddof=1))
    for ax, it in ITEMS.items():
        print(ax, "alpha=%.2f" % alpha(S[[c for c,_ in it]]))
    print(df[list("EGV")].corr().round(2))

    # 党ごと
    use = [p for p in PARTY.values() if p not in ("諸派","無所属")]
    parties = {}
    for p in use:
        d = df[df.party == p]
        rec = {"n": int(len(d))}
        for ax in "EGV":
            v = d[ax].dropna()
            rec[ax] = {"mean": round(float(v.mean()),3), "q25": round(float(v.quantile(.25)),3),
                       "q75": round(float(v.quantile(.75)),3), "n": int(len(v))}
        t = d.type3.dropna().value_counts()
        rec["types"] = {k:int(v) for k,v in t.items()}
        parties[p] = rec
    # 型×政党 (3文字型ごとの党内訳)
    cross = df[ok & df.party.isin(use)].groupby(["type3","party"]).size().unstack(fill_value=0)
    type_party = {t: {p:int(cross.loc[t,p]) for p in cross.columns if cross.loc[t,p] > 0} for t in cross.index}

    # 党首: Q2で好感度を尋ねられた人物(調査票の客観的基準)
    LEADERS = {"高市早苗":"自民党","野田佳彦":"中道改革連合","斉藤鉄夫":"中道改革連合","吉村洋文":"日本維新の会",
               "玉木雄一郎":"国民民主党","山本太郎":"れいわ新選組","田村智子":"共産党","神谷宗幣":"参政党"}
    leaders = {}
    for name, party in LEADERS.items():
        sur = name[:2]  # 氏の先頭2文字で検索(氏名表記は「氏＝名」)
        m = df[df.NAME.astype(str).str.replace("＝","").str.contains(name) | df.NAME.astype(str).str.contains(sur+"＝"+name[2:])]
        if len(m) == 1 and not np.isnan(m.iloc[0][["E","G","V"]].astype(float)).any():
            r = m.iloc[0]
            leaders[name] = {"party":party, **{ax: round(float(r[ax]),3) for ax in "EGV"}, "type3": r.type3}
        else:
            leaders[name] = {"party":party, "status": "調査に回答なし/該当行なし" if len(m)==0 else "回答不足 or 重複", "rows": int(len(m))}
    out = {"source":"2026年東京大学谷口研究室・朝日新聞社共同政治家調査(2026衆院選, 有効回答1197人)",
           "parties":parties, "type_party":type_party, "leaders":leaders}
    json.dump(out, open("positions.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
    rows=[]
    for p,r in parties.items():
        rows.append({"対象":p,"種別":"政党平均","n":r["n"], **{f"{a}_mean":r[a]["mean"] for a in "EGV"}, **{f"{a}_q25":r[a]["q25"] for a in "EGV"}, **{f"{a}_q75":r[a]["q75"] for a in "EGV"}})
    for k,v in leaders.items():
        rows.append({"対象":k+"("+v["party"]+")","種別":"党首", **{f"{a}_mean":v.get(a) for a in "EGV"}})
    pd.DataFrame(rows).to_csv("politician_positions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(leaders, ensure_ascii=False, indent=0)[:1500])
    for p,r in parties.items():
        top = sorted(r["types"].items(), key=lambda x:-x[1])[:3]
        print(p, r["n"], {a:r[a]["mean"] for a in "EGV"}, top)
