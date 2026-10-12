#!/usr/bin/env python3
"""Remark 4 (statistical error of the estimator): variance of the per-run Shapley
estimator with common random numbers (CRN, as implemented in Algorithm 1) versus
the variance the same R runs would give with independent seeds per coalition.

Reads reproduce/data/exact_shapley_results.json (Experiment 1, 243 estimates).
Usage: python experiments/crn_variance.py [path/to/exact_shapley_results.json]
"""
import json, numpy as np, math, itertools, sys
from pathlib import Path
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
      Path(__file__).resolve().parents[1] / "reproduce" / "data" / "exact_shapley_results.json"
d=json.load(open(SRC))
ratios=[]; hw_rel=[]; rows=[]
for alg in d:
    for fn in d[alg]:
        for dim in d[alg][fn]:
            e=d[alg][fn][dim]; ops=e['operators']; n=len(ops); R=e['n_runs']
            # coalition values per run; keys are tuple-like strings
            co={}
            for k,v in e['coalitions'].items():
                idx=tuple(sorted(int(x) for x in k.strip("()").split(",") if x.strip()))
                co[tuple(sorted(ops[i] for i in idx))]=np.asarray(v,float)
            for i,op in enumerate(ops):
                others=[o for o in ops if o!=op]
                coef={}  # net coefficient c_T per coalition T
                for r in range(len(others)+1):
                    for S in itertools.combinations(others,r):
                        w=math.factorial(len(S))*math.factorial(n-len(S)-1)/math.factorial(n)
                        T1=tuple(sorted(S+(op,))); T0=tuple(sorted(S))
                        coef[T1]=coef.get(T1,0)+w; coef[T0]=coef.get(T0,0)-w
                phi_r=sum(c*co[T] for T,c in coef.items())        # per-run phi (CRN)
                var_crn=np.var(phi_r,ddof=1)/R
                var_ind=sum(c*c*np.var(co[T],ddof=1) for T,c in coef.items())/R
                if var_ind>0:
                    ratios.append(var_crn/var_ind)
                rows.append((alg,fn,dim,op,var_crn,var_ind))
                # CI half-width relative to |phi|
                m=phi_r.mean()
                if abs(m)>1e-9: hw_rel.append(1.96*math.sqrt(var_crn)/abs(m))
ratios=np.array(ratios)
print("estimates:",len(ratios),"| median Var_CRN/Var_indep = %.3f | IQR %.3f-%.3f | max %.3f | share with ratio<1: %.2f"%(np.median(ratios),*np.percentile(ratios,[25,75]),ratios.max(),(ratios<1).mean()))
print("equivalent extra runs factor (1/ratio) median: %.2f"%(1/np.median(ratios)))
hw=np.array(hw_rel); print("CI half-width / |phi| : median %.3f, 90th pct %.3f"%(np.median(hw),np.percentile(hw,90)))
