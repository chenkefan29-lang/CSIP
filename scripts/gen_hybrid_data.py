import sys,time,os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from csip.ai.dataset import run_parallel
t0=time.time()
for name,days,seed in (("train",60,101),("val",15,202),("test",15,303)):
    d=run_parallel(days,seed); np.savez_compressed(f"data/hybrid_{name}.npz",**d)
    print(name,{k:v.shape for k,v in d.items()},"%.0fs"%(time.time()-t0),flush=True)
