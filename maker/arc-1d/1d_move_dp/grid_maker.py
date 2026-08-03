"""1D-ARC category: 1d_move_dp — auto-generated maker (bg=0)."""
from __future__ import annotations
import sys, random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))
from maker.sel_helpers import sel_of

MOVE_L, MOVE_R = 23, 22   # ARCLE MoveL / MoveR
FLIP_H = 26
SUBMIT = 34

def _blocks(row):
    """contiguous non-0 runs -> list of (start, end_inclusive, color)."""
    out=[]; i=0; n=len(row)
    while i<n:
        if row[i]!=0:
            j=i
            while j+1<n and row[j+1]!=0 and row[j+1]==row[i]: j+=1
            out.append((i,j,int(row[i]))); i=j+1
        else: i+=1
    return out

def _move_ops(cells, op, steps):
    """select-once + empty-sel glide (correct ARCLE Move idiom)."""
    ops=[op]; sels=[sel_of(cells)]
    for _ in range(steps-1):
        ops.append(op); sels.append(sel_of([]))
    return ops, sels

def _full(O):
    ho,wo=O.shape; return [0,0,ho-1,wo-1]


def sample_colors():
    c=random.choice(range(1,10)); d=random.choice([k for k in range(1,10) if k!=c]); return {"c":c,"d":d}
def generate(max_w,colors):
    c,d=colors["c"],colors["d"]; w=random.randint(14,max_w); L=random.randint(4,min(w-8,14))
    s=random.randint(0,3); gap=random.randint(2,6); dot=s+L+gap
    if dot>=w: raise ValueError
    gi=np.zeros((1,w),int); gi[0,s:s+L]=c; gi[0,dot]=d
    go=np.zeros((1,w),int); go[0,dot-L:dot]=c; go[0,dot]=d
    return {"input":gi,"output":go}
def derive_operations(I,O):
    I=np.asarray(I); O=np.asarray(O); b=_blocks(I[0]); blk=max(b,key=lambda t:t[1]-t[0])
    a,e,c=blk; dot=[t for t in b if t!=blk][0][0]; N=dot-1-e
    cells=[(0,x) for x in range(a,e+1)]
    ops,sels=_move_ops(cells,MOVE_R,N); ops.append(SUBMIT); sels.append(_full(O)); return ops,sels


from maker.base_grid_maker import BaseGridMaker
class GridMaker(BaseGridMaker):
    def parse(self, **kwargs):
        num_samples=kwargs.get("num_samples",1); num_examples=kwargs.get("num_examples",3)
        max_h,max_w=kwargs.get("max_grid_dim",[30,30]); dataset=[]
        for _sn in range(num_samples):
            pr_in,pr_out,ex_in,ex_out=[],[],[],[]; ops,sels=[],[]
            colors=sample_colors()
            j=0
            while j<num_examples+1:
                ok=False
                for _ in range(40):
                    try:
                        r=generate(max_w,colors)
                        I=np.array(r["input"],dtype=np.uint8); O=np.array(r["output"],dtype=np.uint8)
                        if I.shape[1]>max_w or O.shape[1]>max_w: continue
                        ok=True; break
                    except (IndexError,ValueError,KeyError): continue
                if not ok: j+=1; continue
                if j==num_examples:
                    pr_in.append(I); pr_out.append(O); ops,sels=derive_operations(I,O)
                else:
                    ex_in.append(I); ex_out.append(O)
                j+=1
            dataset.append((ex_in,ex_out,pr_in,pr_out,
                {"id":f"1d_move_dp-maker_{_sn+1}","concept":"1D-ARC 1d_move_dp","operations":ops,"selections":sels}))
        return dataset
