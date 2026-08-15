# Fixed-split traffic reconstruction status

This directory is the official `moghadas76/Freq_Spect` checkout at commit
`ecf0e354c9e8ec39d93392d0a8dadd9c8e2446d0`.  The commit's own tree contains
`models/Flow_Spect.py`, but its advertised entry point imports a missing
`exp/exp_main_F.py`.  No commit object available in this checkout contains that
module.  Consequently, the upstream training/evaluation pipeline cannot be
executed as released.

`run_unified_144_reconstruction.py` is therefore an **in-house reconstruction**,
not an official FreqFlow reproduction.  It preserves the released `Flow_Spect`
architecture and uses the benchmark's fixed METR-LA/PEMS-BAY 144-to-144 index.
It defines an explicit, documented surrogate objective:

1. the released spectral forecast is the base trajectory;
2. the released flow head learns the velocity from that base trajectory to the
   observed 288-step trajectory using a linear conditional flow path;
3. at inference, a one-step velocity correction at flow time zero refines the
   base forecast; and
4. the loss is forecast MAE plus `flow_weight` times flow-velocity MSE.

The resulting metrics must be reported separately as “FreqFlow in-house
reconstruction” and must not be represented as results from the official
training code.  If the authors release the missing `exp` module, replace this
runner and rerun the experiment using their official pipeline.
