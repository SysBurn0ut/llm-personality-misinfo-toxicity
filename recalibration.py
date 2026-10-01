"""
Recalibration of confidence_true (post-hoc, see thesis Sec. 4.2).

belief and confidence_true are parsed separately, so a response can be
incoherent: belief=FALSE with confidence_true > 50, or belief=TRUE with
confidence_true < 50. In these cases the model reported the confidence in
its OWN answer instead of the probability that the headline is true
(it happens mostly with gpt-4o-mini). The value is mapped back to the
probability scale as 100 - c. The rule is the same for every model and
does not depend on the traits.
"""
import pandas as pd


def _bel(x):
    s = str(x).strip().upper()
    return s if s in ("TRUE", "FALSE") else None


def recalibrate_confidence(df, verbose=True):
    d = df.copy()
    b = d["belief"].map(_bel)
    c = d["confidence_true"]
    inc = ((b == "FALSE") & (c > 50)) | ((b == "TRUE") & (c < 50))
    d.loc[inc, "confidence_true"] = 100 - c[inc]
    d["recalibrated"] = inc
    if verbose:
        share = inc.groupby(d["model"]).mean().mul(100).round(1)
        print("\n--- confidence_true recalibration: % of responses recoded per model ---")
        print(share.to_string())
    return d
