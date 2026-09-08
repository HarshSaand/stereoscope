import numpy as np


def bad_disparity(prediction, truth):
    error = np.abs(prediction-truth)
    return (error > 3) & (error > .05*np.abs(truth))


def evaluate(disparity, truth, risk=None):
    labelled = np.isfinite(truth) & (truth > 0)
    predicted = np.isfinite(disparity) & (disparity > 0)
    valid = labelled & predicted
    if not valid.any():
        raise ValueError("No valid labelled predictions")
    error = np.abs(disparity[valid]-truth[valid])
    bad = bad_disparity(disparity[valid], truth[valid])
    # Invalid predictions are failures in the all-labelled statistic.
    all_bad = bad_disparity(disparity[labelled], truth[labelled]) | ~predicted[labelled]
    result = dict(labelled_pixels=int(labelled.sum()), predicted_pixels=int(valid.sum()),
        prediction_coverage=float(valid.sum()/labelled.sum()), epe=float(error.mean()),
        d1_valid=float(bad.mean()), d1_all_labelled=float(all_bad.mean()),
        bad1_valid=float((error>1).mean()), bad2_valid=float((error>2).mean()),
        bad4_valid=float((error>4).mean()))
    if risk is not None:
        r = np.clip(risk[valid], 0, 1)
        if not np.isfinite(r).all():
            raise ValueError("Non-finite risk")
        order = np.argsort(r, kind="stable")
        result["brier"] = float(np.mean((r-bad)**2))
        curve=[]
        for c in (.25,.5,.7,.9,.95,1.):
            count=max(1,int(c*len(order)))
            cutoff=r[order[count-1]]
            below=r<cutoff; tied=r==cutoff
            fraction=(count-below.sum())/tied.sum()
            # Expected random tie-break: constant scores cannot exploit raster order.
            curve.append(dict(retained_valid_coverage=c,
                retained_labelled_coverage=float(count/labelled.sum()),
                d1=float((bad[below].sum()+fraction*bad[tied].sum())/count),
                epe=float((error[below].sum()+fraction*error[tied].sum())/count)))
        result['risk_coverage']=curve
    return result
