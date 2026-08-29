"""The metrics CSV writer, in both of the schemas already on disk."""

import json
from typing import Optional

import numpy as np

from arborist_experiments.morphology_cv.classifier import boundary_hits


def write_rows(writer, rep: str, mod: str, cv, chance: Optional[float] = None,
               csv_fh=None) -> None:
    """One row per metric.

    Two column layouts, and this cannot be unified away: the Patch-seq runs write 10
    columns with neither n_features nor a chance-corrected value, the MICrONS runs
    write 12 with both, and the summary tables in scratch/summary_tables read the
    finished CSVs by position. Passing `chance` selects the MICrONS layout, which is
    the only place chance correction applies -- 8 classes there against 13 on
    Patch-seq makes raw balanced accuracy incomparable between the two datasets.

    best_params and grid_boundary repeat down the metric rows so that any one row is
    self-describing; they used to reach disk only as an f-stringed dict in the run log.
    """
    params = json.dumps(cv.best_params, sort_keys=True, default=str)
    hits = ";".join(boundary_hits(cv.best_params))
    for metric_name, vals in cv.metrics.items():
        row = [rep, mod, metric_name, float(np.mean(vals)), float(np.std(vals)),
               cv.n_samples]
        if chance is None:
            row += [cv.n_classes, json.dumps(vals)]
        else:
            row += [cv.n_features, cv.n_classes, json.dumps(vals)]
            row.append((float(np.mean(vals)) - chance) / (1.0 - chance)
                       if metric_name == "balanced_accuracy" else "")
        writer.writerow(row + [params, hits])
    if csv_fh is not None:
        csv_fh.flush()
