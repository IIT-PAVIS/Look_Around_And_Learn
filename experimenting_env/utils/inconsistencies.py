from functools import partial

import numba as nb
import numpy as np
import torch


# @nb.jit
def solve_inconsistency(
    vox_ids, object_id_to_logits, object_ids, voxel_ids, solve_function
):
    results = []
    for vox_id in vox_ids:
        objects = object_ids[voxel_ids == vox_id]
        if len(objects) == 0:
            return None, None
        ids = tuple(np.unique(objects, return_counts=True)[0])

        logits = []
        for i in ids:
            obj_logits = object_id_to_logits[i].cpu()
            if len(obj_logits.shape) == 1:
                obj_logits = obj_logits.unsqueeze(0)
            logits.append(obj_logits)
        logits = torch.cat(logits)
        resolved_class, _ = solve_function(logits)
        results.append((vox_id, (resolved_class, logits)))

    return results


def _seal_impl(logits):

    if len(logits.shape) == 1:
        resolved_class = logits.argmax()
        resolved_logits = logits
    else:

        values, indexes = logits.max(0)
        resolved_logits = logits[indexes[values.argmax()]]
        resolved_class = resolved_logits.argmax()

    return resolved_class, resolved_logits.unsqueeze(0)


def _ours_bayesian(logits):
    if len(logits.shape) == 1:
        resolved_class = logits.argmax()
        resolved_logits = logits
    else:
        resolved_logits = torch.logsumexp(logits, 0)
        resolved_logits = resolved_logits / resolved_logits.sum()
        resolved_class = resolved_logits.argmax()
        if torch.any(torch.isnan(resolved_logits)):
            breakpoint()

    return resolved_class, resolved_logits.unsqueeze(0)


def _ours_impl(logits):
    if len(logits.shape) == 1:
        resolved_class = logits.argmax()
        resolved_logits = logits
    else:

        resolved_class = logits.max(0).values.argmax(0)
        
        resolved_logits = logits.mean(0)

    return resolved_class, resolved_logits.unsqueeze(0)

def _ours_max(logits):
    if len(logits.shape) == 1:
        resolved_class = logits.argmax()
        resolved_logits = logits
    else:

        resolved_class = logits.max(0).values.argmax(0)
        resolved_id = logits.max(0).indices[logits.max(0).values.argmax()]
        resolved_logits = logits[resolved_id]

    return resolved_class, resolved_logits.unsqueeze(0)


# @nb.jit
def _ours_avg(logits):
    if len(logits.shape) == 1:
        resolved_class = logits.argmax()
        resolved_logits = logits
    else:

        resolved_logits = logits.mean(0)

        resolved_class = resolved_logits.argmax()

    return resolved_class, resolved_logits.unsqueeze(0)
