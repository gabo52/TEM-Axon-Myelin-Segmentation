"""Post-processing used by code/predict.py (semantic map -> filtered instances).

    sem -> watershed (no ring filter) -> close_rings -> drop_unwrapped(0.5) -> accept_filter.drop

Difference with the main pipeline (to_instances(sem) -> drop): the ring filter
`drop_unwrapped` runs AFTER the open rings are closed, so fibres whose myelin was
only partly predicted are repaired instead of discarded. With close=False this is
exactly the main pipeline. See otros/postproc_ablation.md.
"""
try:
    from otros.accept_filter import drop
    from otros.instances import MIN_RING, drop_unwrapped, to_instances
    from otros.ring_closure import close_rings
except ImportError:
    from accept_filter import drop
    from instances import MIN_RING, drop_unwrapped, to_instances
    from ring_closure import close_rings

def postprocess(sem, score, tau=0.2, close=True, border=True):
    if not close:
        inst = to_instances(sem)
    else:
        inst = drop_unwrapped(close_rings(to_instances(sem, 0)), MIN_RING)
    return inst if score is None else drop(inst, score, tau, border)
