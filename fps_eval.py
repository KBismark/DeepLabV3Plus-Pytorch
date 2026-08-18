import time
import torch


def _forward_logits(model, x):
    """Handles both stock DeepLabV3 (returns tensor) and GeoDeepLabV3Plus
    (returns dict) transparently."""
    out = model(x)
    if isinstance(out, dict):
        return out["logits"]
    return out


@torch.no_grad()
def benchmark_pure_fps(model, input_size, device, warmup_iters=20, timed_iters=100):
    """
    Pure model forward-pass speed. input_size: (N, C, H, W).
    Always runs model.eval() internally and restores prior mode after --
    does not mutate the model's training state for the caller.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(*input_size, device=device)

    for _ in range(warmup_iters):
        _ = _forward_logits(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(timed_iters):
        _ = _forward_logits(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if was_training:
        model.train()

    latency_ms = (elapsed / timed_iters) * 1000
    fps = timed_iters / elapsed
    return {"pure_latency_ms": latency_ms, "pure_fps": fps, "batch_size": input_size[0]}


@torch.no_grad()
def benchmark_end_to_end_fps(model, loader, device, max_batches=50):
    """
    Times the REAL evaluation pipeline: dataloader fetch -> to(device) ->
    model forward -> argmax postprocessing. Uses the actual val_loader
    passed in (same one validate() uses), so this reflects real
    preprocessing cost (resize/crop/normalize already applied by the
    dataset's transform), not a synthetic random tensor.

    max_batches: cap so this doesn't re-run the full val set every time --
    50 batches is enough to average out warmup/dataloader jitter without
    meaningfully slowing down your validation cycle.
    """
    was_training = model.training
    model.eval()

    n_images = 0
    start = time.perf_counter()
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, dtype=torch.float32)
        logits = _forward_logits(model, images)
        _ = logits.detach().max(dim=1)[1]  # same postprocessing validate() does
        n_images += images.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if was_training:
        model.train()

    if n_images == 0:
        return {"e2e_latency_ms": None, "e2e_fps": None, "n_images": 0}

    e2e_fps = n_images / elapsed
    e2e_latency_ms = (elapsed / n_images) * 1000
    return {"e2e_latency_ms": e2e_latency_ms, "e2e_fps": e2e_fps, "n_images": n_images}


def count_flops(model, input_size):
    """
    FLOPs/MACs via ptflops (pip install ptflops). Returns a dict with a
    clear 'unavailable' reason instead of crashing if the package or the
    model isn't compatible (e.g. models returning a dict output need a
    small wrapper -- handled below).
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return {"flops": None, "params": None,
                "note": "ptflops not installed -- run: pip install ptflops"}

    class _LogitsOnlyWrapper(torch.nn.Module):
        """ptflops needs a single-tensor-output module; unwrap dict outputs."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return _forward_logits(self.m, x)

    was_training = model.training
    model.eval()
    wrapped = _LogitsOnlyWrapper(model)

    try:
        macs, params = get_model_complexity_info(
            wrapped, input_size[1:], as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        flops = macs * 2  # ptflops reports MACs; FLOPs is conventionally ~2x MACs
        result = {"flops": flops, "macs": macs, "params": params,
                  "flops_g": flops / 1e9, "params_m": params / 1e6}
    except Exception as e:
        result = {"flops": None, "params": None, "note": f"ptflops failed: {e}"}

    if was_training:
        model.train()
    return result


def format_extra_metrics(pure_result, e2e_result, flops_result):
    """One consistent print block appended after metrics.to_str(val_score)."""
    lines = ["--- Speed / Compute Metrics ---"]
    lines.append(f"  Pure model:   {pure_result['pure_fps']:.1f} FPS  "
                 f"({pure_result['pure_latency_ms']:.2f} ms/batch, batch={pure_result['batch_size']})")
    if e2e_result["e2e_fps"] is not None:
        lines.append(f"  End-to-end:   {e2e_result['e2e_fps']:.1f} FPS  "
                     f"({e2e_result['e2e_latency_ms']:.2f} ms/image, over {e2e_result['n_images']} images)")
    else:
        lines.append("  End-to-end:   n/a (no batches processed)")
    if flops_result.get("flops") is not None:
        lines.append(f"  FLOPs:        {flops_result['flops_g']:.2f} GFLOPs")
        lines.append(f"  Params:       {flops_result['params_m']:.2f} M")
    else:
        lines.append(f"  FLOPs:        unavailable ({flops_result.get('note', 'unknown reason')})")
    return "\n".join(lines)