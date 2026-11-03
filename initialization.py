# -*- coding: utf-8 -*-
"""DAMI-specific initialization and thin Hugging Face training integration.

FLA supplies every model, recurrence and CUDA kernel. Transformers supplies the
training loop. This module contains DAMI's targets, calibration, architecture
adapters, memory-only export, and the unified entry point for 15 conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import textwrap

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ARCHITECTURES = ("gdn", "kda", "gdn2", "mamba2", "rwkv7")
INITIALIZATIONS = ('random', 'post', 'dami')
FAMILIES = {
    "gdn": ("gated_deltanet", "GatedDeltaNet"),
    "kda": ("kda", "KDA"),
    "gdn2": ("gdn2", "GDN2"),
    "mamba2": ("mamba2", "Mamba2"),
    "rwkv7": ("rwkv7", "RWKV7"),
}


def targets(layers: int, heads: int, tau_min=1.0, tau_max=4096.0):
    """Head 0 has the longest e-folding time. GDN uses a sine depth profile."""
    if layers < 1 or heads < 1 or not 0 < tau_min <= tau_max:
        raise ValueError("Positive dimensions and 0 < tau_min <= tau_max required")
    tau = torch.logspace(math.log10(tau_max), math.log10(tau_min), heads,
                         dtype=torch.float64)
    depth = torch.linspace(0, 1, layers, dtype=torch.float64)
    sensitivity = .25 + .5 * torch.sin(math.pi * depth)
    return tau, sensitivity


def inverse_softplus(x):
    if not torch.isfinite(x).all() or (x <= 0).any():
        raise ValueError("inverse_softplus requires finite positive values")
    return x + torch.log(-torch.expm1(-x))


def solve_bias(z, target, iterations=64):
    """Solve mean_tokens sigmoid(z+b) = target, independently per column."""
    z = z.double()
    target = torch.as_tensor(target, device=z.device, dtype=z.dtype)
    if not torch.isfinite(z).all() or not ((target > 0) & (target < 1)).all():
        raise ValueError("Finite activations and sensitivity in (0, 1) required")
    # Data-dependent brackets work even when the logits lie outside [-60, 60].
    logit = torch.logit(target)
    lo = logit - z.amax(0) - 32
    hi = logit - z.amin(0) + 32
    for _ in range(iterations):
        mid = (lo + hi) / 2
        below = torch.sigmoid(z + mid).mean(0) < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return (lo + hi) / 2


def centered_channel_noise(layer, head, channels, std=.03):
    """The deterministic channel perturbation used for paper KDA/GDN2."""
    if channels == 1:
        return torch.zeros(1)
    gen = torch.Generator(device="cpu").manual_seed(42000 + layer * 997 + head * 37)
    noise = torch.randn(channels, generator=gen)
    noise -= noise.mean()
    noise *= std / noise.std(unbiased=False)
    return noise - noise.mean()


def _layers(model):
    backbone = getattr(model, 'model', None)
    if backbone is None:
        backbone = getattr(model, 'backbone', None)
    if backbone is None or not hasattr(backbone, 'layers'):
        raise ValueError('Unsupported FLA model backbone')
    return list(backbone.layers)


def _normalize_save_metadata(model):
    # Transformers 5 uses a mapping for tied-weight metadata. These paper
    # configurations have untied embeddings; old FLA list metadata is empty
    # semantically and can be normalized without changing any tensor.
    for module in model.modules():
        if isinstance(getattr(module, '_tied_weights_keys', None), list):
            if getattr(model.config, 'tie_word_embeddings', False):
                raise ValueError('This integration supports the paper untied-embedding configurations')
            module._tied_weights_keys = {}
    # The upstream L2Wrap returns a view; Trainer scales loss in place. Cloning
    # this scalar preserves its backward while satisfying the autograd contract.
    if getattr(model.config, 'use_l2warp', False):
        _fix_l2warp_dtype()
        model.register_forward_hook(_clone_wrapped_loss)
    return model


def _clone_wrapped_loss(_module, _inputs, output):
    if getattr(output, 'loss', None) is not None:
        output.loss = output.loss.clone()
    return output


def _fix_l2warp_dtype():
    """Keep the native L2Wrap gradient, fixing its AMP scatter dtype contract."""
    module = importlib.import_module('fla.modules.l2warp')
    cls = module.L2Wrap
    if getattr(cls, '_dami_dtype_fix', False):
        return
    source = textwrap.dedent(inspect.getsource(cls.backward))
    anchor = 'glogits.scatter_(-1, ids, maxx * grad_output)'
    if source.count(anchor) != 1:
        raise RuntimeError('Unsupported FLA L2Wrap API; install the pinned dependency revision')
    source = source.replace('@staticmethod\n', '').replace(
        anchor, 'glogits.scatter_(-1, ids, (maxx * grad_output).to(glogits.dtype))')
    namespace = dict(vars(module))
    exec(compile('from __future__ import annotations\n'+source,
                 '<FLA L2Wrap AMP compatibility>', 'exec'), namespace)
    cls.backward = staticmethod(namespace['backward'])
    cls._dami_dtype_fix = True


def _is_memory(key):
    return key.endswith((".A_log", ".dt_bias", ".decay_log_scale",
                         ".w_lora.lora.2.bias"))


def memory_parameters(model):
    return {k: v.detach().cpu().contiguous().clone()
            for k, v in model.state_dict().items() if _is_memory(k)}


def paper_parameters(architecture, scheme='dami'):
    """Load any of the fifteen 100M memory initializations without FLA."""
    if architecture not in ARCHITECTURES or scheme not in INITIALIZATIONS:
        raise ValueError('Unknown architecture or initialization')
    name = '100m.json' if scheme == 'dami' else 'controls_100m.json'
    source = json.loads((ROOT / 'reference' / name).read_text(encoding='utf-8'))
    key = architecture if scheme == 'dami' else architecture + '_' + scheme
    return {k: torch.tensor(v["values"], dtype=getattr(torch, v["dtype"].split('.')[-1]))
            .reshape(v["shape"])
            for k, v in source['models'][key]['tensors'].items()}


def load_memory(model, state):
    """Load memory tensors strictly, while leaving non-memory weights untouched."""
    destination = {k: v for k, v in model.state_dict().items() if _is_memory(k)}
    if destination.keys() != state.keys():
        raise ValueError(f"Memory keys differ: missing={destination.keys()-state.keys()}, "
                         f"unexpected={state.keys()-destination.keys()}")
    for key, value in state.items():
        if destination[key].shape != value.shape:
            raise ValueError(f"Shape mismatch for {key}: {value.shape} vs {destination[key].shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite parameter: {key}")
    with torch.no_grad():
        for key, value in state.items():
            destination[key].copy_(value.to(destination[key]))


def _rwkv_logits(self, x):
    # Preserve the paper implementation's fp32 master bias with bf16 projections.
    seq = self.w_lora.lora
    hidden = seq[1](seq[0](x))
    out = F.linear(hidden, seq[2].weight, None)
    return (out.float() + seq[2].bias.float()).to(out.dtype)


def install_rwkv_adapter():
    """Replace RWKV's native constant with a frozen per-head scale.

    The dependency's forward implementation is retained. Only its decay-line
    expression changes; no recurrence or CUDA kernel is copied into this repo.
    """
    module = importlib.import_module("fla.layers.rwkv7")
    cls = module.RWKV7Attention
    if getattr(cls, "_dami_adapter", False):
        return
    source = textwrap.dedent(inspect.getsource(cls.forward))
    anchor = "w = -0.6065306597126334 * self.w_lora(xw).sigmoid()"
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported FLA RWKV-7 API. Install the pinned FLA revision in requirements.txt.")
    source = source.replace(anchor,
        "w = -self.decay_log_scale.float().exp().repeat_interleave(self.head_dim) "
        "* self._dami_logits(xw).sigmoid()")
    # Match the paper training backend's 32-token chunk schedule.
    if source.count('chunk_size=64,') != 1:
        raise RuntimeError('Unsupported RWKV-7 chunk scheduler in the FLA dependency')
    source = source.replace('chunk_size=64,', 'chunk_size=32,')
    namespace = dict(vars(module))
    exec(compile("from __future__ import annotations\n" + source,
                 "<DAMI RWKV-7 decay adapter>", "exec"), namespace)

    class DAMIRWKV7Attention(cls):
        _dami_adapter = True
        forward = namespace["forward"]
        _dami_logits = _rwkv_logits

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.decay_log_scale = nn.Parameter(
                torch.full((self.num_heads,), -.5, dtype=torch.float32), requires_grad=False)
            self.w_lora.lora[2].bias.data = self.w_lora.lora[2].bias.data.float()

        def _apply(self, fn, recurse=True):
            super()._apply(fn, recurse=recurse)
            self.decay_log_scale.data = self.decay_log_scale.data.float()
            self.w_lora.lora[2].bias.data = self.w_lora.lora[2].bias.data.float()
            return self

    module.RWKV7Attention = DAMIRWKV7Attention
    importlib.import_module("fla.models.rwkv7.modeling_rwkv7").RWKV7Attention = DAMIRWKV7Attention


def build_model(architecture, config=None, init_seed=42):
    """Build a real FLA model. A size label alone does not define its geometry."""
    if architecture not in FAMILIES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    import fla  # noqa: F401 -- registers the FLA model families
    if architecture == "rwkv7":
        install_rwkv_adapter()
    cfg = config or json.loads((ROOT / f"configs/{architecture}_100m.json").read_text(encoding="utf-8"))
    cfg = dict(cfg)
    torch.manual_seed(init_seed)
    if architecture == 'gdn2':
        # The pinned public FLA commit has the GDN2 layer/kernels, but no HF
        # language-model wrapper. Its norm/MLP/residual stack matches KDA's.
        # Reuse that stack, constructing genuine GDN2 attention in each block.
        from fla.models.kda import KDAConfig, KDAForCausalLM
        from fla.layers.gdn2 import GatedDeltaNet2
        class GDN2AttentionAdapter(GatedDeltaNet2):
            # KDA's HF initialization checks this flag; GDN2 uses its unbounded
            # Softplus gate. This flag does not change the GDN2 forward operator.
            safe_gate = False
        modeling = importlib.import_module('fla.models.kda.modeling_kda')
        cfg.pop('model_type', None)
        cfg['dami_architecture'] = 'gdn2'
        original = modeling.KimiDeltaAttention
        modeling.KimiDeltaAttention = GDN2AttentionAdapter
        try:
            model = KDAForCausalLM(KDAConfig(**cfg))
        finally:
            modeling.KimiDeltaAttention = original
        if not all(isinstance(layer.attn, GatedDeltaNet2) for layer in _layers(model)):
            raise RuntimeError('GDN2 adapter did not replace every attention block')
        return _normalize_save_metadata(model)
    submodule, prefix = FAMILIES[architecture]
    module = importlib.import_module(f"fla.models.{submodule}")
    config_cls = getattr(module, prefix + "Config")
    model_cls = getattr(module, prefix + "ForCausalLM")
    return _normalize_save_metadata(model_cls(config_cls(**cfg)))


@torch.no_grad()
def calibrate_gdn(model, batches, tau=None, sensitivity=None):
    """LSUV-style sequential calibration on actual a_proj outputs.

    batches: iterable of packed [batch, length] token tensors (no padding), or
    dictionaries with input_ids and an all-one attention_mask. Outputs from the
    calibrated layer are recomputed before propagating to the next layer.
    """
    if batches is None:
        raise ValueError('GDN calibration requires real packed token batches')
    layers = _layers(model)
    h = layers[0].attn.A_log.numel()
    default_tau, default_s = targets(len(layers), h)
    tau = torch.as_tensor(default_tau if tau is None else tau, dtype=torch.float64)
    sensitivity = torch.as_tensor(default_s if sensitivity is None else sensitivity)
    if tau.shape != (h,) or sensitivity.shape != (len(layers),) or (tau <= 0).any():
        raise ValueError("tau must have one positive value per head; sensitivity one per layer")
    if not ((sensitivity > 0) & (sensitivity < 1)).all():
        raise ValueError("Sensitivity targets must lie in (0, 1)")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    hidden_batches = []
    try:
        for batch in batches:
            if isinstance(batch, dict):
                mask = batch.get("attention_mask")
                if mask is not None and not torch.as_tensor(mask).bool().all():
                    raise ValueError("Use packed, unpadded calibration sequences")
                batch = batch["input_ids"]
            ids = torch.as_tensor(batch, dtype=torch.long, device=device)
            if ids.ndim != 2:
                raise ValueError("Calibration input_ids must have shape [batch, length]")
            hidden_batches.append(model.get_input_embeddings()(ids).cpu())
        if not hidden_batches:
            raise ValueError("Calibration data is empty")
        report = []
        for index, layer in enumerate(layers):
            attn = layer.attn
            if attn.dt_bias.numel() != h:
                raise ValueError("calibrate_gdn requires scalar, head-wise GDN dt_bias")
            captured = []
            handle = attn.a_proj.register_forward_hook(
                lambda _m, _i, out: captured.append(out.detach().float().cpu().reshape(-1, h)))
            try:
                for hidden in hidden_batches:
                    layer(hidden.to(device), use_cache=False)
            finally:
                handle.remove()
            z = torch.cat(captured)
            bias = solve_bias(z, sensitivity[index])
            mean_dt = F.softplus(z.double() + bias).mean(0)
            a_log = -tau.log() - mean_dt.log()
            attn.dt_bias.copy_(bias.to(attn.dt_bias))
            attn.A_log.copy_(a_log.to(attn.A_log))
            hidden_batches = [layer(hidden.to(device), use_cache=False)[0].cpu()
                              for hidden in hidden_batches]
            # Report realized values AFTER casting to the model's storage dtype.
            b = attn.dt_bias.detach().double().cpu()
            a = attn.A_log.detach().double().cpu()
            measured_s = torch.sigmoid(z.double() + b).mean(0)
            measured_tau = 1 / (a.exp() * F.softplus(z.double() + b).mean(0))
            report.append({"layer":index, "target_sensitivity":float(sensitivity[index]),
                           "realized_sensitivity":measured_s.tolist(), "target_tau":tau.tolist(),
                           "realized_tau":measured_tau.tolist()})
        return report
    finally:
        model.train(was_training)


@torch.no_grad()
def transfer_gdn_to_channels(model, gdn_state, channel_std=.03):
    """Paper KDA/GDN2: transfer head means, add centered channel perturbations."""
    for li, layer in enumerate(_layers(model)):
        a = gdn_state[f"model.layers.{li}.attn.A_log"].float().flatten()
        b = gdn_state[f"model.layers.{li}.attn.dt_bias"].float().flatten()
        attn = layer.attn
        h = attn.A_log.numel()
        if a.numel() != h or b.numel() != h:
            raise ValueError("GDN reference must match the target layer/head geometry")
        d = attn.dt_bias.numel() // h
        values = []
        for head in range(h):
            channel_values = b[head] + centered_channel_noise(li, head, d, channel_std)
            values.append(channel_values - channel_values.mean() + b[head])
        biases = torch.stack(values)
        attn.A_log.copy_(a.reshape_as(attn.A_log).to(attn.A_log))
        attn.dt_bias.copy_(biases.reshape_as(attn.dt_bias).to(attn.dt_bias))


@torch.no_grad()
def design_rwkv7(model, tau_min=1., tau_max=4096.):
    """Static channel-mean sigmoid response, NOT sigmoid derivative sensitivity."""
    layers = _layers(model)
    h = layers[0].attn.num_heads
    tau, _ = targets(len(layers), h, tau_min, tau_max)
    depth = torch.linspace(0, 1, len(layers), dtype=torch.float64)
    parabola = 4 * depth * (1 - depth)
    s = .25 + .5 * parabola / parabola.max().clamp_min(1e-12)
    for li, layer in enumerate(layers):
        attn = layer.attn
        first_projection = attn.w_lora.lora[0].weight
        if first_projection.detach().abs().max().item() != 0:
            raise ValueError("Static RWKV calibration requires the native zero input projection at step zero")
        d = attn.head_dim
        x = torch.linspace(-1, 1, d, dtype=torch.float64)
        fan = 2.5 * x * x.abs()
        macro = solve_bias(fan[:, None], s[li])[0]
        bias = (fan + macro).repeat(h)
        attn.w_lora.lora[2].bias.copy_(bias.to(attn.w_lora.lora[2].bias))
        a_log = -tau.log() - s[li].log()
        attn.decay_log_scale.copy_(a_log.to(attn.decay_log_scale))
        attn.decay_log_scale.requires_grad_(False)


@torch.no_grad()
def design_mamba2(model, gdn_state, tau_max=4096.):
    """Transfer the GDN static Softplus profile; preserve the global bias scale."""
    layers = _layers(model)
    keys = sorted(k for k in gdn_state if k.endswith('.dt_bias'))
    # Numerical layer order is intentional for newly designed models.
    keys.sort(key=lambda k: int(k.split('layers.')[1].split('.')[0]))
    source = torch.stack([F.softplus(gdn_state[k].double()).mean() for k in keys])
    multipliers = F.interpolate(source[None,None], size=len(layers), mode='linear',
                                align_corners=True).flatten()
    multipliers /= multipliers.log().mean().exp()
    native = torch.stack([F.softplus(layer.mixer.dt_bias.detach().double().cpu()).mean()
                          for layer in layers])
    multipliers /= (native * multipliers).mean() / native.mean()
    for li, layer in enumerate(layers):
        mixer = layer.mixer
        h = mixer.A_log.numel()
        a = torch.linspace(0, math.log(tau_max), h, dtype=torch.float64)
        dt = F.softplus(mixer.dt_bias.detach().double()) * multipliers[li].to(mixer.dt_bias.device)
        mixer.A_log.copy_(a.reshape_as(mixer.A_log).to(mixer.A_log))
        mixer.dt_bias.copy_(inverse_softplus(dt).to(mixer.dt_bias))


def initialize(model, architecture, *, scheme='dami', mode="paper", calibration_batches=None,
               gdn_parameters=None, tau_min=1., tau_max=4096., init_seed=42):
    """Initialize a FRESH model with Random, PoST, or DAMI; return memory tensors.

    paper: use the included exact-shape 100M parameters for all 15 conditions.
    calibrate: DAMI GDN with real packed calibration inputs.
    design: fresh-size Random/PoST, or DAMI architecture adaptations.
    """
    if architecture not in ARCHITECTURES or scheme not in INITIALIZATIONS:
        raise ValueError('Unknown architecture or initialization')
    if scheme != 'dami' and mode == 'calibrate':
        raise ValueError('Activation calibration belongs to DAMI, not Random/PoST')
    if mode == "paper":
        expected = json.loads((ROOT/f"configs/{architecture}_100m.json").read_text(encoding='utf-8'))
        for name in ['hidden_size','num_hidden_layers','head_dim','num_heads','vocab_size']:
            if name in expected and getattr(model.config,name,None) != expected[name]:
                raise ValueError(f"Paper parameters require {name}={expected[name]}; use a new design for other sizes")
        load_memory(model, paper_parameters(architecture, scheme))
    elif mode == 'design' and scheme in ('random', 'post'):
        design_control(model, architecture, scheme, gdn_parameters, tau_min, tau_max, init_seed)
    elif architecture == "gdn" and mode == "calibrate":
        h = _layers(model)[0].attn.A_log.numel()
        tau,s = targets(len(_layers(model)),h,tau_min,tau_max)
        model.dami_calibration_report = calibrate_gdn(model,calibration_batches,tau,s)
    elif mode == "design":
        gdn_parameters = paper_parameters('gdn') if gdn_parameters is None else gdn_parameters
        if architecture in ('kda','gdn2'):
            transfer_gdn_to_channels(model,gdn_parameters)
        elif architecture == 'mamba2':
            design_mamba2(model,gdn_parameters,tau_max)
        elif architecture == 'rwkv7':
            design_rwkv7(model,tau_min,tau_max)
        else:
            raise ValueError("GDN requires real calibration inputs; use mode='calibrate'")
    else:
        raise ValueError("Unsupported mode/architecture combination")
    return memory_parameters(model)


@torch.no_grad()
def design_control(model, architecture, scheme, gdn_parameters=None,
                   tau_min=1., tau_max=4096., init_seed=42):
    """Paper-style controls on a fresh model, keeping the common non-memory base."""
    if architecture in ('kda', 'gdn2'):
        if gdn_parameters is None:
            # Use a same-geometry GDN Random reference for the channel baselines.
            config = model.config.to_dict()
            config.pop('model_type', None)
            reference = build_model('gdn', config, init_seed)
            gdn_parameters = memory_parameters(reference)
            del reference
        transfer_gdn_to_channels(model, gdn_parameters)
    if architecture == 'rwkv7':
        for layer in _layers(model):
            layer.attn.decay_log_scale.fill_(-.5)
            layer.attn.decay_log_scale.requires_grad_(False)
    if scheme == 'random':
        return
    for layer in _layers(model):
        if architecture == 'rwkv7':
            attn = layer.attn
            bias = attn.w_lora.lora[2].bias.detach().double()
            sigma = bias.sigmoid().reshape(attn.num_heads, attn.head_dim).mean(-1)
            tau, _ = targets(len(_layers(model)), attn.num_heads, tau_min, tau_max)
            value = -tau.to(sigma.device).log() - sigma.log()
            attn.decay_log_scale.copy_(value.to(attn.decay_log_scale))
        else:
            mixer = layer.mixer if architecture == 'mamba2' else layer.attn
            ladder = torch.linspace(0., math.log(tau_max), mixer.A_log.numel(),
                                    device=mixer.A_log.device, dtype=mixer.A_log.dtype)
            mixer.A_log.copy_(ladder.reshape_as(mixer.A_log))


def export_parameters(parameters, output_dir, architecture, metadata=None, *, scheme='dami'):
    from safetensors.torch import save_file
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if scheme not in INITIALIZATIONS:
        raise ValueError('Unknown initialization')
    file = output/'memory_parameters.safetensors'
    state = {k:v.detach().cpu().contiguous() for k,v in parameters.items()}
    if any(not _is_memory(k) for k in state):
        raise ValueError('Export accepts memory tensors only, not a full model state dict')
    if not state or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Empty or non-finite memory parameter export")
    label = {'random':'Random', 'post':'PoST', 'dami':'DAMI'}[scheme]
    save_file(state,str(file),metadata={"method":label,"architecture":architecture,'initialization':scheme})
    manifest = {"method":label, 'initialization':scheme, "architecture":architecture, "format_version":2,
                "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                "tensors":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in state.items()},
                "metadata":metadata or {}}
    (output/'initialization.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf-8')
    return file


def _token_batches(path, batch_size):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            ids = json.loads(line)['input_ids']
            rows.append(torch.tensor(ids,dtype=torch.long))
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Calibration JSONL must contain equal-length, nonempty input_ids")
    return [torch.stack(rows[i:i+batch_size]) for i in range(0,len(rows),batch_size)]


def _packed_examples(files, tokenizer_path, seq_len):
    """Minimal input adapter; datasets and Transformers own IO/training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = load_dataset('parquet',data_files=list(files),split='train',streaming=True)
    buffer = []
    for row in dataset:
        ids = tokenizer(row['text'],add_special_tokens=False)['input_ids']
        # Match the calibration packer: concatenate documents without separators.
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            block,buffer = buffer[:seq_len],buffer[seq_len:]
            yield {'input_ids':block,'labels':block.copy()}


def _training_defaults(args):
    if args.batch_size is None:
        args.batch_size = 8 if args.architecture in ('mamba2','rwkv7') else 16
    if args.warmup_steps is None:
        args.warmup_steps = 2000 if args.architecture in ('mamba2','rwkv7') else 1000
    if args.seed is None:
        args.seed = 43 if (args.architecture,args.initialization)==('rwkv7','dami') else 42
    return args


def _run_matrix(p, args, argv):
    import copy
    import os
    import subprocess
    import sys
    if int(os.environ.get('WORLD_SIZE','1')) > 1:
        p.error('--all is a single-process orchestrator; use torchrun for individual combinations')
    if args.architecture is not None or any(a.startswith('--initialization') for a in argv):
        p.error('--all selects all architectures and initializations; do not also select one combination')
    if args.config or args.model_size.upper()!='100M' or args.init_mode!='paper' or args.memory_file or args.calibration_jsonl or args.gdn_parameters or args.resume_from_checkpoint:
        p.error('--all uses the 15 bundled 100M paper presets; use individual runs for custom initialization/configuration/resume')
    common=[]
    i=0
    while i<len(argv):
        value=argv[i]
        if value=='--all':i+=1;continue
        if value=='--output-dir':i+=2;continue
        if value.startswith('--output-dir='):i+=1;continue
        common.append(value);i+=1
    runs=[]
    root=Path(args.output_dir)
    visible_device=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0].strip()
    for arch in ARCHITECTURES:
        for scheme in INITIALIZATIONS:
            child=copy.copy(args)
            child.architecture=arch;child.initialization=scheme
            _training_defaults(child)
            directory=root/arch/scheme
            runs.append({'architecture':arch,'initialization':scheme,'seed':child.seed,
                         'batch_size':child.batch_size,'warmup_steps':child.warmup_steps,
                         'cuda_visible_devices':visible_device,
                         'output_dir':str(directory),'status':'pending',
                         'command':[sys.executable,str(ROOT/'train.py'),*common,
                                    '--architecture',arch,'--initialization',scheme,
                                    '--output-dir',str(directory)]})
    if args.dry_run:
        print(json.dumps({'count':15,'runs':runs},indent=2));return
    if any((Path(run['output_dir'])/'final').exists() for run in runs):
        p.error('A planned run already has a final model. Select a new output root or resume individual runs.')
    root.mkdir(parents=True,exist_ok=True)
    journal=root/'matrix.json'
    def save():
        journal.write_text(json.dumps({'count':15,'runs':runs},indent=2)+'\n',encoding='utf-8')
    save()
    for index,run in enumerate(runs,1):
        print(f"[{index}/15] {run['architecture']} / {run['initialization']}",flush=True)
        run['status']='running';save()
        try:
            child_env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible_device)
            subprocess.run(run['command'],check=True,env=child_env)
        except BaseException:
            run['status']='failed';save();raise
        run['status']='completed';save()


def training_main(argv=None):
    """One entry point for five architectures times three initializations."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description='Train Random, PoST or DAMI on five architectures')
    p.add_argument('--architecture',choices=ARCHITECTURES)
    p.add_argument('--initialization',choices=INITIALIZATIONS,default='dami')
    p.add_argument('--all',action='store_true',help='Run all 15 paper presets sequentially')
    p.add_argument('--train-files',nargs='+',required=True,help='Explicit local parquet paths or glob patterns')
    p.add_argument('--tokenizer',required=True,help='HF identifier or local tokenizer directory')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--model-size',default='100M',help='Only 100M has a bundled paper configuration')
    p.add_argument('--init-mode',choices=['paper','calibrate','design'],default='paper')
    p.add_argument('--calibration-jsonl',type=Path)
    p.add_argument('--memory-file',type=Path,help='Load an exported memory safetensors file labeled for the selected initialization')
    p.add_argument('--gdn-parameters',type=Path)
    p.add_argument('--init-seed',type=int,default=42)
    p.add_argument('--seed',type=int)
    p.add_argument('--steps',type=int,default=9537)
    p.add_argument('--seq-len',type=int,default=4096)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--gradient-accumulation',type=int)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--warmup-steps',type=int)
    p.add_argument('--save-steps',type=int,default=1000)
    p.add_argument('--resume-from-checkpoint')
    p.add_argument('--dry-run',action='store_true',help='Print resolved recipe without constructing a model')
    args = p.parse_args(argv)
    if args.all:
        return _run_matrix(p,args,argv)
    if args.architecture is None:
        p.error('Choose --architecture, or --all for the complete matrix')
    _training_defaults(args)
    architecture=args.architecture
    if args.initialization!='dami' and args.init_mode=='calibrate':
        p.error('Only DAMI uses activation calibration')
    import os
    import glob
    world = int(os.environ.get('WORLD_SIZE','1'))
    accum = args.gradient_accumulation
    if accum is None:
        if 128 % (args.batch_size * world):
            p.error('Set --gradient-accumulation explicitly; global batch 128 is not divisible by microbatch * GPUs')
        accum = 128 // (args.batch_size * world)
    if min(accum,args.steps,args.seq_len,args.batch_size) < 1:
        p.error('Batch, accumulation, steps and sequence length must be positive')
    if args.model_size.upper() != '100M' and args.config is None:
        p.error('A different size requires --config with explicit layer/head dimensions')
    recipe = dict(vars(args),gradient_accumulation=accum,
                  global_batch=args.batch_size*world*accum,
                  tokens=args.batch_size*world*accum*args.steps*args.seq_len)
    if args.dry_run:
        print(json.dumps(recipe,default=str,indent=2));return
    from safetensors.torch import load_file
    from transformers import Trainer, TrainingArguments, AutoTokenizer, default_data_collator, set_seed
    from datasets import IterableDataset
    if not torch.cuda.is_available():
        raise RuntimeError('FLA training requires an NVIDIA GPU and Linux CUDA environment')
    if world==1 and torch.cuda.device_count()>1:
        p.error('Single-process training needs one visible GPU. Set CUDA_VISIBLE_DEVICES, or use torchrun for distributed training.')
    files = sorted({str(Path(f).resolve()) for pattern in args.train_files for f in glob.glob(pattern)})
    if not files:
        p.error('No training parquet files matched --train-files')
    local_rank = int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local_rank)
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
    model = build_model(architecture,config,args.init_seed).to(f'cuda:{local_rank}')
    if architecture in ('mamba2','rwkv7') or args.init_mode == 'calibrate':
        model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(f'Tokenizer vocabulary {len(tokenizer)} differs from model vocabulary {model.config.vocab_size}')
    if args.memory_file:
        manifest=args.memory_file.parent/'initialization.json'
        if manifest.is_file():
            data=json.loads(manifest.read_text(encoding='utf-8'))
            tag=data.get('initialization',data.get('method','').lower())
            if tag and tag!=args.initialization:
                raise ValueError(f'Memory file is labeled {tag}, not {args.initialization}')
        load_memory(model,load_file(str(args.memory_file)))
    else:
        calibration = _token_batches(args.calibration_jsonl,8) if args.calibration_jsonl else None
        gdn = load_file(str(args.gdn_parameters)) if args.gdn_parameters else None
        initialize(model,architecture,scheme=args.initialization,mode=args.init_mode,
                   calibration_batches=calibration,gdn_parameters=gdn,init_seed=args.init_seed)
    # The GDN calibration runs in bf16; the GDN-family trainer keeps fp32 masters.
    if architecture in ('gdn','kda','gdn2'):
        model.float()
    model.config.use_cache = False
    # Train-mode RNG is separate from the common step-zero initialization seed.
    set_seed(args.seed)
    rank = int(os.environ.get('RANK','0'))
    output = Path(args.output_dir)
    if rank == 0:
        export_parameters(memory_parameters(model),output/'initialization',architecture,recipe,scheme=args.initialization)
        model.config.save_pretrained(output/'initialization')
        (output/'recipe.json').write_text(json.dumps(recipe,default=str,indent=2)+'\n',encoding='utf-8')
    data = IterableDataset.from_generator(_packed_examples,gen_kwargs={
        'files':files,'tokenizer_path':args.tokenizer,'seq_len':args.seq_len})
    # Respect FLA's no-weight-decay markers for A_log and dt_bias.
    groups = [[],[]]
    for name,value in model.named_parameters():
        if value.requires_grad:
            no_decay = value.ndim < 2 or getattr(value,'_no_weight_decay',False) or name.endswith('bias')
            groups[int(no_decay)].append(value)
    optimizer = torch.optim.AdamW([{'params':groups[0],'weight_decay':.1},
                                   {'params':groups[1],'weight_decay':0.}],
                                  lr=args.lr,betas=(.9,.95),eps=1e-15)
    def lr_factor(step):
        if step < args.warmup_steps:
            return step / max(1,args.warmup_steps)
        progress = min(1.,(step-args.warmup_steps)/max(1,args.steps-args.warmup_steps))
        return .1 + .9 * .5 * (1 + math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    options = TrainingArguments(output_dir=str(output),max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,gradient_accumulation_steps=accum,
        bf16=True,learning_rate=args.lr,max_grad_norm=1.,logging_steps=10,
        save_steps=args.save_steps,save_total_limit=2,report_to='none',seed=args.seed,
        data_seed=args.seed,dataloader_num_workers=0,remove_unused_columns=False,
        ddp_find_unused_parameters=False,torch_compile=False)
    trainer = Trainer(model=model,args=options,train_dataset=data,
                      data_collator=default_data_collator,optimizers=(optimizer,scheduler))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output/'final'))
    if rank == 0:
        tokenizer.save_pretrained(output/'final')


def main():
    p = argparse.ArgumentParser(description='Export Random/PoST/DAMI parameters or calibrate a new DAMI model')
    p.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    p.add_argument('--initialization',choices=INITIALIZATIONS,default='dami')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--mode',choices=['paper','calibrate','design'],default='paper')
    p.add_argument('--model-size',default='100M',help='Other sizes require an explicit --config')
    p.add_argument('--config',type=Path)
    p.add_argument('--calibration-jsonl',type=Path)
    p.add_argument('--gdn-parameters',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
    p.add_argument('--tau-min',type=float,default=1.)
    p.add_argument('--tau-max',type=float,default=4096.)
    args = p.parse_args()
    if args.model_size.upper() != '100M' and args.config is None:
        p.error('A different size requires --config with explicit model dimensions')
    if args.mode == 'paper':
        if args.config or args.calibration_jsonl or args.tau_min != 1. or args.tau_max != 4096.:
            p.error('Paper mode exports the fixed 100M archive; custom configurations require calibrate/design')
        params = paper_parameters(args.architecture,args.initialization)
        metadata = {'mode':'paper','model_size':'approximately 100M',
                    'reference':'reference/100m.json' if args.initialization=='dami' else 'reference/controls_100m.json'}
    else:
        config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
        model = build_model(args.architecture,config).to(device=args.device,dtype=getattr(torch,args.dtype))
        batches = _token_batches(args.calibration_jsonl,8) if args.calibration_jsonl else None
        from safetensors.torch import load_file
        gdn = load_file(str(args.gdn_parameters)) if args.gdn_parameters else None
        params = initialize(model,args.architecture,scheme=args.initialization,mode=args.mode,calibration_batches=batches,
                            gdn_parameters=gdn,tau_min=args.tau_min,tau_max=args.tau_max)
        metadata = {'mode':args.mode,'config':model.config.to_dict(),
                    'calibration':getattr(model,'dami_calibration_report',None)}
    print(export_parameters(params,args.output_dir,args.architecture,metadata,scheme=args.initialization))


if __name__ == '__main__':
    main()
