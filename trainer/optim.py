"""Sonata's encoder-block LR decay, adapted to LitePT's actual block names."""

import re


def layerwise_parameter_groups(student, base_lr, layer_decay):
    """Keep stems/pooling/heads at base LR; decay earlier encoder blocks.

    Discover blocks from the module rather than assuming PTv3's depths. LitePT-S
    has 14 blocks, so their multipliers run from ``layer_decay**13`` to 1.
    Frozen parameters are excluded, and every trainable parameter occurs once.
    """
    if not 0 < layer_decay <= 1:
        raise ValueError("optimizer.layer_decay must be in (0, 1]")
    pattern = re.compile(r"(?:^|\.)enc(\d+)\.block(\d+)\.")
    parameters = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
    matches = {name: pattern.search(name) for name, _ in parameters}
    blocks = sorted({tuple(map(int, match.groups())) for match in matches.values() if match})
    if not blocks:
        raise ValueError("optimizer.layer_decay requires named encoder blocks (encN.blockN)")
    groups = [{"params": [], "lr": base_lr, "lr_scale": 1.0}]
    block_indices = {}
    for index, block in enumerate(blocks):
        scale = layer_decay ** (len(blocks) - index - 1)
        block_indices[block] = len(groups)
        groups.append({"params": [], "lr": base_lr * scale, "lr_scale": scale})
    for name, parameter in parameters:
        match = matches[name]
        index = block_indices[tuple(map(int, match.groups()))] if match else 0
        groups[index]["params"].append(parameter)
    return groups
