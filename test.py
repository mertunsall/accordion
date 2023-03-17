import torch
from models.AdaptableResNetCifar import AdaptableResNetCifar
from typing import List

def forward_and_get_activations(net: AdaptableResNetCifar, x: torch.Tensor, eval = True) -> List[torch.Tensor]:
    activations = []

    def getActivation():
        # the hook signature
        def hook(model, inp, output):
            #print(inp[0].shape)
            activations.append(output.detach())
        return hook 

    hooks = []

    for layer in net.layers:
        for block in layer:
            hooks.append(block.register_forward_hook(getActivation()))
    
    if eval:
        net.eval()
        
    out = net(x)

    for hook in hooks:
        hook.remove()

    return out, activations

def test_adaptability(net: AdaptableResNetCifar, num_blocks: int, device: torch.device) -> None:

    net.eval()
    net.reconfigure_blocks(num_blocks)
    x = torch.rand(1,3,32,32).to(device)
    out, activations = forward_and_get_activations(net, x)

    i = 0
    for layer in net.layers:
        for block in layer:
            if i == 0:
                i += 1
                continue
            
            if not block.active:
                if block.downsample is None:
                    assert torch.allclose(activations[i-1], activations[i])
                else:
                    assert torch.allclose(block.downsample(activations[i-1]), activations[i])
            i += 1
    
    print(f'Test completed with no error for {num_blocks} blocks.')
    return
