import torch
def count_parameters(model, ifName = False, oLog = None):
    if ifName:
        for name, param in model.named_parameters():
            if oLog is None:
                print(name, param.numel(), param)
            else:
                if name == '_model.oNonLinTRF.LinearKernels.nan.weight':
                    param = param.permute(2, 0, 1)
                elif name in ['_model.oNonLinTRF.oNonLinear.oEncoder.conv.weight', '_model.oNonLinTRF.oNonLinear.oEncoder.conv.bias','_model.oNonLinTRF.bias']:
                    torch.save(param.data, f'{name}.pth')
                

                oLog(name, param.numel(),param.shape, param)
        # for p in model.parameters():
        #     if p.requires_grad:
        #         if oLog is None:
        #             print(p.numel())
        #         else:
        #             oLog(p.numel())

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
