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

def get_parameters_will_optim(model, optimizer):
    # Extract optimizer parameters
    # Thanks to The DeepSeek R1 model give me the result below
    optimizer_params = []
    for group in optimizer.param_groups:
        optimizer_params.extend(group['params'])

    # Map model parameters to their names
    model_params = {param: name for name, param in model.named_parameters()}

    # Find tuned parameter names
    tuned_parameters = [model_params[param] for param in optimizer_params if param in model_params]

    print("Parameters being tuned:", tuned_parameters)
    return tuned_parameters