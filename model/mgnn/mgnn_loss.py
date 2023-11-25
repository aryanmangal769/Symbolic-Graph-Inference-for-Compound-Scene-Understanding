import torch
import torch.nn as nn

class MGNNLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(MGNNLoss, self).__init__()
        self.alpha = alpha

    def forward(self, GSNN_outputs, verbs):
        loss = torch.tensor(0.0, device=GSNN_outputs.device, requires_grad=True)
        # print(GSNN_outputs.shape)
        for i in range(GSNN_outputs.shape[0]):
            GSNN_output = GSNN_outputs[i]
            # print(GSNN_output)
            # print(verbs)
            verb = verbs[i].item()  # Ensure verb is a Python scalar
            # print(verb)
            # Check if verb is in GSNN_output
            condition = GSNN_output == verb
            if condition.sum() == 0:
                # If verb is not in GSNN_output, add the maximum value of GSNN_output to the loss
                loss = loss+ torch.tensor(1.0, device=GSNN_outputs.device)
            else:
                # Calculate the loss using torch.where
                arange_tensor = torch.arange(GSNN_output.shape[0]).to(GSNN_outputs.device)
                loss =loss+ torch.where(condition, self.alpha * torch.ones_like(GSNN_output) * arange_tensor, torch.zeros_like(GSNN_output)).sum()

        return loss
