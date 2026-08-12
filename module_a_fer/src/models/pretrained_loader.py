import os
import torch

def load_local_pretrained(model, filename):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pretrained-models', filename)
    if os.path.exists(local):
        state = torch.load(local, map_location='cpu')
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            elif 'state_dict' in state:
                state = state['state_dict']
        model.load_state_dict(state, strict=False)
        return True
    return False

