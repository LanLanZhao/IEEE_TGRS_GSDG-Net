import torch.optim as optim


def load_scheduler(model_name, model):
    if model_name != 'GSDG':
        raise ValueError("This release keeps only the GSDG model.")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = None
    epoch = 400
    return optimizer, scheduler, epoch
