def eval_metrics(outputs, labels):



def validation(model, val_loader):
    model.eval()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs).view(-1)
        metrics = eval_metrics(outputs, labels)

