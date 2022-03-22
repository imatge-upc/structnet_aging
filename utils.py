import torch
import numpy as np
def seed_everything(seed):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
#############################
#         TRAINING          #
#############################
import torch
def train_step(model, data, optimizer, criterion, device):
    data = data.to(device)
    optimizer.zero_grad()

    out = model(data).squeeze()
    loss = criterion(out, data.y)

    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def eval_step(model, data, criterion, device):
    data = data.to(device)
    out  = model(data).squeeze()
    loss = criterion(out, data.y)
    return loss, out

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        loss = train_step(model, data, optimizer, criterion, device)
        total_loss += loss.item()
    return total_loss

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    y_hat, y_true = list(), list()    
    for data in loader:
        loss, out = eval_step(model, data, criterion, device)
        total_loss += loss.item()
        y_hat.append(out)
        y_true.append(data.y)
    
    # concat batched predictions / true labels
    y_hat  = torch.cat(y_hat,  dim=0).cpu().numpy().ravel()
    y_true = torch.cat(y_true, dim=0).cpu().numpy().ravel()
    
    # compute metrics
    metrics=dict()
    metrics['loss'] = total_loss / y_hat.shape[0]
    metrics['mse']  = np.power(y_true - y_hat, 2).mean()
    metrics['mae']  = np.abs(y_true - y_hat).mean()
    metrics['corr'] = np.corrcoef(y_true, y_hat)[0,1]
    
    return metrics

def train_and_eval(epochs, model, 
                  train_loader, valid_loader, 
                  criterion, optimizer, device,
                  score="-mae", best_model_path="./best_model.pt"):
    metrics         = list()
    curr_best_score = 0.0

    for epoch in range(1, epochs+1):
        epoch_metrics = dict()

        # 1 - train the model
        _ = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 2 - eval the model
        epoch_metrics["train"] = eval_epoch(model, train_loader, criterion, device)
        epoch_metrics["valid"] = eval_epoch(model, valid_loader, criterion, device)

        # 3 - save checkpoint if it's the best wrt a SCORE
        curr_score = - epoch_metrics["valid"][score[1:]] if score.startswith("-") else epoch_metrics["valid"][score[1:]]
        if (epoch==1 or curr_score > curr_best_score):
            curr_best_score = curr_score
            storage    = {
                'epoch'            : epoch,
                'model_state_dict' : model.state_dict(),
                'opt_sate_dict'    : optimizer.state_dict(),
                'metrics'          : epoch_metrics
            }  
            torch.save(storage, best_model_path)
            
        metrics.append(epoch_metrics)

        # 4 - plot
        if epoch % 10 == 0 or epoch == 1:
            score_kw = score[1:] if score.startswith("-") else score
            print("Epoch : {:03d} TrainLoss : {:.4f} ValidLoss : {:.4f} TrainScore : {:.4f} ValidScore : {:.4f}"\
                .format(epoch, epoch_metrics["train"]["loss"], 
                               epoch_metrics["valid"]["loss"],
                               epoch_metrics["train"][score_kw],
                               epoch_metrics["valid"][score_kw]))
    
    return metrics