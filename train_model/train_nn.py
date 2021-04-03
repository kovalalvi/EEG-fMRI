import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from shutil import copyfile



def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    file_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, file_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        copyfile(file_path, best_fpath)

def load_checkpoint(model, filepath, freeze_param =True):
    checkpoint = torch.load(filepath)
    # model = checkpoint['model']
    model.load_state_dict(checkpoint)
    if freeze_param:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    return model

def train_regression(train_loader, val_loader, model, losses, filepath, device,
                     learn_rate=0.001, EPOCHS=1, w_d = 0):
    model = model.to(device)
    batch_size = train_loader.batch_size
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay = w_d)
    min_loss = 10000
    print("Starting Training of our model",
          "\nNumber of samples", batch_size*len(train_loader),
          "\nSize of batch:", batch_size,"Number batches", len(train_loader))

    for epoch in range(EPOCHS):
        avg_loss_mse = 0 ; avg_loss_bce = 0
        counter = 0
        # training
        for x_batch, y_batch in train_loader:
            model.train()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            counter += 1
            model.zero_grad()
            out = model(x_batch)

            loss = criterion_mse(out, y_batch)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.gru.parameters(), 1)

            optimizer.step()
            avg_loss_mse += loss.item()
            # avg_loss_bce += loss2.item()

            # LOGGING
            if counter%250 == 0:
                train_corr, _ = pearsonr(out.to('cpu').detach()[:, -1].view(-1),
                                         y_batch.to('cpu').detach()[:, -1].view(-1))

                # count val_loss on random batch
                x, y = next(iter(val_loader))
                x = x.to(device) ; y = y.to(device)
                with torch.no_grad():
                    model.eval()
                    yhat = model(x)
                    val_loss = criterion_mse(y, yhat)
                    val_corr, _ = pearsonr(yhat.to('cpu').view(-1),
                                           y.to('cpu').view(-1))
                # only last element use
                general_out_ = 'Epoch {}  Step: {}/{} '
                train_loss_ = 'avg train_loss_mse....: {:.2}, '
                # out3 = 'avg train_loss_bce....: {:.3}, '
                train_corr_ = ' train_corr: {:.2}'
                val_loss_ = " val_loss: {:.2} "
                val_corr_ = " val_corr: {:.2} "

                out_string = general_out_ + train_loss_ + train_corr_ + val_loss_ + val_corr_
                print(out_string.format(epoch+1, counter, len(train_loader), avg_loss_mse/counter,
                                        train_corr, val_loss.item(), val_corr))
        #validation
        with torch.no_grad():
            model.eval()
            val_loss = []
            val_corr = []
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device); y_batch = y_batch.to(device)
                out = model(x_batch)
                val_loss.append(criterion_mse(out, y_batch).item())

                corr, _ = pearsonr(out.to('cpu').view(-1), y_batch.to('cpu').view(-1))
                val_corr.append(corr)

            val_loss = torch.mean(torch.tensor(val_loss))
            val_corr = torch.mean(torch.tensor(val_corr))
            train_value = '\nPer epoch || train_loss:  {:.2} '
            out_string = "val_loss: {:.2}: val_corr: {:.2}"
            sum_str = train_value + out_string
            print(sum_str.format(avg_loss_mse/counter, val_loss.item(), val_corr.item()))

        # logging and saving
        # losses['train'].append((avg_loss_mse+avg_loss_bce)/counter)
        losses['train'].append((avg_loss_mse)/counter)
        losses['test'].append(val_loss)

        # model_tmp = model.to('cpu')
        checkpoint = model.state_dict()
        save_checkpoint(checkpoint, False, filepath[0], filepath[1])
        if val_loss<min_loss:
            min_loss = val_loss
            save_checkpoint(checkpoint, True, filepath[0], filepath[1])

    return model, losses
