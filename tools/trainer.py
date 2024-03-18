import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import *


dtype = torch.float32 
WEIGHTS = torch.tensor([0.1, 10, 10, 10],dtype=dtype)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preds(prob, boxes):
    num_correct = 0
    predicted = prob            
    # find boxes with highest probability
    max_boxes = np.argmax(predicted,axis=0)     
                
    # GT RESULTS     
    for cls in range(1,4):
        ind = max_boxes[cls]
        winner_prob = predicted[ind,cls]

        if max_boxes[cls]==cls-1:
            num_correct += 1
            result = 'Right'
        else: 
            result = 'Wrong'
        
        print('CLASS ID:'+ str(cls), '('+str(result)+')')
        print('PROBABILITIES')
        print('GT: ' + str(predicted[cls-1,cls]) +', winner:' + str(winner_prob))
    
    return num_correct


def Evaluate(
    val_loader,
    model,
    n_class=3
):
    val_loss = 0
    visualizations = []
    preds, gts = [], []
    num_correct = 0
    num_samples = 0
    
    model.eval()
    for batch_idx, (data, target) in enumerate(val_loader):
        data[0] = data[0].to(DEVICE)
        data[1] = data[1].to(DEVICE)
        data[2] = data[2].to(DEVICE)
        target = target.view(-1)
        target = target.to(DEVICE)
        with torch.no_grad():
            score = model(data)
            prob = F.softmax(score)
            #feed in predictions + bounding box candidates
            correct = get_preds(prob, data[2].view(200,4))
            num_correct += correct
            num_samples += data[0].size(0)
    accuracy = num_correct / (2*n_class)
    print('Got %d / %d correct (%.2f)' % (num_correct, (num_samples*n_class), 100 * accuracy))

    return accuracy

def Train(
    model,
    loss_func,
    optim,
    epochs,
    train_loader,
    val_loader,
    test_loader,
    display_interval = 10
):

    total_prediction = 0
    print("Init Model")
    for i in range(epochs):
        print("Epochs: {}".format(i))
        total_loss = 0
        model.train()
        if i % 3 == 0:
          checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optim.state_dict()}
          save_checkpoint(checkpoint)

        for batch_idx, (data, target) in enumerate(train_loader):
            data[0] = data[0].to(device=DEVICE, dtype=dtype)
            data[1] = data[1].to(device=DEVICE, dtype=dtype)
            data[2] = data[2].to(device=DEVICE, dtype=dtype)
            target = target.to(device=DEVICE, dtype=torch.long)
            
            optim.zero_grad()
            score = model(data)
            loss = loss_func(score, target.view(-1))

            loss_data = loss.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            
            loss.backward()
            optim.step()
            print('Epoch %d, Iteration %d, loss = %.4f' % (i, batch_idx + 1, loss_data))
            total_loss += loss.item()
            
        total_loss /= len(train_loader)
        model.eval()
        val_accuracy = Evaluate(
            val_loader,
            model
            )
    
        print("Epoch Loss: {:.4}, Avg Acc: {:.4}".format(
                total_loss, val_accuracy
            ))
    
    test_accuracy= Evaluate(
        val_loader,
        model,
    )

    print("Test Acc: {:.4}".format(
            test_accuracy
        ))
    
    return model


def Trainer(model, train_loader,val_loader,test_loader,optimizer,num_epochs=25):    
    loss_func = nn.CrossEntropyLoss(weight=WEIGHTS, ignore_index=4)

    best_model = Train(
        model,
        loss_func,
        optimizer,
        num_epochs,
        train_loader,
        val_loader,
        test_loader
    )

    return best_model