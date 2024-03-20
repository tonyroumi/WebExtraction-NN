import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import *




dtype = torch.float32 
WEIGHTS = torch.tensor([0.1, 10, 10, 10],dtype=dtype)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_every = 10


def get_preds(prob):
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
        sys.stdout.flush()
    
    return num_correct

def get_acc(scores):
    title_acc = 0
    date_acc = 0
    content_acc = 0
    with torch.no_grad():
        predicted = F.softmax(scores)
               
    # find boxes with highest probability
    max_boxes = np.argmax(predicted,axis=0)     
                
    # GT RESULTS     
    for idx, cls in enumerate(range(1, 4), start=0):
        if max_boxes[cls] == idx:
            if idx == 0:
                title_acc += 1
            elif idx == 1:
                date_acc += 1
            elif idx == 2:
                content_acc += 1
    
    return title_acc, date_acc, content_acc

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
    length = 0
    
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
            correct = get_preds(prob)
            num_correct += correct
            num_samples += data[0].size(0)
            length+=1
    accuracy = float(float(num_correct) / float(length*n_class))
    print('Got %d / %d correct (%.2f)' % (num_correct, (length*n_class), accuracy))

    return accuracy

def Train(
    model,
    loss_func,
    optim,
    epochs,
    position_maps,
    train_loader,
    val_loader,
    test_loader,
    display_interval = 10
):
    title_results = []
    date_results = []
    content_results = []
    # position_title_results = []
    # position_date_results = []
    # position_content_results = []
    

    total_prediction = 0
    iters = 0
    print("Init Model")
    for i in range(epochs):
        print("Epochs: {}".format(i))
        total_loss = 0
        model.train()
        if i+1 % 5 == 0:
          checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optim.state_dict()}
          save_checkpoint(checkpoint)

        for batch_idx, (data, target) in enumerate(train_loader):
            data[0] = data[0].to(device=DEVICE, dtype=dtype)
            data[1] = data[1].to(device=DEVICE, dtype=dtype)
            data[2] = data[2].to(device=DEVICE, dtype=dtype)
            target = target.to(device=DEVICE, dtype=torch.long)
            
            optim.zero_grad()
            score = model(data)
            correct = get_acc(score)
            title_results.append(correct[0])
            date_results.append(correct[1])
            content_results.append(correct[2])
    
            

            # results_with_position = get_results_with_position(data[2], score, position_maps)
            # position_title_results.append(results_with_position[0])
            # position_date_results.append(results_with_position[1])
            # position_content_results.append(results_with_position[2])

            loss = loss_func(score, target.view(-1))

            loss_data = loss.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            
            loss.backward()
            optim.step()
            if (batch_idx + 1) % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (i, batch_idx + 1, loss_data))
                total_loss += loss.item()
        
        # position_title_accuracy = np.mean(position_title_results)
        # position_date_accuracy = np.mean(position_date_results)
        # position_content_accuracy = np.mean(position_content_results)
        
        # print('NET+POSITION: title accuracy:'+ position_title_accuracy)
        # print('NET+POSITION: position accuracy:'+ position_date_accuracy)
        # print('NET+POSITION: content accuracy:'+ position_content_accuracy)
        # sys.stdout.flush()
        print('NET TRAINING title accuracy ' + str(np.mean(title_results)))
        print('NET TRAINING date accuracy ' + str(np.mean(date_results)))
        print('NET TRAINING content accuracy ' + str(np.mean(content_results)))
            
        total_loss /= len(train_loader)
        model.eval()
        print("Checking accuracy on validaition set")
        val_accuracy = Evaluate(
            val_loader,
            model
            )
        total_prediction += val_accuracy
        iters += 1
        avg_accuracy = total_prediction/iters
        print("Epoch Loss: {:.4}, Avg Acc: {:.4}".format(
                total_loss, avg_accuracy
            ))
    
    test_accuracy= Evaluate(
        val_loader,
        model,
    )
    


    print("Test Acc: {:.4}".format(
            test_accuracy
        ))
    
    return model


def Trainer(model, train_loader,val_loader,test_loader,optimizer, position_maps=None ,num_epochs=25):    
    loss_func = nn.CrossEntropyLoss(weight=WEIGHTS, ignore_index=4)

    best_model = Train(
        model,
        loss_func,
        optimizer,
        num_epochs,
        position_maps,
        train_loader,
        val_loader,
        test_loader
    )

    return best_model