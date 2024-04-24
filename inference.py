import yaml
from yaml.loader import SafeLoader
import torch
import torchvision
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import random
import time,json
import copy,sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")
from data_model_loading import load_dataset_inference, load_model

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
        
    return config_data

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print("\n")
        
class StandardizeTransform(nn.Module):
    def __init__(self):
        super(StandardizeTransform, self).__init__()
        self.transform = None
        
    def forward(self, batch_data):
        """
        batch_data: [N, 3, W, H]
        """
        
        mean_values = []
        std_values = []
        
        mean_values.append(batch_data[:,0:1,:,:].mean())
        mean_values.append(batch_data[:,1:2,:,:].mean())
        mean_values.append(batch_data[:,2:3,:,:].mean())
        
        std_values.append(batch_data[:,0:1,:,:].std())
        std_values.append(batch_data[:,1:2,:,:].std())
        std_values.append(batch_data[:,2:3,:,:].std())
        
        self.transform = torchvision.transforms.Normalize(mean_values, std_values)
        
        return self.transform(batch_data)
        
def evaluate(model, loader ,n_classes, device, transformations, fta_path=None, return_logs=False):
    correct = 0;samples =0
    fpr_tpr_auc = {}
    pre_prob = []
    lab = []
    predicted_labels = []

    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = transformations(x)
            x = x.to(device)
            y = y.to(device)
            # model = model.to(config.device)

            scores = model(x)
            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            predictions = predictions.to('cpu')
            y = y.to('cpu')
            predict_prob = predict_prob.to('cpu')

            predicted_labels.extend(list(predictions.numpy()))
            pre_prob.extend(list(predict_prob.numpy()))
            lab.extend(list(y.numpy()))

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        
        print('correct are {:.3f}'.format(correct/samples))

    lab = np.array(lab)
    predicted_labels = np.array(predicted_labels)
    pre_prob = np.array(pre_prob)
    
    
#     binarized_labels = label_binarize(lab,classes=[i for i in range(n_classes)])
#     for i in range(n_classes):
#         fpr,tpr,_ = roc_curve(binarized_labels[:,i],pre_prob[:,i])
#         aucc = auc(fpr,tpr)
#         fpr_tpr_auc[i] = [fpr,tpr,aucc]

#     # print("auc:",{i:j[2] for i,j in fpr_tpr_auc.items()})
    
#     if fta_path is not None:
#         with open(fta_path,'wb') as f:
#             pickle.dump(fpr_tpr_auc,f)
    return fpr_tpr_auc,lab,predicted_labels,pre_prob, correct/samples

def evaluate_single_server(config, server_model, test_loader, transformations, device):
    server_model = server_model.to(device)
    server_model.train()

    if config.get("eval_mode",False) == True:
        print("putting model into eval mode")
        server_model.eval()

    test_fta,y_true,y_pred,prob,acc = evaluate(server_model, test_loader, config['nclass'], device, transformations, return_logs=config['return_logs'])
    
    return acc

def aggregate_server(clientlogits):
    """
    [N, [BS, NC]]
    """
    updated_logits = [logit.unsqueeze(2) for logit in clientlogits] # [BS,NC,1]
    CAT_LOGITS = torch.cat(updated_logits,dim=2) # [BS,NC,N]
    avg_logits = torch.mean(CAT_LOGITS, dim=2) # [BS,NC]
    return avg_logits

def ensembled_acc(models, test_loader, device, return_logs):
    correct = 0
    total_samples = 0
    acc_clients = {}
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            cur_logits = []
            for ijdx, mdl in enumerate(models):
                mdl = mdl.to(device)
                mdl.train()
                output = F.softmax(mdl(data), dim=1)
                cur_logits.append(output)
                
                _,cur_preds = output.max(dim=1)
                cur_mdl_correct = (cur_preds == target).sum()
                acc_clients[ijdx] = acc_clients.get(ijdx, 0.0) + cur_mdl_correct
                acc_clients['mean'] = acc_clients.get('mean', 0.0) + cur_mdl_correct
            
            avg_logits = F.softmax(aggregate_server(cur_logits))
            
            _,predictions = avg_logits.max(dim=1)
            cur_score = (predictions == target).sum()
            correct += cur_score
            total_samples += data.shape[0]
            
            if return_logs:
                progress(idx+1, len(test_loader))
            
    acc = correct / total_samples
    avg_acc_all = {i: j/total_samples for i,j in acc_clients.items()}
    avg_acc_all['mean'] /= len(models)
    return acc, avg_acc_all

def evaluate_no_server(config, client_models, test_loader, transformations, device):
    client_mdl_accuracies = []
    for idx, client_mdl in enumerate(client_models):
        client_mdl = client_mdl.to(device)
        if config.get('eval_mode',False):
            print("Putting in eval mode")
            client_mdl.eval()
        else:
            client_mdl.train()
        test_fta,y_true,y_pred,prob,acc = evaluate(client_mdl, test_loader, config['nclass'], device, transformations, return_logs=config['return_logs'])
        client_mdl_accuracies.append(acc)
    
    overall_acc = sum(client_mdl_accuracies)/len(client_mdl_accuracies)
    return client_mdl_accuracies, overall_acc


def roc_plot(fta, n_classes, roc_title, roc_path):
    plt.figure(figsize=(5,4))
    for i in range(n_classes):
        fpr,tpr,aucc = fta[i]
        plt.plot(fpr,tpr,label=f'auc_{i}: {aucc:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(roc_title)
    plt.legend()
    plt.savefig(roc_path)
    
if __name__ == "__main__":
    
    config = yaml_loader(sys.argv[1])
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")       
     
    transformations = StandardizeTransform()
        
    print(device)
    
    test_loader = load_dataset_inference(config)
    
    if config.get('cronus',False) or config.get("fedmd",False):
        with open(config['client_models_txt'],'r') as f:
            client_models = f.readlines()
            client_models = list(map(lambda x: x[:-1], client_models))
            print(client_models)
        client_mdl_accuracies = []
        for idx, model in enumerate(client_models):
            client_mdl = load_model(model, config['nclass'])
            mdl_name = os.path.join(config["client_models"],f"{config['client_model_name']}_{idx}.pt")
            print(f"loading {mdl_name}")
            client_mdl = client_mdl.to(device)
            client_mdl.load_state_dict(torch.load(mdl_name, map_location=device))
            if config['eval_mode']:
                print('using eval mode')
                client_mdl.eval()
            else:
                client_mdl.train()
            test_fta,y_true,y_pred,prob,acc = evaluate(client_mdl, test_loader, config['nclass'], device, transformations, return_logs=config['return_logs'])
            client_mdl_accuracies.append(acc)
        print(client_mdl_accuracies)
        print(sum(client_mdl_accuracies)/len(client_mdl_accuracies))
            
    else:
        print(f"loading {config['server_model']}")
        print(f"model_name: {config['model']}")

        server_model = load_model(config['model'], config['nclass'])
        server_model = server_model.to(device)
        server_model.load_state_dict(torch.load(config['server_model'],map_location=device))
        server_model.train()

        if config.get("eval_mode",-1) == True:
            print("putting model into eval mode")
            server_model.eval()

        test_fta,y_true,y_pred,prob,acc = evaluate(server_model, test_loader, config['nclass'], device, transformations, return_logs=config['return_logs'])
        roc_plot(test_fta, config['nclass'], config['roc_title'], config['roc_curves'])

        print(classification_report(y_true,y_pred))
        test_pre,test_rec,test_f1,_ = precision_recall_fscore_support(y_true,y_pred)

        print('class-wise')
        print(test_pre)
        print(test_rec)
        print(test_f1)

        print('avg-out')
        print(test_pre.mean())
        print(test_rec.mean())
        print(test_f1.mean())

        print(f"roc_plot saved to {config['roc_curves']}")
