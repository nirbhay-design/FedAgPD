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
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import random
import time,json
import copy,sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")
from data_model_loading import load_dataset, load_model as return_model, load_dataset_test
from data_model_loading import return_opt, StandardizeTransform
from inference import evaluate_single_server, evaluate_no_server, ensembled_acc

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)

    args = sys.argv

    if '-s' in args:
        config_data['SEED'] = int(args[args.index('-s') + 1])
        # config_data['acc_curves'] = config_data['acc_curves'][:-4] + f"_{config_data['SEED']}" + '.png'
        # config_data['acc_curves_pkl'] = config_data['acc_curves_pkl'][:-4] + f"_{config_data['SEED']}" + '.pkl'

    if '-e' in args:
        config_data['total_iterations'] = int(args[args.index('-e') + 1])

    if '-b' in args:
        config_data['beta'] = float(args[args.index('-b') + 1])

    if '-nc' in args:
        config_data['n_clients'] = int(args[args.index('-nc') + 1])
    
    if '-sc' in args:
        config_data['sample_clients'] = float(args[args.index('-sc') + 1])

    if '-rl' in args:
        config_data['return_logs'] = eval(args[args.index('-rl') + 1])

    if '-ac' in args:
        config_data['acc_curves'] = str(args[args.index('-ac') + 1])
        config_data['acc_curves_pkl'] = config_data['acc_curves'][:-3] + 'pkl'

    if '-ci' in args:
        print('here')
        config_data['client_iterations'] = int(args[args.index('-ci') + 1])
    
    if '-si' in args:
        config_data['server_iterations'] = int(args[args.index('-si') + 1])

    if '-clr' in args:
        config_data['lr'] = float(args[args.index('-clr') + 1])

    if '-g' in args:
        gpu_id = int(args[args.index('-g') + 1])
        config_data['gpu'] = gpu_id
        config_data['server_gpu'] = gpu_id

    if '-gm' in args:
        config_data["model"] = args[args.index('-gm') + 1]
    
    if '-cm' in args:
        clientmodels = args[args.index('-cm') + 1]
        config_data['client_model'] = list(map(
                                        lambda x: x.strip(), 
                                        clientmodels.split(',')))
    
    if '-ndm' in args:
        nodmlmodels = args[args.index('-ndm') + 1]
        config_data['NO_DML_MODELS'] = list(map(
                                        lambda x: x.strip(), 
                                        nodmlmodels.split(',')))


    return config_data

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()

class Lossfunction(nn.Module):
    def __init__(self,dml=True):
        super(Lossfunction,self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.distill = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.dml = dml
        
    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.ce(student_logits, labels)
        if self.dml:
            student_log_logits = self.log_softmax(student_logits)
            teacher_softmax_logits = self.softmax(teacher_logits)
            return ce_loss + self.distill(student_log_logits, teacher_softmax_logits)  
        return ce_loss

def return_essentials(config, client_data, distribution):
    n_clients = config['n_clients']
    lr = config['lr']
    nclass = config['nclass']
    model_name = config['client_model']
    mdl_len = len(model_name)
    client_essential_arr = []
    for i in range(n_clients):
        if config['return_logs']:
            progress(i+1,n_clients)
        cur_model_name = model_name[i % mdl_len]
        dml = False if cur_model_name in config['NO_DML_MODELS'] else True
        lossfunction = Lossfunction(dml)
        print(f'client {i+1}, model_name: {cur_model_name} dml: {dml}')
        if dml:
            client_model = return_model(cur_model_name, nclass, config.get('channel',3))
        else:
            client_model = return_model(config['model'], nclass, config.get('channel',3))
        optimizer = return_opt(config.get("opt",-1), client_model, lr, config.get("momentum",None))
        essential = {
            "lossfun": lossfunction,
            "optimizer": optimizer,
            "model":client_model,
            "data":client_data[i],
            "distribution":distribution[i],
            "model_name": cur_model_name,
            "dml": dml
        }
        client_essential_arr.append(essential)
        
    return client_essential_arr

def return_essential_models(model, lr, config):
    return {
        "model":model,
        "optimizer":return_opt(config.get("opt",-1), model, lr, config.get("momentum",None)),
        "lossfun":Lossfunction()
    }

def ensemble_(softmax_logits_arr, ensemble_type, ensemble_softmax):
    logits = torch.zeros_like(softmax_logits_arr[0])
    if ensemble_type == 'max':
        for softmax_logit in softmax_logits_arr:
            logits = torch.max(logits, softmax_logit)
    
    elif ensemble_type=='avg':
        for softmax_logit in softmax_logits_arr:
            logits += softmax_logit
        logits = logits / len(softmax_logits_arr)
        
    
    if ensemble_softmax:
        return F.softmax(logits,dim=1)
    
    return logits
    
def aggregate_models(server_model, clients_model):
    n_clients = len(clients_model)
    params_dict = {}
    server_params = {**server_model.state_dict()}
    for name, params in server_params.items():
        params_dict[name] = torch.zeros_like(server_params[name])
        for client in clients_model:
            client_dict = client.state_dict()
            if params_dict[name].dtype == torch.int64:
                params_dict[name] += client_dict[name] // n_clients
            else:    
                params_dict[name] += client_dict[name] / n_clients
    
    print(server_model.load_state_dict(params_dict))
    
    return server_model

def train_client(model, knowledge_network, train_loader, lossfunction, lossfunction_kn, optimizer, optimizer_kn, transformations, n_epochs, device, return_logs=False): 
    tval = {'trainacc':[],"trainloss":[],'kn_acc':[],'kn_loss':[]}
    model = model.to(device)
    knowledge_network = knowledge_network.to(device)
    model.train()
    knowledge_network.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        curacc = 0
        cur_kn_loss = 0
        cur_knacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            if data.shape[0] == 1:
                continue
            data = transformations(data)    
            data = data.to(device)
            target = target.to(device)
            
            data = Variable(data)
            target = Variable(target)
            
            scores = model(data)
            scores_kn = knowledge_network(data)
            
            loss = lossfunction(scores, Variable(scores_kn), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_loss += loss.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            loss_kn = lossfunction_kn(scores_kn, Variable(scores), target)
            
            optimizer_kn.zero_grad()
            loss_kn.backward()
            optimizer_kn.step()
            
            cur_kn_loss += loss_kn.item() / (len_train)
            scores_kn = F.softmax(scores_kn,dim = 1)
            _,predicted_kn = torch.max(scores_kn,dim = 1)
            correct_kn = (predicted_kn == target).sum()
            samples_kn = scores_kn.shape[0]
            cur_knacc += correct_kn / (samples_kn * len_train)
        
            if return_logs:
                progress(idx+1,len(train_loader))
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        tval['kn_acc'].append(float(cur_knacc))
        tval['kn_loss'].append(float(cur_kn_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f} kn_acc: {cur_knacc:.3f} kn_loss: {cur_kn_loss:.3f}")
    return model, knowledge_network, tval

def train_client_single(model, train_loader, lossfunction, optimizer, transformations, n_epochs, device, return_logs=False): 
    tval = {'trainacc':[],"trainloss":[]}
    model = model.to(device)
    model.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            if data.shape[0] == 1:
                continue
            data = transformations(data)    
            data = data.to(device)
            target = target.to(device)
            
            scores = model(data)
            
            loss = lossfunction(scores, None, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_loss += loss.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader))
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
    return model, tval

def train_server_fedavg(model, local_models, local_kns, lossfunction, optimizer, n_epochs, device, img_size, distribution, return_logs=False): 
    tval = {'trainloss':[]}
    img_size = config['img_size']
    n_clients = config['n_clients']
    noise_batch_size = config['noise_batch_size']
    noise_batch = config['noise_batch']
    
    model = model.to(device)
    for idx, local_kn in enumerate(local_kns):
        local_kns[idx] = local_kns[idx].to(device)
        local_kns[idx].train()
    model.train()
    
    model = aggregate_models(model, local_kns)
    
    for epochs in range(n_epochs):
        cur_loss = 0
        train_loader = [torch.randn(noise_batch_size,config.get('channel',3),*img_size) for i in range(noise_batch)]
        len_train = len(train_loader)
        for idx, data in enumerate(train_loader):    
            data = data.to(device)
            
            scores1 = model(data)
            softmax_score1 = F.log_softmax(scores1,dim=1)
            
            all_scores = []
            
            with torch.no_grad():
                for local_model in local_models:
                    local_model = local_model.to(device)
                    local_model.train()
                    scores2 = local_model(data)
                    softmax_score2 = F.softmax(scores2,dim=1)
                    all_scores.append(softmax_score2)
            
            loss = 0 
            
            for idx, local_softscore in enumerate(all_scores):
                loss += distribution[idx] * lossfunction(softmax_score1, local_softscore)
            
            cur_loss += loss.item() / len_train
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if return_logs:
                progress(idx+1,len_train)
      
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_loss: {cur_loss:.3f}")
    return model, tval

def pipeline(config, client_essential, test_data, transformations, device, server_device, return_logs=False):
    total_iterations = config['total_iterations']
    server_iterations = config['server_iterations']
    client_iterations = config['client_iterations']
    n_clients = config['n_clients']
    nclass = config['nclass']
    server_lr = config['server_lr']
    img_size = config['img_size']
    sample_clients = config['sample_clients']
    clients_one_iter = int(sample_clients * n_clients)
    print(f"clients used in one iteration: {clients_one_iter}")
    
    lossfunction = nn.KLDivLoss(reduction='batchmean')
    server_model = return_model(config['model'], config['nclass'], config.get('channel',3))
    optimizer = return_opt(config.get("opt",-1), server_model, server_lr, config.get("momentum",None))
    
    server_logs_iterations = {}
    test_acc = {}
    
    start_time = time.perf_counter()
    
    for idx in range(total_iterations):
        print(f"iteration [{idx+1}/{total_iterations}]")
        clients_selected = random.sample([i for i in range(n_clients)],clients_one_iter)
        for jdx in clients_selected:
            print(f"############## client {jdx+1} model {client_essential[jdx]['model_name']} ##############")
            
            if client_essential[jdx].get('kn_network',-1) == -1 and client_essential[jdx]['dml'] == True:
                client_essential[jdx]['kn_network'] = copy.deepcopy(server_model)
                
            client_essential[jdx]['optimizer'] = return_opt(
                config.get("opt",-1),
                client_essential[jdx]['model'],
                config['lr'],
                config.get("momentum",None)
            )
            
            if client_essential[jdx]['dml']:
                kn_network = return_essential_models(client_essential[jdx]['kn_network'], config['lr'], config)

                client_model, kn_network_j, log = train_client(
                    client_essential[jdx]['model'],
                    kn_network['model'],
                    client_essential[jdx]['data'],
                    client_essential[jdx]['lossfun'],
                    kn_network['lossfun'],
                    client_essential[jdx]['optimizer'],
                    kn_network['optimizer'],
                    transformations,
                    n_epochs = client_iterations,
                    device=device,
                    return_logs=return_logs
                )

                client_essential[jdx]['model'] = copy.deepcopy(client_model)
                client_essential[jdx]['kn_network'] = copy.deepcopy(kn_network_j)
            else:
                client_model, log = train_client_single(
                    client_essential[jdx]['model'],
                    client_essential[jdx]['data'],
                    client_essential[jdx]['lossfun'],
                    client_essential[jdx]['optimizer'],
                    transformations,
                    n_epochs = client_iterations,
                    device=device,
                    return_logs=return_logs
                )

                client_essential[jdx]['model'] = copy.deepcopy(client_model)
        
        print("############## server + avg + combine ##############")
        server_model, logs = train_server_fedavg(
            server_model,
            [client_essential[i]['model'] for i in clients_selected],
            [client_essential[i]['kn_network'] if client_essential[i]['dml'] == True else client_essential[i]['model'] for i in clients_selected],
            lossfunction,
            optimizer,
            server_iterations,
            server_device,
            img_size,
            [client_essential[i]['distribution'] for i in clients_selected],
            return_logs=return_logs
        )
        
        optimizer = return_opt(config.get("opt",-1), server_model, server_lr, config.get("momentum",None))
        
        
        server_logs_iterations[idx] = logs
        
        for kdx in clients_selected:
            if client_essential[kdx]['dml']:
                client_essential[kdx]['kn_network'] = copy.deepcopy(server_model)
                client_essential[kdx]['kn_network'].train()
            else:
                client_essential[kdx]['model'] = copy.deepcopy(server_model)
                client_essential[kdx]['model'].train()
            
            
        cur_acc = evaluate_single_server(
            config,
            server_model,
            test_data,
            transformations,
            server_device
        )
        
        if config.get('getavg', False):
            
            ensemble_accuracy, avg_acc_all = ensembled_acc(
                [client_essential[i]['model'] for i in range(n_clients)],
                test_data,
                server_device,
                config['return_logs']
            )
            
            avg_model_acc = avg_acc_all['mean']

            print(f"avg models accuracy: {avg_model_acc:.3f}")
            print(f"ensemble accuracy: {ensemble_accuracy:.3f}")
            print(f"per client accuracy:")
            for mndx in range(n_clients):
                print(f"client {mndx}: {avg_acc_all[mndx]:.3f}")

        
        for kdx in clients_selected:
            cpu_device = torch.device('cpu')
            client_essential[kdx]['model'] = client_essential[kdx]['model'].to(cpu_device)
            if client_essential[kdx]['dml']:
                client_essential[kdx]['kn_network'] = client_essential[kdx]['kn_network'].to(cpu_device)

        test_acc[idx+1] = cur_acc
                
    end_time = time.perf_counter()
    elapsed_time = int(end_time - start_time)
    hr = elapsed_time // 3600
    mi = (elapsed_time - hr * 3600) // 60
    print(f"training done in {hr} H {mi} M")
    return server_model, client_essential, server_logs_iterations, test_acc

def plot_logs(logs, save_path,save_pkl):
    combine_logs = []
    for iteration in logs.keys():
        combine_logs.extend(logs[iteration]['trainloss'])
    plt.figure(figsize=(5,4))
    plt.plot([i+1 for i in range(len(combine_logs))],combine_logs)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss vs iterations')
    plt.savefig(save_path)
    with open(save_pkl,'wb') as f:
        pickle.dump(logs,f)
    
def plot_test_acc(logs,save_path,save_pkl):
    idx, values = zip(*list(logs.items()))
    plt.figure(figsize=(5,4))
    plt.plot(idx,values)
    plt.xlabel('iterations')
    plt.ylabel('test accuracy')
    plt.title('acc vs iterations')
    plt.savefig(save_path)
    with open(save_pkl,'wb') as f:
        pickle.dump(logs,f)

if __name__ == "__main__":
    
    config = yaml_loader(sys.argv[1])
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("environment: ")
    print(f"YAML: {sys.argv[1]}")
    for key, value in config.items():
        print(f"==> {key}: {value}")

        
    client_data, distribution = load_dataset(config)
    test_data = load_dataset_test(config)
     
    client_essential = return_essentials(config, client_data, distribution)
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    server_device = torch.device(f"cuda:{config['server_gpu']}" if torch.cuda.is_available() else "cpu")
    print(device)
    print(server_device)
    transformations = StandardizeTransform()
    
    global_model, client_models, logs_iterations,test_logs = pipeline(
        config, 
        client_essential,
        test_data,
        transformations,
        device,
        server_device,
        return_logs=config["return_logs"]
    )
    
    # if not config.get("save", True):
    #     print("not saving")
    #     exit(0)
    
    plot_test_acc(test_logs, config['acc_curves'],config['acc_curves_pkl'])
    
    

    
        
        
    
