import argparse
import torch
from termcolor import colored
import pandas as pd
import copy
from config import create_config
from cluster import cluster_module
import torchvision.transforms as transforms
from functionality import get_trainer, get_model, get_backbone, get_optimizer, get_dataset, get_val_dataset ,get_dataloader, get_criterion, get_transform, adjust_learning_rate, collate_custom, validation_loader, evaluate_all

FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier',default=0)
FLAGS.add_argument('-p',help='prefix')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')


args = FLAGS.parse_args()
prefix = args.p

print('§1_prefix: ',prefix)
p = create_config(args.root_dir, "config_files/"+prefix+'.yml', prefix)
p['device'] = 'cuda:'+str(args.gpu)

transform = get_transform(p)
print('§3_transform: ',str(transform)) #@
    #ok#
dataset = get_dataset(p,transform)
print("§4_dataset_type: ",type(dataset)) #@
    #ok#
train_loader = get_dataloader(p,dataset)
print('§5_train_loader',str(train_loader)) #@
    #ok#
backbone = get_backbone(p)
    #print('§6_backbone: ',str(backbone)) #@
    #ok#
model = get_model(p,backbone['backbone'],backbone['out_dim'])
    #print('§7_model ',str(model)) #@

loss_function = get_criterion(p)
print('§8_criterion: ',str(loss_function)) #@
    #ok#
optimizer = get_optimizer(p,model)
    #print('§9_optimizer: ',str(optimizer)) #@
    #ok#
trainer = get_trainer(p,loss_function)
print('§10_trainer: ',str(trainer)) #@
    #ok#
val_loader = validation_loader(p)

end_epoch = p['epochs']
start_epoch = 0

version = p['version']

print(colored('Starting main loop', 'blue'))
for epoch in range(start_epoch, end_epoch):
    print(colored('Epoch %d/%d' %(epoch, end_epoch), 'yellow'))
    print(colored('-'*15, 'yellow'))
    # Adjust lr
    lr = adjust_learning_rate(p, optimizer, epoch)
    print('Adjusted learning rate to {:.5f}'.format(lr))

    print('Train ...')
    trainer.train_one_epoch(train_loader, model, optimizer, epoch)


torch.save(trainer.best_model.state_dict(),p['result_save_path'])
results = evaluate_all(p,val_loader, model,p['device'])
session_row = pd.DataFrame({"weighted_kNN": results['weighted_kNN'],"k_means_accuracy":results['ACC'],"AMI":results['AMI'],"ARI":results['ARI']},index=[prefix])
print(session_row)

with open('results.txt','a') as f:       
    f.writelines('wKNN.'+prefix+': '+str(results['weighted_kNN']))
    f.writelines('ACC.'+prefix+': '+str(results['ACC']) )
        
print("-----------TRAINING_PROCESS_FINISHED--------------")
