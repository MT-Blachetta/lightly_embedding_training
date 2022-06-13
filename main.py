import argparse
import torch
from termcolor import colored
from config import create_config
from cluster import cluster_module
import torchvision.transforms as transforms
from functionality import get_trainer, get_model, get_backbone, get_optimizer, get_dataset, get_val_dataset ,get_dataloader, get_criterion, get_transform, adjust_learning_rate, collate_custom

FLAGS = argparse.ArgumentParser(description='loss training')
FLAGS.add_argument('-gpu',help='number as gpu identifier')
FLAGS.add_argument('-list',help='path to the trial list')
FLAGS.add_argument('--root_dir', help='root directory for saves', default='RESULTS')
#FLAGS.add_argument('--config_exp', help='Location of experiments config file')
#FLAGS.add_argument('--model_path', help='path to the model files')

#prefix = 'default'

args = FLAGS.parse_args()

with open(args.list,'r') as lf:
    pstr = lf.read()
    session_list = eval(pstr.strip(' \n'))

for prefix in session_list:
    print('§1_prefix: ',prefix) #@
    #ok#
    p = create_config(args.root_dir, "config_files/"+prefix+'.yml', prefix)
    print(p) #@
    #ok#
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
    print('§6_backbone: ',str(backbone)) #@
    #ok#
    model = get_model(p,backbone['backbone'],backbone['out_dim'])
    print('§7_model ',str(model)) #@
    #ok#
    loss_function = get_criterion(p)
    print('§8_criterion: ',str(loss_function)) #@
    #ok#
    optimizer = get_optimizer(p,model)
    print('§9_optimizer: ',str(optimizer)) #@
    #ok#
    trainer = get_trainer(p,loss_function)
    print('§10_trainer: ',str(trainer)) #@
    #ok#
    end_epoch = p['epochs']
    start_epoch = 0

    clusterer = cluster_module(num_clusters=p['num_classes'],temperature=p['temperature'],gpu_id=0)

    #val_transform = transforms.Compose([
    #                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
    #                transforms.ToTensor(), 
    #                transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    
    #full_dataset = get_val_dataset(p,val_transform)
    cluster_loader = torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom, drop_last=False, shuffle=False)
    cluster_features = torch.zeros([len(dataset),p['feature_dim']])
    cluster_features_I = torch.zeros([len(dataset),p['feature_dim']])


    # load from checkpoint

    # Main loop
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, end_epoch):
        print(colored('Epoch %d/%d' %(epoch, end_epoch), 'yellow'))
        print(colored('-'*15, 'yellow'))
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # compute clustering
        model.eval()
        i = 0
        with torch.no_grad():
            model = model.cuda()
            for batch in cluster_loader:
                origin_batch = batch['image']
                bsize = len(origin_batch)
                augmented_batch = batch['image_augmented'][0]
                origin_batch = origin_batch.cuda()
                augmented_batch = augmented_batch.cuda()
                first = model.group(origin_batch)
                second = model.group(augmented_batch)
                cluster_features[i:i+bsize] = first
                cluster_features_I[i:i+bsize] = second
                i += bsize
            clusterer.features = cluster_features
            clusterer.features_I = cluster_features_I
            print('compute features for clustering OK')
            clusterer.clustering()

        # Train
        print('Train ...')
        trainer.train_one_epoch(train_loader, model, optimizer, epoch, clusterer)
        # save training configuration to checkpoint 
        

    torch.save(trainer.best_model.get_backbone().state_dict(),p['result_save_path'])
    print("-----------TRAINING_PROCESS_FINISHED--------------")
