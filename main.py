import argparse
import torch
from termcolor import colored
from config import create_config
from functionality import get_trainer, get_model, get_backbone, get_optimizer, get_dataset, get_dataloader, get_criterion, get_transform, adjust_learning_rate

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
    print('prefix: ',prefix)
    p = create_config(args.root_dir, "config_files/"+prefix+'.yml', prefix)
    transform = get_transform(p)
    dataset = get_dataset(p,transform)
    train_loader = get_dataloader(p,dataset)
    backbone = get_backbone(p)
    model = get_model(p,backbone['backbone'],backbone['out_dim'])
    loss_function = get_criterion(p)
    optimizer = get_optimizer(p)
    trainer = get_trainer(p,loss_function)

    end_epoch = p['epochs']
    start_epoch = 0

    # load from checkpoint

    # Main loop
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, end_epoch):
        print(colored('Epoch %d/%d' %(epoch, end_epoch), 'yellow'))
        print(colored('-'*15, 'yellow'))
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        # Train
        print('Train ...')
        trainer.train_one_epoch(train_loader, model, optimizer, epoch)
        # save training configuration to checkpoint 

    torch.save(trainer.best_model.get_backbone().state_dict(),p['result_save_path'])
    print("-----------TRAINING_PROCESS_FINISHED--------------")
