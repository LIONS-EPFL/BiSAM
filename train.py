import torch
import os
import time

from args import parse_args

from loss_fnc import smooth_crossentropy, perturbation_loss_tanh, perturbation_loss_log
from utility.setup import get_dataset
from utility.log import Log
from utility.initialize import initialize
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.setup import get_model, get_optim, get_dataset



if __name__ == "__main__":
    args = parse_args()
    initialize(args, seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_path = './save/' + args.optim + "/adaptive("+str(args.adaptive)+")_rho"+str(args.rho)+"_lr"+str(args.learning_rate)+"_bz"+str(args.batch_size)+"_"+str(args.model)+"_"+time.strftime('%m%d%H%M%S')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    dataset = get_dataset(args)
        
    log = Log(log_each=10, save_path=save_path)
    model = get_model(args, device)
    
    base_optimizer = torch.optim.AdamW if args.adam else torch.optim.SGD
    optimizer = get_optim(model, base_optimizer, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
            
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            
            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            if args.optim == 'sam':
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            elif args.optim == 'bisam_log':
                loss = perturbation_loss_log(predictions, targets, args.mu)
            elif args.optim == 'bisam_tanh':
                loss = perturbation_loss_tanh(predictions, targets, args.mu, args.alpha)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)
            
            # second forward-backward step
            disable_running_stats(model)
            pred = model(inputs)
            smooth_crossentropy(pred, targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)
            
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                lr = optimizer.base_optimizer.param_groups[0]["lr"]
                log(model, loss.cpu(), correct.cpu(), lr)
            
        scheduler.step()

        model.eval()
        log.eval(len_dataset=len(dataset.valid))

        with torch.no_grad():
            for batch in dataset.valid:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs) 
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()
    
    model.load_state_dict(torch.load(save_path+"/best_model.pth"))
    model.eval()
    total_loss, correct, steps = 0, 0, 0

    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            total_loss += loss.sum().item()
            correct += (torch.argmax(predictions, 1) == targets).sum().item()
            steps += loss.size(0)
            
    loss = total_loss/steps
    accuracy = correct/steps
    print("Best_Test_Accuracy: " + str(accuracy*100) + '%')
    print("Best_Test_Loss: " + str(loss))
    