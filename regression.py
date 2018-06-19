import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import regression_vgg as vgg
import numpy as np

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

###############################################################################
"""arg parse for command prompt"""
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg16_bn)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=8e-3, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp.txt', type=str)
parser.add_argument('--save-flag', dest='save_flag',
                    help='Flag to decide to save the trained models, because they are a lot of memory',
                    default=False, type=bool)
parser.add_argument('--model-dir', dest='model_dir',
                    help='Directory to save model',
                    default='save_temp', type=str)
parser.add_argument('--gpu-flag', dest='gpu_on',
                    help='Flag to turn on and off gpu control',
                    default=False, type=bool) 
###############################################################################
best_prec1 = 0
def main():
    #start up jargon for parser
    global args, best_prec1
    args = parser.parse_args()
    
    #text file for logging (smaller than saving model params)
    with open(args.save_dir, "w") as text_file: 
        text_file.write(str(args))

    # Check the save_dir exists or not
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model = vgg.__dict__[args.arch]()

    model.features = torch.nn.DataParallel(model.features) #turn on parallel gpu
    if(args.gpu_on): #print at the begining of a run to confirm setting
        print(torch.cuda.is_available())
        print(args)
        model.cuda()
    #cudnn.benchmark = True #optimizes for single test, so it is bad for hyper param optimization
    
###############################################################################    
    """load data sets"""
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])         
    uo2_dataset = datasets.ImageFolder(root='sample/256_dataset/train',
                                           transform=data_transform)
    train_loader = torch.utils.data.DataLoader(uo2_dataset,
                                             batch_size= args.batch_size,
                                             num_workers=args.workers, shuffle = True, pin_memory=False)
    uo2_evalset = datasets.ImageFolder(root='sample/256_dataset/val',
                                           transform=data_transform)
    val_loader = torch.utils.data.DataLoader(uo2_evalset,
                                             batch_size = args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers, pin_memory=False)

###############################################################################    
    """ define loss function (criterion) and optimizer then run train and eval for epochs"""
    if(args.gpu_on):
        criterion = nn.MSELoss(size_average=False).cuda()
    else:
        criterion = nn.MSELoss(size_average=False)  
        
    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch) #dataset, neural net, thing to define loss, thing to optimize loss, iterate over epoch

        # evaluate on validation set
        #accuracy_final(val_loader,model)
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
###############################################################################
        #for saving param weights   
        if(args.save_flag):
           save_checkpoint({
               'epoch': epoch + 1,
               'state_dict': model.state_dict(),
               'best_prec1': best_prec1,
           }, is_best, filename=os.path.join(args.model_dir, 'checkpoint_{}.tar'.format(epoch)))

###############################################################################    
    #my final accuracy measure over all of the eval set    
    #accuracy_final(val_loader,model) 


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end) # measure data loading time
        if(args.gpu_on): #switch to use gpu or cpu
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input) .cuda()
        else:
            input_var = torch.autograd.Variable(input)
        #target =  targetClassToTemp(target)   
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # heavy lifting
        output = model(input_var)
        #output = torch.sum(output,1) #sum fc layers
        #output = output.view(-1,1)
        target_var = target_var.float()
        target_var = target_var.view(-1,1)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #first print to command line then write to text file
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            with open(args.save_dir, "a") as text_file:
                text_file.write('\nEpoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1))
            


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if(args.gpu_on):
                target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input) .cuda()
            else:
                input_var = torch.autograd.Variable(input)

            #target =  targetClassToTemp(target)
            target_var = torch.autograd.Variable(target)
            
            if args.half:
                input_var = input_var.half()
    
            #heavy lifting
            output = model(input_var)
            #output = torch.sum(output,1) #sum fc layers
            #output = output.view(-1,1)
            
            target_var = target_var.float()
            target_var = target_var.view(-1,1)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
    
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
                with open(args.save_dir, "a") as text_file:
                    text_file.write('\n \n Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n \n'.format(
                              i, len(val_loader), batch_time=batch_time, loss=losses,
                              top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 10)) #was 30
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) #takes the max score i.e. most likely class
    pred = pred.t() #convert to tensor
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size)) #multiply by 100 divide by batchsize
    return res

def accuracy_final(val_loader, model)  :
    """
    more comprehensive class giving total accuracy for every class 
    as well as statistics of t-p f-p t-n f-n. Useful for overall accuracy + class imbalance
    """    
    with torch.no_grad():
        classLen =  4#tp,tn,fp,fn,total * 4 classes
        tp = np.zeros((classLen,1))
        tn = np.zeros((classLen,1))
        fp = np.zeros((classLen,1))
        fn = np.zeros((classLen,1))
        total =  np.zeros((classLen,1))
        count = 0
        #calculate accuracy
        for i, (input, labels) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input)
            outputs = model(input_var)
            _, predicted = torch.max(outputs, 1)
            #total
            for j in range(0,labels.size()[0]):
                total[labels[j]] += 1
                if(labels[j].item() == predicted[j].item()): #tp
                    tp[labels[j]] += 1
                else: #tn ^ fn
                    tn[predicted[j]] += 1
                    fn[labels[j]] += 1             
                #print(count)
                count+=1
        #print them to command prompt       
        for i in range(0,classLen):
            print("class: ", i)
            print("accuracy: ", tp[i] / total[i])
            print("false negatives: " , tn[i] / total[i])       
        print("final accuracy: ", (np.sum(tp[:])/np.sum(total[:]))   )
        #write them to text file
        with open(args.save_dir, "a") as text_file:
            x = np.sum(tp[:]) 
            y = np.sum(total[:])
            z = x/y
            text_file.write('\n \n Accuracy of Class 1 {}\t' 
                            'Accuracy of Class 2 {}\t'
                            'Total accuracy of Class{}\t'.format((tp[0] / total[0]),(tp[1] / total[1]),z)
                            )                                                          

def targetClassToTemp(target):
    """ 
    0 1 2 3 4 class scores to
    600 - 800 kelvin
    """
    target = target.numpy()
    for i in range(0,target.size):
        target[i] = target[i] * 50 + 600
        #print(target[i])
    target = torch.from_numpy(target)
    return target

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

if __name__ == '__main__':
    main()
    



  
    
