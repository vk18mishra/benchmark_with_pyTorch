# Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# --WITH CHANGES

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import realtime_monitoring
import torch
import rapl
from memory_profiler import profile
import io
import pprofile
import cProfile
import pstats
import sys
import pyRAPL
import psutil
import pandas as pd
import zipfile
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import copy

# pyRAPL.setup()

#csv_output = pyRAPL.outputs.CSVOutput('energy_pyRAPL.csv')

# @pyRAPL.measure(output=csv_output)
# @profile(precision=3)
def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()
    # -------------------------------------------------------------------
    #rank = 3
    # dist.init_process_group(
    #	backend='gloo',
    #		init_method='env://',
    #	world_size=4,
    #	rank=rank
    # )
    # -------------------------------------------------------------------
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # s = realtime_monitoring.intermediate()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    plt.ion()   # interactive mode

    # ------------------------
    #os.environ['MASTER_ADDR'] = '10.57.23.164'
    #os.environ['MASTER_PORT'] = '8888'

    # create default process group
    #dist.init_process_group("gloo", rank=3, world_size=4)
    #rank = 3
    # dist.init_process_group(
    #    backend='gloo',
    #            init_method='env://',
    #    world_size=4,
    #    rank=rank
    # )
    # -----------------------

    zip = zipfile.ZipFile('hymenoptera_data.zip')
    zip.extractall()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(80)
    num_thred = torch.get_num_threads()
    num_interop_thred = torch.get_num_interop_threads()
    print("Num Threads:",num_thred," ",num_interop_thred)


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # ------------------------------------------------------------
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #	image_datasets,
    #    num_replicas=4,
    #	rank=3)
    # ------------------------------------------------------------

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)  # ,pin_memory=True,
                   # sampler=train_sampler)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    # model_ft = models.resnet18(pretrained=True)
    # torch.save(model_ft, "resnet18.pt")
    model_ft = torch.load("resnet18.pt")
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    # -------------------------------------------
    # model_ft = nn.parallel.DistributedDataParallel(model_ft,
    #                                            device_ids=None)
    # -------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    # gives a single float value
    cpu_per_b = psutil.cpu_percent()
    # gives an object with many fields
    # vir_mem_b = psutil.virtual_memory()
    # you can convert that object to a dictionary
    vir_mem_b = dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    vir_mem_per_b = psutil.virtual_memory().percent
    # you can calculate percentage of available memory
    mem_av_per_b = psutil.virtual_memory().available * 100 / \
        psutil.virtual_memory().total

    num_epochs = 1
    #torch.set_num_threads(4)
    #torch.set_num_interop_threads(80)
    #num_thred = torch.get_num_threads()
    #num_interop_thred = torch.get_num_interop_threads()
    #print("Num Threads:",num_thred," ",num_interop_thred)
    profiler = pprofile.Profile()
    pr = cProfile.Profile()
    #s1 = rapl.RAPLMonitor.sample()
    pr.enable()
    start_train = time.time()
    with profiler:
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=1)
        # mp.spawn(train_model, nprocs=4, args=(criterion,optimizer_ft,exp_lr_scheduler,
        #    num_epochs))
        # csv_output.save()
    end_train = time.time()
    pr.disable()
    #s2 = rapl.RAPLMonitor.sample()
    #diff = s2 - s1
    #print(diff)
    # Print the difference in microjoules
    #print("CPU Energy Usage(microjoules): ",diff.energy("package-0", "core", rapl.UJOULES))

    # Print the average power
    #print("Average Power(CPU Usage): ",diff.average_power("package-0", "core"))
   # for d in diff.domains:
   #     print(d)
   #     domain = diff.domains[d]
   #     power = diff.average_power(package=domain.name)
   #     print("%s - %0.2f W" % (domain.name, power))

   #     for sd in domain.subdomains:
   #         subdomain = domain.subdomains[sd]
   #         power = diff.average_power(
   #             package=domain.name, domain=subdomain.name)
   #         print("\t%s - %0.2f W" % (subdomain.name, power))
    elapsed_time = end_train-start_train
    with open('Elapsed_time.txt', 'w') as f1:
        f1.write("Training Time(Elapsed):")
        f1.write(str(elapsed_time))
    f1.close()
    # gives a single float value
    profiler.dump_stats("exec_time.txt")
    cpu_per_a = psutil.cpu_percent()
    # gives an object with many fields
    # you can convert that object to a dictionary
    vir_mem_a = dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    vir_mem_per_a = psutil.virtual_memory().percent
    # you can calculate percentage of available memory
    mem_av_per_a = psutil.virtual_memory().available * 100 / \
        psutil.virtual_memory().total
    with open('CPU_table.txt', 'w') as f:
        f.write("BEFORE TRAINING:--------\n")
        f.write("CPU USAGE(%):")
        f.write(str(cpu_per_b))
        f.write("\n")
        f.write("MEMORY USE:")
        f.write(str(vir_mem_b))
        f.write("\n")
        f.write("MEMORY USE(%):")
        f.write(str(vir_mem_per_b))
        f.write("\n")
        f.write("MEMORY AVAIL(%):")
        f.write(str(mem_av_per_b))
        f.write("\n")
        f.write("\n\n\n\n")
        f.write("AFTER TRAINING:---------\n")
        f.write("CPU USAGE(%):")
        f.write(str(cpu_per_a))
        f.write("\n")
        f.write("MEMORY USE:")
        f.write(str(vir_mem_a))
        f.write("\n")
        f.write("MEMORY USE(%):")
        f.write(str(vir_mem_per_a))
        f.write("\n")
        f.write("MEMORY AVAIL(%):")
        f.write(str(mem_av_per_a))
        f.write("\n")
    f.close()

    result = io.StringIO()
    pstats.Stats(pr, stream=result).print_stats()
    result = result.getvalue()
    # chop the string into a csv-like buffer
    result = 'ncalls' + result.split('ncalls')[-1]
    result = '\n'.join([','.join(line.rstrip().split(None, 5))
                       for line in result.split('\n')])
    # save it to disk

    with open('memory_logs.csv', 'w+') as f:
        # f=open(result.rsplit('.')[0]+'.csv','w')
        f.write(result)
        f.close()
