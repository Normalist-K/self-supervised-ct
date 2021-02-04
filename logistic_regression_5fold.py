import time
import os
import argparse
import random
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils import yaml_config_hook

from modules import SimCLR, LogisticRegression, get_resnet, EarlyStopping, KFoldSplit
from modules.transformations import TransformsSimCLR



def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, valid_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    valid_X, valid_y = inference(valid_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    valid = torch.utils.data.TensorDataset(
        torch.from_numpy(X_valid), torch.from_numpy(y_valid)
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False
    )
    
    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, valid_loader, test_loader


def train_model(args, model, train_loader, valid_loader, criterion, optimizer, patience):
    epochs = args.logistic_epochs
    device = args.device
    valid_loss_min = np.Inf
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    train_acc, valid_acc = [],[]
    #valid_acc =[]
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(epochs):

        start = time.time()
        
        model.train()
        total_train = 0
        correct_train = 0
        total_valid = 0
        correct_valid = 0
        
        for step, (inputs, labels) in enumerate(train_loader):
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            predicted = outputs.argmax(1)
            total_train += labels.nelement()
            correct_train += (predicted == labels).sum().item()
            train_accuracy = correct_train / total_train
            model.eval()
            
        with torch.no_grad():
            accuracy = 0
            for inputs, labels in valid_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())
                # Calculate accuracy
                predicted = outputs.argmax(1)
                total_valid += labels.nelement()
                correct_valid += (predicted == labels).sum().item()
                valid_accuracy = correct_valid / total_valid
            
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        valid_acc.append(valid_accuracy) 
        train_acc.append(train_accuracy)

        # calculate average losses
        
        # print training/validation statistics 
        print(f"Epoch {epoch+1}/{epochs}.. ")
        #print('train Loss: {:.3f}'.format(epoch, loss.item()), "Training Accuracy: %d %%" % (train_accuracy))
        #print('Training Accuracy: {:.6f}'.format(
        #    train_accuracy))
        print('Training Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))
        train_losses = []
        valid_losses = []        
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
        early_stopping(valid_loss, args, model, optimizer, save=False)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    print('Best val Acc: {:4f}'.format(best_acc*100))  
    # model.load_state_dict(torch.load('checkpoint.pt'))
    # plt.title("Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Accuracy")      
    # plt.plot(train_acc, label='Training acc')
    # plt.plot(valid_acc, label='Validation acc')
    # plt.legend(frameon=False)
    # plt.show()
    return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc
    
# test(args, arr_test_loader, model, criterion, optimizer)
def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    pred = []
    true = []
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        outputs = model(x)
        loss = criterion(outputs, y)

        predicted = outputs.argmax(1)
        preds = predicted.cpu().numpy()
        labels = y.cpu().numpy()
        preds = np.reshape(preds, (len(preds), 1))
        labels = np.reshape(labels, (len(preds), 1))

        for i in range(len(preds)):
            pred.append(preds[i][0].item())
            true.append(labels[i][0].item())
        
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    cnf_matrix = confusion_matrix(true, pred)
    print('Confusion Matrix:\n', cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    accuracy_epoch = np.diag(cnf_matrix).sum().item() / len(true)
    
    # Specificity or true negative rate
    specificity = TN/(TN+FP) 

    print_specificity(specificity)

    report = classification_report(true, pred, target_names=['covid', 'healthy', 'others'])
    print(report)

    return loss_epoch, accuracy_epoch, (pred, true)

def print_specificity(specificity):
    print('\t\tspecificity')
    print('')

    print(f'       covid\t{specificity[0]:.2f}')
    print(f'     healthy\t{specificity[1]:.2f}')
    print(f'      others\t{specificity[2]:.2f}')
    print('')

    macro_specificity = sum(specificity) / 3.0
    print(f'   macro avg\t{macro_specificity:.2f}')

    weighted = [434/835, 152/835, 249/835] 
    weighted_specificity = weighted @ specificity
    print(f'weighted avg\t{weighted_specificity:.2f}')
    print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #------- added by young ---------
    torch.cuda.manual_seed(args.seed)
    if args.gpus > 1:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(args, encoder, n_features)
    model_fp = os.path.join(
        args.model_path, "model{}.tar".format(args.model_num)
    )
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()
    
    n_classes = 3 # covid, healthy, other
    patience = 20
    
    # 5-fold cross validation
    merge_data = torchvision.datasets.ImageFolder(
        '/home/opticho/source/SimCLR/datasets/dataset2(1)/train', 
        transform=TransformsSimCLR(size=(args.image_size, args.image_size)).test_transform)
    test_dataset = torchvision.datasets.ImageFolder(
        '/home/opticho/source/SimCLR/datasets/dataset2(1)/test', 
        transform=TransformsSimCLR(size=(args.image_size, args.image_size)).test_transform)
    
    k_fold_split = KFoldSplit(5, merge_data, random_state = args.seed)
    test_loss, test_accuracy = [], []
    models = []
    
    for fold, (train_dataset, valid_dataset) in enumerate(k_fold_split):
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.logistic_batch_size,
                                                shuffle=True, drop_last=False, num_workers=args.workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.logistic_batch_size,
                                                shuffle=True, drop_last=False, num_workers=args.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.logistic_batch_size,
                                                shuffle=False, drop_last=False, num_workers=args.workers)

        ## Logistic Regression
        model = LogisticRegression(simclr_model.n_features, n_classes)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        print("### Creating features from pre-trained context model ###")
        (train_X, train_y, valid_X, valid_y, test_X, test_y) = get_features(
            simclr_model, train_loader, valid_loader, test_loader, args.device
        )

        arr_train_loader, arr_valid_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, valid_X, valid_y, test_X, test_y, args.logistic_batch_size
        )
        
        # train model
        model, train_loss, valid_loss, train_acc, valid_acc = train_model(
            args, model, 
            arr_train_loader, arr_valid_loader, 
            criterion, optimizer, patience)
        
        models.append(model)
        
        # save downstream model 
        if fold == 0:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)

            out = os.path.join(args.model_path, f"downstream_{args.model_num}_kfold.tar")

            torch.save(model.state_dict(), out)
        
        # final testing
        loss_epoch, accuracy_epoch, result = test(
            args, arr_test_loader, model, criterion, optimizer
        )
        print(
            f"[test {fold+1} fold]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch}"
        )
        test_loss.append(loss_epoch/len(arr_test_loader))
        test_accuracy.append(accuracy_epoch)
    
    print(f"[FINAL]\t Loss: {sum(test_loss) / len(test_loss)}\t Accuracy: {sum(test_accuracy) / len(test_accuracy)}")

    # compare pred-true label by imabes
    if False:
        import os
        import shutil

        pred, true = result
        images_path = test_loader.dataset.samples
        # images_path -> [ [images path, label] * 835 ]

        wrong = []
        for idx, (trg, label) in enumerate(images_path):
            if label != true[idx]:
                print('something wrong...')
            if pred[idx] != true[idx]:
                wrong.append((trg, label, pred[idx]))           

        def cmkdir(d):
            if not os.path.isdir(d):
                os.mkdir(d)
                print(f'make directory ...{d}')

        root = os.path.join(os.path.expanduser('~'), 'source/SimCLR/datasets/result')

        dir = os.path.join(root, f'model{args.model_num}')
        cmkdir(dir)

        for src, label, pred in wrong:
            image_name = os.path.basename(src)[:-4]
            pred_full = ['Covid', 'Healthy', 'Others']    
            trg = os.path.join(dir, f'{image_name}-{pred_full[pred]}.png')
            if not os.path.isfile(trg):
                shutil.copy(src, trg)