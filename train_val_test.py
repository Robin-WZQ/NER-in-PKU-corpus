import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter


def train(device,train_iter,optimizor,criterion,n_epoches,batch_size,net,val_iter):
    '''
    训练函数
    '''
    print("training on -> {}".format(device))
    writer = SummaryWriter('runs/exp')
    for epoch in range(n_epoches):
        train_l_sum,m = 0.0, 0
        f_all,p_all,r_all = 0.0,0.0,0.0
        for i,data in enumerate(train_iter):
            feature,true_labels=data[0],data[1]
            feature = feature.to(device)
            true_labels = true_labels.to(device)
            predict = net(feature) # [batch_size,3]

            Loss = criterion(predict,true_labels.long())

            optimizor.zero_grad()
            Loss.backward()
            optimizor.step()

            train_l_sum += Loss.item()

            f1,p,r = compute(predict,true_labels)
            f_all+=f1
            p_all+=p
            r_all+=r

            m += 1
            l = train_l_sum / m

            if m%2000==0:
                print('epoch %d [%d/%d], loss %.4f, train recall %.3f, train precision %.3f, train F1 %.3f'% (epoch + 1, m,len(train_iter),l,r_all/m,p_all/m,f_all/m ))

        if epoch % 3 == 0:
            f1,p,r = evaluate(device,val_iter,net)
            writer.add_scalar('F1-score_val/epoch', f1, global_step=epoch, walltime=None)
            print("------------------------------------------------------------------")
            print("validation  ->  val_recall=%.3f val_precision=%.3f val_F1=%.3f"% (p,r,f1))
            print("------------------------------------------------------------------")
            torch.save(net.state_dict(), 'results/'+f'{epoch}_softmax_net.pkl')
        
        writer.add_scalar('loss/epoch', l, global_step=epoch, walltime=None)
        writer.add_scalar('recall/epoch', r_all/m, global_step=epoch, walltime=None)
        writer.add_scalar('precision/epoch',p_all/m, global_step=epoch, walltime=None)
        writer.add_scalar('F1-score/epoch', f_all/m, global_step=epoch, walltime=None)

def evaluate(device,val_iter,net):
    '''
    验证函数
    '''
    net.eval() 
    f_all,p_all,r_all,n = 0.0,0.0,0.0,0
    for i,data in enumerate(val_iter):
        feature,true_labels=data[0],data[1]
        feature = feature.to(device)
        true_labels = true_labels.to(device)
        predict = net(feature) # [batch_size,3]
        f1,p,r = compute(predict,true_labels)
        f_all+=f1
        p_all+=p
        r_all+=r
        n+=1

    return f_all/n,p_all/n,r_all/n

def testall(device,test_iter,net):
    '''
    测试函数
    '''
    net.eval()
    f_all,p_all,r_all,n = 0.0,0.0,0.0,0
    for i,data in enumerate(test_iter):
        feature,true_labels=data[0],data[1]
        feature = feature.to(device)
        true_labels = true_labels.to(device)
        predict = net(feature) # [batch_size,3]
        f1,p,r = compute(predict,true_labels)
        f_all+=f1
        p_all+=p
        r_all+=r
        n+=1

    print("测试结果：")
    print('test recall %.3f, test precision %.3f, test F1 %.3f'% (r_all/n,p_all/n,f_all/n))

def compute(tensor1,tensor2):
    '''
    计算查准率、查全率与F1值
    '''
    y = tensor1.argmax(dim=1)
    y_pred = y.tolist()
    y_true = tensor2.tolist()
    f1 = f1_score( y_true, y_pred, average='macro' )
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')

    return f1,p,r
