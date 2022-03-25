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
    writer = SummaryWriter('runs/exp') # 记录训练过程
    for epoch in range(n_epoches):
        train_l_sum,m = 0.0, 0
        f_all,p_all,r_all = 0.0,0.0,0.0
        for i,data in enumerate(train_iter): # 按batch迭代
            feature,true_labels=data[0],data[1]
            feature = feature.to(device) # 将特征转入设备
            true_labels = true_labels.to(device) # 将标签转入设备
            predict = net(feature) # [batch_size,3]

            Loss = criterion(predict,true_labels.long()) # 交叉熵损失

            optimizor.zero_grad() # 梯度清零
            Loss.backward() # 反向传播
            optimizor.step() # 梯度更新

            train_l_sum += Loss.item() # 将tensor转化成int类型

            f1,p,r = compute(predict,true_labels) # 计算查全率、查准率、F1值
            f_all+=f1
            p_all+=p
            r_all+=r

            m += 1
            l = train_l_sum / m

            if m%2000==0: # 打印相关结果
                print('epoch %d [%d/%d], loss %.4f, train recall %.3f, train precision %.3f, train F1 %.3f'% (epoch + 1, m,len(train_iter),l,r_all/m,p_all/m,f_all/m ))

        if epoch % 3 == 0:
            f1,p,r = evaluate(device,val_iter,net) # 进行验证集下的验证
            writer.add_scalar('F1-score_val/epoch', f1, global_step=epoch, walltime=None)
            print("------------------------------------------------------------------")
            print("validation  ->  val_recall=%.3f val_precision=%.3f val_F1=%.3f"% (p,r,f1))
            print("------------------------------------------------------------------")
            torch.save(net.state_dict(), 'results/'+f'{epoch}_softmax_net.pkl') # 保存模型
        
        writer.add_scalar('loss/epoch', l, global_step=epoch, walltime=None)
        writer.add_scalar('recall/epoch', r_all/m, global_step=epoch, walltime=None)
        writer.add_scalar('precision/epoch',p_all/m, global_step=epoch, walltime=None)
        writer.add_scalar('F1-score/epoch', f_all/m, global_step=epoch, walltime=None)

def evaluate(device,val_iter,net):
    '''
    验证函数
    '''
    net.eval() # 选择调试模型
    f_all,p_all,r_all,n = 0.0,0.0,0.0,0
    for i,data in enumerate(val_iter): # 按batch迭代
        feature,true_labels=data[0],data[1]
        feature = feature.to(device) # 将特征转入设备
        true_labels = true_labels.to(device) # 将标签转入设备
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
    计算查准率、查全率与F1值\n
    方法来自sklearn（三分类下的）
    '''
    y = tensor1.argmax(dim=1)
    y_pred = y.tolist() # 首先转换成列表
    y_true = tensor2.tolist() # 首先转换成列表
    f1 = f1_score( y_true, y_pred, average='macro' )
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')

    return f1,p,r
