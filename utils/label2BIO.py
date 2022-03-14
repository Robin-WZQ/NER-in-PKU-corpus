'''
@ file function: 将label转化为BIO文件
@ author: 王中琦
@ date: 2022/3/12
''' 

def name_label(file_name):
    '''
    对名字进行标注\n
    列举一些例子:
        张/nrf  仁伟/nrg  -> 张/B 仁伟/I \n
        陈/nrf  方/nrf  安生/nrg -> 陈/B  方/I  安生/I\n
        张/nrf  教授/n -> 张/B  教授/O

    '''
    with open(file_name,'r',encoding='utf-8') as inp,open('name_test.txt','w',encoding='utf-8') as outp:
        for line in inp.readlines(): #读入每行文字
            line = line.replace("[",'')
            line = line.replace("]",'')
            line = line.split('  ') #以两个空格分割
            i = 1
            if line[0]=='\n':
                continue
            if 19980126<=eval((line[0].split("-"))[0])<=19980131:
                line.pop() #去除最后转行符
                while i<len(line):
                    if line[i].split('/')[1]=='nrf': #判断是否为人名
                        word = line[i].split('/')[0]
                        i+=1
                        outp.write(word+'/B ') #将姓写成B
                        if i<len(line) and line[i].split('/')[1]=='nrg': #如果是nrg（名）则标记为I
                            outp.write(line[i].split('/')[0]+'/I ')
                        elif i<len(line) and line[i].split('/')[1]=='nrf': #如果是nrf（复姓）则标记为I 
                            outp.write(line[i].split('/')[0]+'/I ')
                            i+=1
                            if i<len(line) and line[i].split('/')[1]=='nrg':
                                outp.write(line[i].split('/')[0]+'/I ')
                        elif i<len(line) and line[i].split('/')[1]=='n': #如果直接是n则无关，标记为O
                            outp.write(line[i].split('/')[0]+'/O ')
                    elif line[i].split('/')[1]=='nrg': #同理
                        word = line[i].split('/')[0]
                        i+=1
                        outp.write(word+'/B ')
                        if i<len(line) and line[i].split('/')[1]=='n':
                            outp.write(line[i].split('/')[0]+'/O ')
                    elif line[i].split('/')[1]=='nr': #同理
                        word = line[i].split('/')[0]
                        outp.write(word+'/B ')
                    else:
                        word = line[i].split('/')[0] # 如果不是上述，则直接标记O
                        outp.write(word+'/O ')
                    i+=1
                outp.write('\n')

def place_label(file_name):
    '''
    对地名进行标注\n
    列举一些例子:
        四川省/ns  天津市/ns  -> 四川省/B  天津市/B\n
        [广西/ns  环江/ns  毛南族/nz  自治县/n]ns  -> 广西/B  环江/I  毛南族/I  自治县/I

    难点在于对"[]"的处理
    '''
    with open(file_name,'r',encoding='utf-8') as inp,open('place_test.txt','w',encoding='utf-8') as outp:
        for line in inp.readlines(): #读入每行文字
            line = line.split('  ') #以两个空格分割
            i = 1
            if line[0]=='\n':
                continue
            if 19980126<=eval((line[0].split("-"))[0])<=19980131:
                line.pop() #去除最后转行符
                while i<len(line):
                    if line[i][0]=='[': #如果是组合词需要特别处理
                        line[i] = line[i].replace("[","")
                        j=i
                        flag = 1 #指示变量，判断是机构还是地名的方括号
                        while j<len(line):
                            if line[j][-3]==']':
                                if line[j][-2]=='n' and line[j][-1]=="s": # 是地名
                                    outp.write(line[i].split("/")[0]+'/B ') 
                                    flag = 0 
                                break
                            j+=1 # 否则记录中间经过多少个
                        if flag==0: # 如果是,将除第一个外标为I
                            for k in range(j-i-1):
                                outp.write(line[i+k+1].split("/")[0]+'/I ')
                            final = line[j].split("]")[0]
                            outp.write(final.split("/")[0]+'/I ')
                            i=j
                        else: # 如果不是地名，所有用O表示
                            outp.write(line[i].split("/")[0]+'/O ')
                            for k in range(j-i-2):
                                outp.write(line[i+k+1].split("/")[0]+'/O ')
                            final = line[j-1].split("]")[0]
                            outp.write(final.split("/")[0]+'/O ')
                            i=j-1
                    elif line[i].split('/')[1]=='ns': # 直接标B
                        word = line[i].split('/')[0]
                        outp.write(word+'/B ')
                    else: #直接标O
                        word = line[i].split('/')[0]
                        outp.write(word+'/O ')
                    i+=1
                outp.write('\n')

def org_label(file_name):
    '''
    对机构名进行标注
    同理
    '''
    with open(file_name,'r',encoding='utf-8') as inp,open('orgnization_test.txt','w',encoding='utf-8') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            if line[0]=='\n':
                continue
            if 19980126<=eval((line[0].split("-"))[0])<=19980131:
                line.pop()
                while i<len(line):
                    if line[i][0]=='[':
                        line[i] = line[i].replace("[","")
                        j=i
                        flag = 1
                        while j<len(line):
                            if line[j][-3]==']':
                                if line[j][-2]=='n' and line[j][-1]=="t": # 是地名则跳出循环
                                    outp.write(line[i].split("/")[0]+'/B ') 
                                    flag = 0 
                                break
                            j+=1
                        if flag==0:
                            for k in range(j-i-1):
                                outp.write(line[i+k+1].split("/")[0]+'/I ')
                            final = line[j].split("]")[0]
                            outp.write(final.split("/")[0]+'/I ')
                            i=j
                        else:
                            outp.write(line[i].split("/")[0]+'/O ')
                            for k in range(j-i-2):
                                outp.write(line[i+k+1].split("/")[0]+'/O ')
                            final = line[j-1].split("]")[0]
                            outp.write(final.split("/")[0]+'/O ')
                            i=j-1
                    elif line[i].split('/')[1]=='nt':
                        word = line[i].split('/')[0]
                        outp.write(word+'/B ')
                    else:
                        word = line[i].split('/')[0]
                        outp.write(word+'/O ')
                    i+=1
                outp.write('\n')

if __name__=="__main__":
    file_name = "input.txt"
    name_label(file_name)
    place_label(file_name)
    org_label(file_name)
    # data = openreadtxt('input.txt')
    # print(data[0])
