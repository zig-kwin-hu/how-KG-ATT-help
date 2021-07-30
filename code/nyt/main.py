
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from apex import amp
from tqdm import trange
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from models import *
from sklearn.metrics import average_precision_score
from apex.parallel import DistributedDataParallel
from scipy.special import softmax
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)


def f1_score(output, label, rel_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]
        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0 :
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)
    return micro_f1, f1_by_relation

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, devBagTest=None, testBagTest=None):
    # total step
    step_tot = len(train_dataloader) * args.max_epoch

    # optimizer
    if args.optim == "adamw":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)
    elif args.optim == "sgd":
        params = model.parameters()
        optimizer = optim.SGD(params, args.lr)
    elif args.optim == "adam":
        params = model.parameters()
        optimizer = optim.Adam(params, args.lr)

    # amp training
    if args.optim == "adamw":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Data parallel
    #model = MyDataParallel(model, device_ids=[0, 1])
    #model = DistributedDataParallel(model)
    model.train()
    model.zero_grad()

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    best_dev_score = 0
    best_test_score = 0
    best_epoch = 0
    for i in range(args.max_epoch):
        # dev
        
        for batch in train_dataloader:
            inputs = {
                "label":batch[0],
                "input_ids":batch[3],
                "mask":batch[4],
                "h_pos":batch[5],
                "t_pos":batch[6],
                'head_id':batch[7],
                'tail_id':batch[8]
            }
            if args.use_bag:
                inputs["scope"] = batch[2]
            model.training = True
            model.train()
            loss, output = model(**inputs)
            if args.optim == "adamw":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if args.optim == "adamw":
                scheduler.step()
            model.zero_grad()
            global_step += 1

            output = output.cpu().detach().numpy()
            label = batch[0].cpu().numpy()
            try:
                crr = (output == label).sum()
            except:
                print(crr)
                print(output)
                print(label)
                exit(0)
            tot = label.shape[0]

            sys.stdout.write("epoch: %d, loss: %.6f, acc: %.3f\r" % (i, loss, crr/tot))
            sys.stdout.flush()        
        
        with torch.no_grad():
            print("")
            print("deving....")
            model.training = False
            model.eval()

            if args.dataset == "semeval" or args.dataset == "tacred":
                eval_func = eval_F1
            elif args.dataset == "wiki80" or args.dataset == "chemprot":
                eval_func = eval_ACC
            elif args.dataset == "nyt" or args.dataset == "gids":
                eval_func = eval_AP
            
            score = eval_func(args, model, dev_dataloader)
            print('score',score)
            if score > best_dev_score:
                best_dev_score = score
                best_test_score = score#eval_func(args, model, test_dataloader)
                best_epoch = i
                print("Best Dev score: %.3f,\tTest score: %.3f" % (best_dev_score, best_test_score))
                temp = '_bag_kg.mdl' if args.use_bag else '_kg.mdl'
                if args.direct_feature:
                    temp = '_direct_'+temp
                if args.use_seg:
                    temp = '_seg'+temp 
                if args.ckpt_to_load != 'None':
                    torch.save(model.state_dict(), '../../save/nyt/'+args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp)
                else:
                    torch.save(model.state_dict(), '../../save/nyt/'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp)
            else:
                print("Dev score: %.3f" % score)
            print("-----------------------------------------------------------") 
    print("@RESULT: " + args.dataset +" Test score is %.3f" % best_test_score)
    f = open("../log/re_log", 'a+')
    temp = '_bag_kg.mdl' if args.use_bag else '_kg.mdl'
    if args.direct_feature:
        temp = '_direct_'+temp
    if args.use_seg:
        temp = '_seg'+temp 
    if args.ckpt_to_load == "None":
        f.write(args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'\t'+str(time.ctime())  +"\n")
    else:
        f.write(args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'\t' +str(time.ctime()) +"\n")
    f.write("@RESULT: Best Dev score is %.3f, Test score is %.3f\n, at epoch %d" % (best_dev_score, best_test_score, best_epoch))
    f.write("--------------------------------------------------------------\n")
    f.close()


def eval_AP(args, model, dataloader, return_output=False):
    tot_label = []
    tot_logits = []
    for batch in dataloader:
        inputs = {
            "label":batch[0],
            "input_ids":batch[3],
            "mask":batch[4],
            "h_pos":batch[5],
            "t_pos":batch[6],
            'head_id':batch[7],
            'tail_id':batch[8]
        }
        if args.use_bag:
            inputs["scope"] = batch[2]
        logits, output = model(**inputs)
        tot_label.extend(batch[0].cpu().detach().tolist())
        tot_logits.extend(logits.cpu().detach().tolist())
    tot_logits = np.array(tot_logits)
    tot_labels = np.zeros(tot_logits.shape)
    try:
        tot_labels[range(len(tot_labels)), tot_label] = 1
    except:
        print(tot_labels.shape,len(tot_label))
        print(logits.shape)
        exit(0)

    if not args.use_bag:
        test_scope = json.load(open('../../data/'+args.dataset+'/scope_test.json'))
        new_logits = np.zeros((len(test_scope), len(tot_labels[0])))
        new_labels = np.zeros((len(test_scope), len(tot_labels[0])))
        tot_logits = softmax(tot_logits,axis=1)
        for i in range(len(test_scope)):
            new_logits[i] = np.mean(tot_logits[test_scope[i][0]:test_scope[i][1]], axis=0)
            new_labels[i] = tot_labels[test_scope[i][0]]
        tot_logits = new_logits
        tot_labels = new_labels

    tot_labels = tot_labels.astype(np.int)
    ap =  average_precision_score(tot_labels, tot_logits, average='micro')
    ap2 =  average_precision_score(tot_labels[:,1:], tot_logits[:,1:], average='micro')
    if return_output:
        return ap2, tot_logits, tot_labels
    else:          
        return ap2
    
def eval_F1(args, model, dataloader):
    tot_label = []
    tot_output = []
    for batch in dataloader:
        inputs = {
            "label":batch[0],
            "input_ids":batch[3],
            "mask":batch[4],
            "h_pos":batch[5],
            "t_pos":batch[6],
            'head_id':batch[7],
            'tail_id':batch[8]
        }
        if args.use_bag:
            inputs["scope"] = batch[2]
        _, output = model(**inputs)
        tot_label.extend(batch[0].cpu().tolist())
        tot_output.extend(output.cpu().detach().tolist())
    f1, _ = f1_score(tot_output, tot_label, args.rel_num) 
    return f1
    

def eval_ACC(args, model, dataloader):
    tot = 0.0
    crr = 0.0
    for batch in dataloader:
        inputs = {
            "label":batch[0],
            "input_ids":batch[3],
            "mask":batch[4],
            "h_pos":batch[5],
            "t_pos":batch[6],
            'head_id':batch[7],
            'tail_id':batch[8]
        }
        if args.use_bag:
            inputs["scope"] = batch[2]
        _, output = model(**inputs)
        output = output.cpu().detach().numpy()
        label = batch[0].cpu().numpy()
        crr += (output==label).sum()
        tot += label.shape[0]

        sys.stdout.write("acc: %.3f\r" % (crr/tot)) 
        sys.stdout.flush()

    return crr / tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=0, help="batch size pre gpu")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default='tacred',help='dataset to use')
    parser.add_argument("--lr", dest="lr", type=float,
                        default=3e-5, help='learning rate')
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768,help='hidden size')
    parser.add_argument("--encoder", dest="encoder", type=str,
                        default='bert',help='encoder')
    parser.add_argument("--optim", dest="optim", type=str,
                        default='adamw',help='optimizer')
    
    
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=64, help="max sentence length")
    parser.add_argument("--ckpt_to_load", dest="ckpt_to_load", type=str,
                        default="None", help="ckpt to load")
    parser.add_argument("--entity_marker", action='store_true', 
                        help="if entity marker or cls")
    parser.add_argument("--train_prop", dest="train_prop", type=float,
                        default=1, help="train set prop")
    
    parser.add_argument("--mode", dest="mode",type=str, 
                        default="CM", help="{CM,OC,CT,OM,OT}")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    parser.add_argument("--use_bag", dest="use_bag", action='store_true',
                        default=False, help="whether train in a bag of sentence setting")
    parser.add_argument("--bag_size", dest="bag_size", type=int,
                        default=0, help="size of each bag")
    parser.add_argument("--entity_embedding_load_path", dest="entity_embedding_load_path", type=str,
                        default=None, help="where pretrained entity embedding is stored")
    parser.add_argument("--direct_feature", dest="direct_feature", action='store_true',
                        default=False, help="whether directly use kg embedding as feature")

    parser.add_argument("--kg_method", dest="kg_method", type=str,
                        default=None, help="how entity embedding is trained")
    parser.add_argument("--prefix", dest="prefix", type=str,
                        default='', help="prefix of model name")
    parser.add_argument("--freeze_entity", dest="freeze_entity", action='store_true',
                        default=False, help="whether freeze entity embedding during training")
    parser.add_argument("--test", dest="test", action='store_true',
                        default=False, help="whether test")
    parser.add_argument("--load", dest="load", action='store_true',
                        default=False, help="whether load")
    parser.add_argument("--use_seg", dest="use_seg", action='store_true',
                        default=False, help="whether use seg")
    args = parser.parse_args()

    # print args
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # Warning
    print("*"*30)
    if args.dataset == 'semeval':
        print("Warning! The results reported on `semeval` may be different from our paper. Because we use the official evaluation script. See `finetune/readme` for more details.")
    print("*"*30)

    # set seed
    set_seed(args)
        
    if not os.path.exists("../log"):
        os.mkdir("../log")
    # params for dataloader
    rel2id = json.load(open(os.path.join("../../data/"+args.dataset, "rel2id.json")))
    args.rel_num = len(rel2id)
    ent2id = json.load(open(os.path.join("../../data/"+args.dataset, "entity2id.json")))
    args.entity_num = len(ent2id)
    args.entity_embedding_size = args.hidden_size*2
    if args.use_bag:
        model = REBagModel_KG(args)
    else:
        model = REModel_KG(args)
    if args.test or args.load:
        temp = '_bag_kg.mdl' if args.use_bag else '_kg.mdl'
        if args.direct_feature:
            temp = '_direct_'+temp
        if args.use_seg:
            temp = '_seg'+temp 
        if args.ckpt_to_load != 'None':
            print('loading',args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp)
            loaded = torch.load("../../save/nyt/"+args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp)
        else:
            print('loading',args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp)
            loaded = torch.load("../../save/nyt/"+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp)
        new_loaded = {i.strip('module.'):loaded[i] for i in loaded}
        model.load_state_dict(new_loaded)
        #print('sum',torch.sum(model.entity_embedding.weight))#,torch.sum(model.transfer.weight))
        #exit(0)
    #model = nn.DataParallel(model)

    model.cuda()

    test_set = REBagDataset_KG("../../data/"+args.dataset, "test.txt", args)
    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size_per_gpu, shuffle=False, collate_fn=test_set.collate_fn)
    
    
    if args.test:
        print("")
        print("testing....")
        model.training = False
        model.eval()

        if args.dataset == "semeval" or args.dataset == "tacred":
            eval_func = eval_F1
        elif args.dataset == "wiki80" or args.dataset == "chemprot":
            eval_func = eval_ACC
        elif args.dataset == "nyt" or args.dataset == "gids":
            eval_func = eval_AP
        
        score, tot_logits, tot_labels = eval_func(args, model, test_dataloader, return_output=True)
        print('test_score', score)
        if args.ckpt_to_load != 'None':
            temp = '_bag_kg_' if args.use_bag else '_kg_'
            if args.direct_feature:
                temp = '_direct_'+temp
            if args.use_seg:
                temp = '_seg'+temp 
            np.save('../../result/nyt'+args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp+'logits.npy', tot_logits)
            np.save('../../result/nyt'+args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp+'labels.npy', tot_labels)
        else:
            temp = '_bag_kg_' if args.use_bag else '_kg_'
            if args.direct_feature:
                temp = '_direct_'+temp
            if args.use_seg:
                temp = '_seg'+temp 
            np.save('../../result/nyt'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'logits.npy', tot_logits)
            np.save('../../result/nyt'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'labels.npy', tot_labels)
        exit(0)
    #model.training = False
    #score = eval_AP(args, model, dev_dataloader)
   
    if args.train_prop == 1:
        print("Use all train data!")
        train_set = REBagDataset_KG("../../data/"+args.dataset, "train.txt", args)
    elif args.train_prop == 0.1:
        print("Use 10% train data!")
        train_set = REBagDataset_KG("../../data/"+args.dataset, "train_0.1.txt", args)
    elif args.train_prop == 0.01:
        print("Use 1% train data!")
        train_set = REBagDataset_KG("../../data/"+args.dataset, "train_0.01.txt", args)
    if args.dataset == 'nyt':
        dev_set = test_set
    else:
        dev_set = REBagDataset_KG("../../data/"+args.dataset, "dev.txt", args)
    dev_dataloader = data.DataLoader(dev_set, batch_size=args.batch_size_per_gpu, shuffle=False, collate_fn=dev_set.collate_fn)
    
    #train_set = dev_set

    train_dataloader = data.DataLoader(train_set, batch_size=args.batch_size_per_gpu, shuffle=True, collate_fn=train_set.collate_fn)        
    
    
    
    train(args, model, train_dataloader, dev_dataloader, test_dataloader)
    
    


