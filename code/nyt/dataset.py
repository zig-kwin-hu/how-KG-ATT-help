
import os 
import re
import ast 
import sys 
sys.path.append("..")
import json 
import pdb
import random 
import torch 
import numpy as np 
from torch.utils import data
sys.path.append("../../../")
from utils.utils import EntityMarker


class REDataset(data.Dataset):
    """Data loader for semeval, tacred
    """
    def __init__(self, path, name, args):
        data = []
        with open(os.path.join(path, name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                data.append(ins)
        
            
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, "rel2id.json")):
            rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        else:
            raise Exception("Error: There is no `rel2id.json` in "+ path +".")
        if os.path.exists(os.path.join(path, "type2id.json")):
            type2id = json.load(open(os.path.join(path, "type2id.json")))
        else:
            print("Warning: There is no `type2id.json` in "+ path +", If you want to train model using `OT`, `CT` settings, please firstly run `utils.py` to get `type2id.json`.")
    
        print("pre process " + name)
        # pre process data
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=int) 
        self.h_pos = np.zeros((tot_instance), dtype=int)
        self.t_pos = np.zeros((tot_instance), dtype=int)
        self.label = np.zeros((tot_instance), dtype=int)

        for i, ins in enumerate(data):
            self.label[i] = rel2id[ins["relation"]]            
            # tokenize
            if args.mode == "CM":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'])
            elif args.mode == "OC":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], None, None, True, True)
            elif args.mode == "CT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], h_type, t_type)
            elif args.mode == "OM":
                head = entityMarker.tokenizer.tokenize(ins['h']['name'])
                tail = entityMarker.tokenizer.tokenize(ins['t']['name'])
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT(head, tail, h_first)
            elif args.mode == "OT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT([h_type,], [t_type,], h_first)
            else:
                raise Exception("No such mode! Please make sure that `mode` takes the value in {CM,OC,CT,OM,OT}")

            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, args.max_length-1) 
            self.t_pos[i] = min(pt, args.max_length-1) 
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]
        label = self.label[index]

        return input_ids, mask, h_pos, t_pos, label, index
class REBagDataset(data.Dataset):
    """Data loader for semeval, tacred
    """
    def __init__(self, path, name, args):
        data = []
        with open(os.path.join(path, name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                data.append(ins)
        
            
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, "rel2id.json")):
            rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        else:
            raise Exception("Error: There is no `rel2id.json` in "+ path +".")
        if os.path.exists(os.path.join(path, "type2id.json")):
            type2id = json.load(open(os.path.join(path, "type2id.json")))
        else:
            print("Warning: There is no `type2id.json` in "+ path +", If you want to train model using `OT`, `CT` settings, please firstly run `utils.py` to get `type2id.json`.")
    
        print("pre process " + name)
        # pre process data
        self.bag_size = args.bag_size
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=int) 
        self.h_pos = np.zeros((tot_instance), dtype=int)
        self.t_pos = np.zeros((tot_instance), dtype=int)
        self.label = np.zeros((tot_instance), dtype=int)
        if 'test' in name:
            self.scope = json.load(open(os.path.join(path, "scope_test.json")))
            self.bag_name = json.load(open(os.path.join(path, "triple_test.json")))
        elif 'dev' in name:
            self.scope = json.load(open(os.path.join(path, "scope_dev.json")))
            self.bag_name = json.load(open(os.path.join(path, "triple_dev.json")))
        elif 'train' in name:
            self.scope = json.load(open(os.path.join(path, "scope_train.json")))
            self.bag_name = json.load(open(os.path.join(path, "triple_train.json")))
        else: 
            print('out of format to load data', name)
        print('scope', name, len(self.scope),self.scope[0],self.scope[-1])
        for i, ins in enumerate(data):
            self.label[i] = rel2id[ins["relation"]]            
            # tokenize
            if args.mode == "CM":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'])
            elif args.mode == "OC":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], None, None, True, True)
            elif args.mode == "CT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], h_type, t_type)
            elif args.mode == "OM":
                head = entityMarker.tokenizer.tokenize(ins['h']['name'])
                tail = entityMarker.tokenizer.tokenize(ins['t']['name'])
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT(head, tail, h_first)
            elif args.mode == "OT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT([h_type,], [t_type,], h_first)
            else:
                raise Exception("No such mode! Please make sure that `mode` takes the value in {CM,OC,CT,OM,OT}")

            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, args.max_length-1) 
            self.t_pos[i] = min(pt, args.max_length-1) 
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
    
    def __len__(self):
        return len(self.scope)

    def __getitem__(self, index):
        scope = self.scope[index]
        bag_name = self.bag_name[index]
        if self.bag_size > 0 and scope[1]-scope[0] > self.bag_size:
            count = self.bag_size
            to_select = random.sample(range(scope[0],scope[1]), self.bag_size)
            input_ids = self.input_ids[to_select]
            mask = self.mask[to_select]
            h_pos = self.h_pos[to_select]
            t_pos = self.t_pos[to_select]
        else:
            input_ids = self.input_ids[scope[0]:scope[1]]
            mask = self.mask[scope[0]:scope[1]]
            h_pos = self.h_pos[scope[0]:scope[1]]
            t_pos = self.t_pos[scope[0]:scope[1]]
            count = scope[1] - scope[0]
        try:
            label = self.label[scope[0]]
        except:
            print(scope, len(self.label), index, self.scope[index], len(self.scope))
            exit(0)
        
        return label, bag_name, count, input_ids, mask, h_pos, t_pos
    @staticmethod
    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
    
        label = np.array(label)
        input_ids = np.concatenate(data[3],axis=0)
        mask = np.concatenate(data[4],axis=0)
        h_pos = np.concatenate(data[5])
        t_pos = np.concatenate(data[6])
        
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == len(input_ids))

        #scope = torch.tensor(scope).long()
        #print(label[0],type(label))
        label = torch.tensor(label).long().cuda() # (B)
        #print(input_ids[0],type(input_ids))
        input_ids = torch.tensor(input_ids).long().cuda()
        #print(mask[0],type(mask))
        mask = torch.tensor(mask).long().cuda()
        #print(h_pos[0],type(h_pos))
        #h_pos = torch.tensor(h_pos).long().cuda()
        #print(t_pos[0],type(t_pos))
        #t_pos = torch.tensor(t_pos).long().cuda()
        return label, bag_name, scope, input_ids, mask, h_pos, t_pos
class REBagDataset_KG(data.Dataset):
    """Data loader for semeval, tacred
    """
    def __init__(self, path, name, args):
        data = []
        with open(os.path.join(path, name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                data.append(ins)
        
        self.use_bag = args.use_bag
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, "rel2id.json")):
            rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        else:
            raise Exception("Error: There is no `rel2id.json` in "+ path +".")
        if os.path.exists(os.path.join(path, "type2id.json")):
            type2id = json.load(open(os.path.join(path, "type2id.json")))
        else:
            print("Warning: There is no `type2id.json` in "+ path +", If you want to train model using `OT`, `CT` settings, please firstly run `utils.py` to get `type2id.json`.")
    
        print("pre process " + name)
        # pre process data
        self.bag_size = args.bag_size
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=int) 
        self.h_pos = np.zeros((tot_instance), dtype=int)
        self.t_pos = np.zeros((tot_instance), dtype=int)
        self.label = np.zeros((tot_instance), dtype=int)
        self.head_id = np.zeros((tot_instance), dtype=int)
        self.tail_id = np.zeros((tot_instance), dtype=int)
        self.entity2id = json.load(open(os.path.join(path, "entity2id.json")))
        if 'test' in name:
            self.scope = json.load(open(os.path.join(path, "scope_test.json")))
            self.bag_name = json.load(open(os.path.join(path, "triple_test.json")))
        elif 'dev' in name:
            self.scope = json.load(open(os.path.join(path, "scope_dev.json")))
            self.bag_name = json.load(open(os.path.join(path, "triple_dev.json")))
        elif 'train' in name:
            self.scope = json.load(open(os.path.join(path, "scope_train.json")))
            self.bag_name = json.load(open(os.path.join(path, "triple_train.json")))
        else: 
            print('out of format to load data', name)
        print('scope', name, len(self.scope),self.scope[0],self.scope[-1])
        for i, ins in enumerate(data):
            self.label[i] = rel2id[ins["relation"]]            
            # tokenize
            if args.mode == "CM":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'])
            elif args.mode == "OC":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], None, None, True, True)
            elif args.mode == "CT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], h_type, t_type)
            elif args.mode == "OM":
                head = entityMarker.tokenizer.tokenize(ins['h']['name'])
                tail = entityMarker.tokenizer.tokenize(ins['t']['name'])
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT(head, tail, h_first)
            elif args.mode == "OT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT([h_type,], [t_type,], h_first)
            else:
                raise Exception("No such mode! Please make sure that `mode` takes the value in {CM,OC,CT,OM,OT}")
            self.head_id[i] = self.entity2id[data[i]['h']['id']]
            self.tail_id[i] = self.entity2id[data[i]['t']['id']]

            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, args.max_length-1) 
            self.t_pos[i] = min(pt, args.max_length-1) 
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
    
    def __len__(self):
        if self.use_bag:
            return len(self.scope)
        else:
            return len(self.input_ids)

    def __getitem__(self, index):
        if self.use_bag:
            scope = self.scope[index]
            bag_name = self.bag_name[index]
            if self.bag_size > 0 and scope[1]-scope[0] > self.bag_size:
                count = self.bag_size
                to_select = random.sample(range(scope[0],scope[1]), self.bag_size)
            else:
                count = scope[1] - scope[0]
                to_select = range(scope[0],scope[1])
            input_ids = self.input_ids[to_select]
            mask = self.mask[to_select]
            h_pos = self.h_pos[to_select]
            t_pos = self.t_pos[to_select]
            head_id = self.head_id[to_select]
            tail_id = self.tail_id[to_select]
            try:
                label = self.label[scope[0]]
            except:
                print(scope, len(self.label), index, self.scope[index], len(self.scope))
                exit(0)
            
            return label, bag_name, count, input_ids, mask, h_pos, t_pos, head_id, tail_id
        else:
            label = self.label[index]
            input_ids = self.input_ids[index]
            mask = self.mask[index]
            h_pos = self.h_pos[index]
            t_pos = self.t_pos[index]
            head_id = self.head_id[index]
            tail_id = self.tail_id[index]
            return label, None, None, input_ids, mask, h_pos, t_pos, head_id, tail_id
    @staticmethod
    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
    
        label = np.array(label)
        
        if count[0]:
            input_ids = np.concatenate(data[3],axis=0)
            mask = np.concatenate(data[4],axis=0)
            h_pos = np.concatenate(data[5])
            t_pos = np.concatenate(data[6])
            head_id = np.concatenate(data[7])
            tail_id = np.concatenate(data[8])
            scope = [] # (B, 2)
            start = 0
            for c in count:
                scope.append([start, start + c])
                start += c
            assert(start == len(input_ids))
            scope = torch.tensor(scope).long().cuda()
        else:
            input_ids = np.array(data[3])
            mask = np.array(data[4])
            h_pos = np.array(data[5])
            t_pos = np.array(data[6])
            head_id = np.array(data[7])
            tail_id = np.array(data[8])
            scope = None
        #scope = torch.tensor(scope).long()
        #print(label[0],type(label))
        label = torch.tensor(label).long().cuda() # (B)
        #print(input_ids[0],type(input_ids))
        input_ids = torch.tensor(input_ids).long().cuda()
        #print(mask[0],type(mask))
        mask = torch.tensor(mask).long().cuda()
        #print(h_pos[0],type(h_pos))
        h_pos = torch.tensor(h_pos).long().cuda()
        #print(t_pos[0],type(t_pos))
        t_pos = torch.tensor(t_pos).long().cuda()
        head_id = torch.tensor(head_id).long().cuda()
        tail_id = torch.tensor(tail_id).long().cuda()
        #print('collate', len(label), len(bag_name), len(scope), len(input_ids), len(mask))
        return label, bag_name, scope, input_ids, mask, h_pos, t_pos, head_id, tail_id
