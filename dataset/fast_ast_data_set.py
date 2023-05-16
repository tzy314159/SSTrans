import torch
from torch_geometric.data import Data
from tqdm import tqdm
import random
from dataset import BaseASTDataSet
import scipy.sparse as sp
import numpy as np
__all__ = ['FastASTDataSet']

def convert_dict_to_tree(dict1,dim_matrix=200): 
    node_max= [[0 for _ in range(dim_matrix)] for _ in range(dim_matrix)]
    for node_set in dict1.keys():

        if node_set[0] or node_set[1]>=dim_matrix:
            continue
        node_max[node_set[0]][node_set[1]] = dict1[node_set]
        node_max[node_set[1]][node_set[0]] = dict1[node_set]

    node_max = torch.tensor(node_max, dtype=torch.float) 
    return node_max
    



class FastASTDataSet(BaseASTDataSet):
    def __init__(self, config, data_set_name):
        print('Data Set Name : < Fast AST Data Set >')
        super(FastASTDataSet, self).__init__(config, data_set_name)
        self.max_par_rel_pos = config.max_par_rel_pos
        self.max_bro_rel_pos = config.max_bro_rel_pos

        self.edges_data = self.convert_ast_to_edges()

        
    def convert_ast_to_edges(self):
        print('building edges.')

        par_edge_data = self.matrices_data['parent']
        bro_edge_data = self.matrices_data['brother']

        edges_data = []

        def edge2list(edges, edge_type):
            if edge_type == 'par':
                max_rel_pos = self.max_par_rel_pos
            if edge_type == 'bro':
                max_rel_pos = self.max_bro_rel_pos
            ast_len = min(len(edges), self.max_src_len)
            start_node = -1 * torch.ones((self.max_rel_pos + 1, self.max_src_len), dtype=torch.long)
            for key in edges.keys(): 
                if key[0] < self.max_src_len and key[1] < self.max_src_len:
                    value = edges.get(key)
                    if value > max_rel_pos and self.ignore_more_than_k:
                        continue
                    value = min(value, max_rel_pos)
                    start_node[value][key[1]] = key[0]

            start_node[0][:ast_len] = torch.arange(ast_len)
            return start_node

        for i in tqdm(range(self.data_set_len)):
            par_edges = par_edge_data[i]
            bro_edges = bro_edge_data[i]
            ast_seq = self.ast_data[i]
            nl = self.nl_data[i]

            priority_seq = self.priority_seq[i]
            node_max=convert_dict_to_tree(par_edges)



            par_edge_list = edge2list(par_edges, 'par')
            bro_edge_list = edge2list(bro_edges, 'bro')

            ast_vec = self.convert_ast_to_tensor(ast_seq)
            nl_vec = self.convert_nl_to_tensor(nl)
            priority_seq = self.convert_code_to_tensor(priority_seq)
            ast_vec_0 = ast_vec.clone().detach()
            code_seq = self.code_seqence[i]
            code_1 = ""
            for i in code_seq:
                code_1 += i + " "
            code_seq=code_1

            nl_1 = ""
            for i in nl:
                nl_1 += i+" "
            nl = nl_1
            data = Data(nl = nl,
                        code_seq = code_seq,
                        src_seq_0 = ast_vec_0,
                        src_seq=ast_vec,
                        par_edges=par_edge_list,
                        bro_edges=bro_edge_list,
                        tgt_seq=nl_vec[:-1],
                        target=nl_vec[1:],
                        priority_seq=priority_seq,
                        node_max = node_max
                        )
            edges_data.append(data)
        print("Done!")


        return edges_data

    def __getitem__(self, index):
        # print("__getitem"*5)
        # print(self.edges_data[index])
        return self.edges_data[index], self.edges_data[index].target
