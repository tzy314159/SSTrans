import random
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch
import numpy as np
import torch.utils.data as data
import re
import wordninja
import string
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater
from tqdm import tqdm
from utils import PAD, UNK
from torch.utils.data.dataset import T_co
punc = string.punctuation

__all__ = ['BaseASTDataSet', 'clean_nl', 'subsequent_mask', 'make_std_mask']


class BaseASTDataSet(data.Dataset):
    def __init__(self, config, data_set_name):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = config.data_dir + '/' + data_set_name + '/'

        self.ignore_more_than_k = config.is_ignore
        self.max_rel_pos = config.max_rel_pos
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len

        ast_path = data_dir + 'split_pot.seq' if config.is_split else 'un_split_pot.seq'
        matrices_path = data_dir + 'split_matrices.npz' if config.is_split else 'un_split_matrices.npz'

        self.ast_data = load_list(ast_path)
        self.nl_data = load_seq(data_dir + 'nl.original')
        self.matrices_data = load_matrices(matrices_path)

        self.data_set_len = len(self.ast_data)
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab
        self.collector = Collater([], [])



        code_path = data_dir + 'code.seq'
        code_seqence = load_code(code_path)


        self.code_seqence = code_seqence.copy()
        self.priority_seq=[]
        for i,seq in enumerate(code_seqence) :
            code_seqence[i]=get_priority_seq(seq)
        
        self.priority_seq = code_seqence


    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq):
        ast_seq = ast_seq[:self.max_src_len]
        return word2tensor(ast_seq, self.max_src_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[:self.max_tgt_len - 2]
        nl = ['<s>'] + nl + ['</s>']
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)
    
    def convert_code_to_tensor(self, code_seq):
        if len(code_seq)>self.max_src_len:
            code_seq = code_seq[:self.max_src_len]
        else:
            for _ in range(self.max_src_len-len(code_seq)):
                code_seq.append(0)
        return torch.tensor(code_seq,dtype=torch.long)





def word2tensor(seq, max_seq_len, vocab):
    seq_vec = [vocab.w2i[x] if x in vocab.w2i else UNK for x in seq]
    seq_vec = seq_vec + [PAD for i in range(max_seq_len - len(seq_vec))]
    seq_vec = torch.tensor(seq_vec, dtype=torch.long)
    return seq_vec


def load_list(file_path):
    _data = []
    print(f'loading {file_path}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            _data.append(eval(line))

    return _data


def load_seq(file_path):
    data_ = []
    print(f'loading {file_path} ...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


def load_matrices(file_path):
    print('loading matrices...')
    matrices = np.load(file_path, allow_pickle=True)

    return matrices


def clean_nl(s):
    s = s.strip()
    if s[-1] == ".":
        s = s[:-1]
    s = s.split(". ")[0]
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    processed_words = []
    for w in s.split():
        if w not in punc:
            processed_words.extend(wordninja.split(w))
        else:
            processed_words.append(w)
    return processed_words


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_sequent_mask) != 0


def make_std_mask(nl, pad):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | Variable(
        subsequent_mask(nl.size(-1)).type_as(nl_mask.data))
    return nl_mask



code_Keyword = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case',
                  'catch', 'char', 'class', 'const', 'continue', 'default',
                  'do', 'double', 'else', 'enum', 'extends', 'final',
                  'finally', 'float', 'for', 'goto', 'if', 'implements',
                  'import', 'instanceof', 'int', 'interface', 'long', 'native',
                  'new', 'package', 'private', 'protected', 'public', 'return',
                  'short', 'static', 'strictfp', 'super', 'switch',
                  'synchronized', 'this', 'throw', 'throws', 'transient', 'try',
                  'void', 'volatile', 'while'                    ]

code_keySymbols = ['+', '-', '*', '/', '=', '!', '%', '.', '&', '&&', ',', '+=', '-=', '*=', '/=', '|',
                       '||','>=', '<=', '==', '!=', '>', '<', '^', "<<", '...','[', ']', '(', ')', '{', '}', '"', "'", 
                       '\n', '?', ':', ';', '@']
code_common = ['String', 'expection', 'time', 'Array', 'List', '', 'Object', 'Map', '0', '1', '2', '3', '4',
                 '5',
                 '6', '7', '8', '9',
                 'Override', 'run', 'Command', 'Error', 'Output', 'IO', 'Exception', 'Arrays', 'as', 'find', 'M',
                    'ID', 'Date', 'date', 'Str', 'get', 'Context', 'p', 'No', 'STRING', 'length', 'NUM', 'As',
                    'System', 'current', 'Time', 'Millis', 'log', 'config', 'ret', 'Value', 'sql', 'Statement', 'set',
                    'Int', 'Result', 'Set', 'rs', 'execute', 'Query', 'next', 'close', 'SQL', 'e', 'Level', 'Is',
                    'total', 'Attributes', 'Builder', 'sb', 'append', 'first', 'BOOL', 'key', 'to', 'Bytes', 'buffer',
                    's', 'Random', 'remove', 'All', 'From', 'Collection', 'Unit', 'm', 'contains', 'I', 'Element',
                    'size', 'i', '++', 'name', 'Class', 'File', 'result', 'arraycopy', 'element', 'Data', 'last',
                    'Index', 'Of', 'o', 'index', '--',
                    'equals', 'generate', 'Text', 'Key', 'Instance', 'init', 'on', 'token', 'Token', 'Lock', 'Log',
                    'v', 'TAG', 'Service', 'update', 'Manager', 'create', 'val', 'Math', 'Configuration', 'New',
                    'Operation', 'Name', 'Config', 'Mode', 'Code', 'Server', 'Util', 'parse', 'Calendar', 'Request',
                    'Status', 'Reference', 'add', 'buf', 'Pos', 'text', 'substring', 'pos', 'entity', 'Entity',
                    'Search', 'Test', 'Throwable', 'Document', 'doc', 'Node', 'Character', 'child', 'load', 'Elements',
                    'By', 'Tag', 'item', 'First', 'Child', 'insert', 'ex', 'code', 'SIZE', 'True', 'test', 'Format',
                    'format', 'Equals', 'is', 'replace', 'query', 'matcher', 'Pattern', 'Buffer', 'group', 'With',
                    'Prefix', 'type', 'Item', 'root', 'x', 'Min', 'y', 'Max', 'Stack', 'print', 'Trace', 'handle',
                    'column', 'On', 'Column', 'Header', 'draw', 'Graphics', 'D', 'g', 'Font', 'g2', 'Rectangle',
                    'Bounds', 'Height', 'Width', 'max', 'Float', 'fill', 'Color', 'Line', 'End', 'prefix', 'end',
                    'Entry', 'Info', 'attribute', 'Properties', 'Table', 'Model', 'Names', 'res', 'data', 'id', 'Next',
                    'Queue', 'Id', 'Base', 'base', 'Match', 'str', 'read', 'Buffered', 'Reader', 'in', 'starts',
                    'Path', 'Input', 'Stream', 'tmp', 'Field', 'field', 'Utils', 'Type', 'Byte', 'Boolean', 'Short',
                    'Char', 'Long', 'Double', 'Default', 'Http', 'URL', 'Connection', 'input', 'reader', 'line',
                    'Hash', 'put', 'Response', 'Message', 'Content', 'output', 'value', 'logger', 'info', 'Method',
                    'send', 'message', 'out', 'bytes', 'Read', 't', 'Debug', 'Enabled', 'debug', 'println', 'Size',
                    'Integer', 'msg', 'Count', 'curr', 'start', 'min', 'position', 'E', 'lock', 'q', 'Thread',
                    'Client', 'clear', 'check', 'For', 'Property', 'Types', 'params', 'array', 'To', 'In', 'Target',
                    'X', 'Y', 'target', 'abs', 'State', 'file', 'Collections', 'sort', 'Top', 'Filter', 'Case',
                    'Illegal', 'Argument', 'Number', 'property', 'Location', 'Pool', 'Not', 'Found', 'source', 'Get',
                    'f', 'URI', 'uri', 'Null', 'Pointer', 'delete', 'Access', 'temp', 'Block', 'write', 'show',
                    'width', 'height', 'At', 'hash', 'Store', 'Resource', 'task', 'Task', 'Option', 'list', 'Options',
                    'Vector', 'Image',
                    'request', 'Container', 'Iterator', 'Cache', 'K', 'Empty', 'iterator', 'Root', 'Values', 'Update',
                    'Rule', 'Ref', 'Constants', 'count', 'cur', 'Label', 'Changed', 'len', 'Position', 'builder',
                    'Internal', 'encode', 'Listener', 'Start', 'Listeners', 'flush', 'fail', 'has', 'User', 'Matrix',
                    'actual', 'DEFAULT', 'a', 'b', 'MAX', 'c', 'obj', 'off', 'Out', 'Provider', 'my', 'j', 'db',
                    'Order', 'build', 'raw', 'Factory', 'Socket', 'Point', 'Idx', 'Space', 'method', 'Handler',
                    'Process', 'Bean', 'XML', 'xml', 'offset', 'Offset', 'Length', 'ch', 'C', 'model', 'Address',
                    'node', 'Loader', 'process', 'Param', 'response', 'ERROR', 'action', 'Session', 'And', 'error',
                    'Policy', 'Package', 'Version', 'Area', 'Rect', 'encoded', 'Runtime', 'block', 'compare',
                    'Directory', 'path', 'Parse', 'object', 'attr', 'Attr', 'Sequence', 'sub', 'Writer', 'Num', 'user',
                    'left', 'right', 'Page', 'Action', 'Event', 'decode', 'listener', 'old', 'Change', 'View', 'Bits',
                    'args', 'Source', 'exists', 'Scale', 'Big', 'Decimal', 'scale', 'R', 'Row', 'r', 'N', 'LOG', 'Box',
                    'T', 'items', 'J', 'state', 'image', 'filter', 'Storage', 'Port', 'port', 'range', 'Progress',
                    'progress', 'Host', 'previous', 'Bit', 'bits', 'V', 'entry', 'Range', 'resource', 'Parser',
                    'namespace', 'k', 'num', 'Xml', 'Record', 'color', 'view', 'parent', 'Group', 'Parent',
                    'connection', 'top', 'TYPE', 'JSON', 'map', 'header', 'expected', 'Selected', 'random', 'Tree',
                    'd', 'it', 'Last', 'Pair', 'Channel', 'iter', 'from', 'Current', 'values', 'event', 'Sign',
                    'service', 'context', 'local', 'Dir', 'Non', 'Left', 'Variable', 'string', 'stream', '64', 'java',
                    'Paint', 'paint', 'Attribute', 'Keys', 'copy', 'Len', 'writer', 'Url', 'Files', 'Var', 'tag', 'B',
                    'Op', 'other', 'VALUE', 'scroll', 'Scroll', 'Bottom', 'Layout', 'Params', 'Right', 'row', 'table',
                    'n', 'A', 'os', 'Or', 'Digit', 'w', 'Dimension', 'Parameter', 'number', 'Spec', 'Helper',
                    'content', 'Pane', 'Panel', 'Button', 'NAME', 'sum', 'delta', 'Uri',
                    'Pull', 'Wrapper', 'S', 'l', 'Points', 'point', 'version', 'Impl', 'src', 'dest', 'filename',
                     'cache', 'status', 'url', 'Selection', 'Buf', 'drawable', 'prev', 'match', 'Nodes', 'host',
                    'session', 'Component', 'h', 'Frame', 'Linked', 'server', 'split', 'Local', 'Val', 'idx', 'label',
                    'Menu', 'client', 'Ignore', 'pattern', 'Volume', 'z', 'col', 'Border', 'Edge', 'Bitmap', 'cursor',
                    'address', 'gl', 'Grid', 'XSLT', 'zz']
def get_priority_seq(code_seq):
    priority_seq = []
    temp = []
    for i in code_seq:
        if i in code_Keyword:
            priority_seq.append(3)
        elif i in code_keySymbols:
            priority_seq.append(2)
        elif i in code_common:
            priority_seq.append(1)
        else:
            priority_seq.append(0)
    return priority_seq


def load_ast_index(file_path):
    data_ = []
    def convert_ast_to_graph_format(line):
        line = json.loads(line)
        node_list = [1 for _ in range(len(line))]
        edge_index = [ [0 for _ in range(len(line))] for _ in range(len(line))] # 暂时用临界矩阵
        for id,node in enumerate(line):
            if 'children' in node.keys():
                child_index = node['children']
                for child in child_index:
                    edge_index[id][child] = 1
        edge_index = np.array(edge_index)
        edge_index = torch.tensor(edge_index)
        edge_index = sp.coo_matrix(edge_index)
        indices=np.vstack((edge_index.row,edge_index.col))
        index=torch.tensor(indices)
        values=torch.tensor(edge_index.data)
        edge_index=torch.sparse_coo_tensor(index,values,edge_index.shape)
        node_list = torch.tensor(node_list)
        return node_list,edge_index
    data_ = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            node,edge = convert_ast_to_graph_format(line)
            data_.append([node,edge])
        
    return data_


def load_code(file_path):
    data_ = []
    print(f'loading code {file_path} ...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


