import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import threading
import time
import yaml
import json
from distutils import dir_util
import sys

APP_NAME = 'fc_svm'


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for master, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.master = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.input_train = None
        self.input_test = None
        self.output_train = None
        self.output_test = None
        self.output_model = None
        self.split_mode = None
        self.split_dir = None

        self.sep = None
        self.label = None


        self.data_train_X = []
        self.data_test_X = []
        
        self.data_train_y = []
        self.data_test_y = []

        self.filename = None
        
        self.learning_rate = None
        self.regularization = None
        self.batch_size = None
        self.local_steps = None
        self.convergence_rounds = None
        self.convergence_factor = None
        
        self.m = None # data dimensionality
        self.splits = None
        self.parameter_w = None
        self.parameter_b = None
        self.local_costs = None

        self.labels = {}
        self.global_costs = None
        self.global_w = None
        self.global_b = None

        self.min_costs = None
        self.min_w = None
        self.min_b = None

        self.converged_splits = []
        self.converged_counts = []

        self.lock = threading.Lock()


    def handle_setup(self, client_id, master, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.master = master
        self.clients = clients
        print(f'Received setup: {self.id} {self.master} {self.clients}')

        self.read_config()

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()


    def handle_incoming(self, data):
        # This method is called when new data arrives
        self.lock.acquire()
        d = json.loads(data.read().decode())
        self.data_incoming.append(d)
        self.lock.release()


    def handle_outgoing(self):
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing


    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[APP_NAME]

            self.input_train = config['input'].get('train')
            self.input_test = config['input'].get('test')

            self.output_train = config['output'].get('train')
            self.output_test = config['output'].get('test')
            self.output_model = config['output'].get('model')

            self.split_mode = config['split'].get('mode')
            self.split_dir = config['split'].get('dir')

            self.sep = config['format']['sep']
            self.label = config['format']['label']
            
            self.learning_rate = config['parameter'].get('learning_rate')
            self.regularization = config['parameter'].get('regularization')
            self.batch_size = config['parameter'].get('batch_size')
            self.local_steps = config['parameter'].get('local_steps')
            self.convergence_rounds = config['parameter'].get('convergence_rounds')
            self.convergence_factor = config['parameter'].get('convergence_factor')

            
    def app_flow(self):
        # This method contains a state machine for the slave and master instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_compute_local = 3
        state_gather = 4
        state_wait = 5
        state_finishing = 7
        state_done = 8

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            if state == state_initializing:
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input

            # COMMON PART

            if state == state_read_input:

                def read_input(ins, path, mode):
                    d = pd.read_csv(path, sep=ins.sep)
                    if mode == 'train':
                        y = np.array([l[0] for l in d.filter([self.label], axis=1).to_numpy()])
                        labels = np.unique(y)
                        if len(labels) != 2:
                            print('Error: more than 2 classes')
                            state = state_finishing
                        else:
                            self.labels[labels[0]] = -1
                            self.labels[labels[1]] = 1
                            ins.data_train_y.append(np.array([self.labels[yy] for yy in y]))
                            ins.data_train_X.append(d.drop(self.label, axis=1).to_numpy())
                    elif mode == 'test':
                        y = np.array([l[0] for l in d.filter([self.label], axis=1).to_numpy()])
                        ins.data_test_y.append(np.array([self.labels[yy] for yy in y]))
                        ins.data_test_X.append(d.drop(self.label, axis=1).to_numpy())

                print('Reading input...')
                
                if self.split_mode == 'directory':
                    base_dir = os.path.normpath(os.path.join(f'/mnt/input/', self.split_dir))
                    for split_name in os.listdir(base_dir):
                        read_input(self, os.path.join(base_dir, split_name, self.input_train), 'train')
                        if self.input_test is not None:
                            read_input(self, os.path.join(base_dir, split_name, self.input_test), 'test')
                elif self.split_mode == 'file':
                    base_dir = os.path.normpath('/mnt/input/')
                    read_input(self, os.path.join(base_dir, self.input_train), 'train')
                    if self.input_test is not None:
                        read_input(self, os.path.join(base_dir, self.input_test), 'test')

                state = state_compute_local

            if state == state_compute_local:
                print('Calculate local values...')
                
                if len(self.data_incoming) == 0:
                    # initialize local params
                    print('Initializing parameter')
                    
                    self.m = len(self.data_train_X[0][0])
                    self.splits = len(self.data_train_X)
                    
                    self.parameter_w = [np.zeros(self.m) for s in range(self.splits)]
                    self.parameter_b = [0 for s in range(self.splits)]
                    self.local_costs = [None for s in range(self.splits)]                    
                    
                elif 'final_model' in self.data_incoming[0]:
                    state = state_finishing
                    self.min_b = self.data_incoming[0]['b']
                    self.min_w = [np.array(w) for w in self.data_incoming[0]['w']]
                    self.global_costs = self.data_incoming[0]['costs']

                else:
                    #read global params from self.data_incoming
                    self.parameter_b = self.data_incoming[0]['b']
                    self.parameter_w = [np.array(w) for w in self.data_incoming[0]['w']]
                    self.converged_splits = self.data_incoming[0]['converged_splits']
                    

                if state == state_compute_local:

                    for split in range(self.splits):  
                        if self.converged_splits is not None and split not in self.converged_splits:
                            print('Train accuracy split '+str(split)+' :   ' + str(self.train_accuracy(split)))
                            self.cost(split)
                            for i in range(self.local_steps):
                                self.update(split)
                        
                    self.data_incoming = []
                    
                    if self.master:
                        self.lock.acquire()
                        self.data_incoming.append({'id':self.id, 'costs':self.local_costs, 'b':self.parameter_b, 'w': [w.tolist() for w in self.parameter_w]})#str.encode(json.dumps({'id':self.id, 'costs':self.local_costs, 'b':self.parameter_b, 'w': [w.tolist() for w in self.parameter_w]})))
                        self.lock.release()
                        state = state_gather
                    else:
                        self.data_outgoing = {'id':self.id, 'costs':self.local_costs, 'b':self.parameter_b, 'w':[w.tolist() for w in self.parameter_w]}
                        self.status_available = True
                        state = state_wait


            # GLOBAL PART

            if state == state_gather:

                if len(self.data_incoming) > len(self.clients):
                    print('data correction...')
                    tmp = {}
                    for cli in self.data_incoming:
                        print(cli['id'])
                        tmp[cli['id']] = cli 
                    self.data_incoming = [tmp[k] for k in tmp]
                    
                if len(self.data_incoming) == len(self.clients):
                    
                    print(f'Have everything, continuing...')
                    
                    client_num = len(self.data_incoming)
                    
                    if self.global_costs is None:
                        self.global_costs = []
                    
                    self.global_costs.append([0 for s in range(self.splits)])

                    for s in range(self.splits):
                        if s not in self.converged_splits:
                            for cli in self.data_incoming:                        
                                self.global_costs[-1][s] += cli['costs'][s]

                    #print('\n\nglobal costs:   ' + str(self.global_costs) + '\n\n')

                    if self.min_costs is None:
                        self.min_costs = [sys.maxsize for s in range(self.splits)]
                        self.converged_counts = [0 for s in range(self.splits)]
                        self.min_w = [None for s in range(self.splits)] 
                        self.min_b = [None for s in range(self.splits)]

                    
                    # checking for convergence
                    
                    for s in range(self.splits):
                        if s not in self.converged_splits:

                            if self.global_costs[-1][s] < self.min_costs[s] * self.convergence_factor:
                                self.converged_counts[s] = 0
                                print('split ' + str(s) + ' not converged')

                            else:
                                self.converged_counts[s] += 1
                                if self.converged_counts[s] >= self.convergence_rounds:
                                    self.converged_splits.append(s)
                                    print('split ' + str(s) + ' converged')


                            if self.global_costs[-1][s] < self.min_costs[s] and self.global_w is not None:
                                self.min_costs[s] = self.global_costs[-1][s]
                                self.min_w[s] = self.global_w[s]
                                self.min_b[s] = self.global_b[s]


                    # global parameter aggregation          

                    self.global_b = []
                    self.global_w = []

                    for s in range(self.splits):
                        b = 0
                        w = np.zeros(self.m)

                        if s not in self.converged_splits:
                            for cli in self.data_incoming:
                                b += cli['b'][s]
                            b /= client_num                            
                            for cli in self.data_incoming:
                                w += np.array(cli['w'][s])
                            w /= client_num
                            w = w.tolist()
                        else:
                            w = self.min_w[s]
                            b = self.min_b[s]


                        self.global_b.append(b)
                        self.global_w.append(w)


                    if len(self.converged_splits) != self.splits:

                        data_outgoing = {'b': self.global_b, 'w': self.global_w, 'converged_splits': self.converged_splits}
                        self.data_incoming = [{'b': self.global_b, 'w': self.global_w, 'converged_splits': self.converged_splits}]

                        self.data_outgoing = json.dumps(data_outgoing)
                        self.status_available = True
                        
                        state = state_compute_local

                    else:
                        data_outgoing = {'final_model': True, 'b': self.min_b, 'w': self.min_w, 'costs': self.global_costs}
                        self.data_incoming = [{'final_model': True, 'b': self.min_b, 'w': self.min_w, 'costs': self.global_costs}]
                        
                        self.data_outgoing = json.dumps(data_outgoing)
                        self.status_available = True

                        state = state_finishing
                        
                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_finishing:
                
                inv_labels = {self.labels[l]:l for l in self.labels}

                def write_output(path, X, s):
                    print("writing output to " + path)
                    with open(path, 'w') as w:
                        w.write('y_pred\n')
                        for x in X:
                            y = self.predict(x, self.min_w[s], self.min_b[s])
                            w.write(str(inv_labels[y]) + '\n')

                def write_output_y(path, Y):
                    print("writing output to " + path)
                    with open(path, 'w') as w:
                        w.write('y_true\n')
                        for y in Y:
                            w.write(str(inv_labels[y]) + '\n')



                if self.split_mode == 'directory':
                    base_dir = os.path.normpath(os.path.join(f'/mnt/output/', self.split_dir))
                    for s, split_name in enumerate(os.listdir(base_dir)):
                        write_output(os.path.join(base_dir, split_name, self.output_train), self.data_train_X[s], s)
                        write_output_y(os.path.join(base_dir, split_name, 'train_y.csv'), self.data_train_y[s])
                        if self.input_test is not None:
                            write_output(os.path.join(base_dir, split_name, self.output_test), self.data_test_X[s], s)
                            write_output_y(os.path.join(base_dir, split_name, 'test_y.csv'), self.data_test_y[s])
                
                elif self.split_mode == 'file':
                    base_dir = os.path.normpath('/mnt/output/')
                    write_output(os.path.join(base_dir, self.output_train), self.data_train_X[0], 0)
                    write_output_y(os.path.join(base_dir, split_name, 'train_y.csv'), self.data_train_y[0])
                    if self.input_test is not None:
                        write_output(os.path.join(base_dir, self.output_test), self.data_test_X[0], 0)
                        write_output_y(os.path.join(base_dir, split_name, 'test_y.csv'), self.data_test_y[0])

                state = state_done
                self.status_finished = True
                break


            # LOCAL PART

            if state == state_wait:
                if len(self.data_incoming) > 0:
                    state = state_compute_local
                else:
                    print('Client: Waiting for data...')
                
            time.sleep(1)


    def update(self, split):
        if self.batch_size == None or self.batch_size == -1:
            X = self.data_train_X[split]
            y = self.data_train_y[split]
        else:
            rn = range(len(self.data_train_y[split]))
            ind = [np.random.choice(rn) for i in range(self.batch_size)] 

            X = [self.data_train_X[split][i] for i in ind]
            y = [self.data_train_y[split][i] for i in ind]

        self.gradient_descent(X, y, split)


    def gradient_descent(self, X, y, split):
        dw, db = 0, 0
        l = len(X)
        lb = 2/(l*self.regularization)
    
        for i in range(l):
            x = X[i]
            sd = y[i] * (np.dot(self.parameter_w[split], x) + self.parameter_b[split])
            if sd < 1:
                dw += lb * self.parameter_w[split] - y[i] * x
                db += -y[i]
            else:
                dw += lb * self.parameter_w[split]

        self.parameter_w[split] -= dw * self.learning_rate / l
        self.parameter_b[split] -= db * self.learning_rate / l


    def cost(self, split):
        loss = 0
        n = len(self.data_train_y[split])
        for i in range(n):
            l = max(0, 1 - self.data_train_y[split][i] * (np.dot(self.parameter_w[split], self.data_train_X[split][i]) + self.parameter_b[split]))
            loss += l

        c = 0.5 * np.dot(self.parameter_w[split], self.parameter_w[split]) + self.regularization * loss

        self.local_costs[split] = c


    def predict(self, x, w, b):
        return int(np.sign(np.dot(w, x) + b))


    def test_accuracy(self, split):
        X = self.data_test_X[split]
        w = self.parameter_w[split]
        b = self.parameter_b[split]
        c = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i], w, b)
            
            y_true = self.data_test_y[split][i]
            if y_true == y_pred:
                c += 1
        return c*1.0/len(X)


    def train_accuracy(self, split):
        X = self.data_train_X[split]
        w = self.parameter_w[split]
        b = self.parameter_b[split]
        c = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i], w, b)
            y_true = self.data_train_y[split][i]
            if y_true == y_pred:
                c += 1
        return c*1.0/len(X)


logic = AppLogic()
