# Copyright (C) 2022-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Authors: 
# Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
# Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
# Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)


import time
import serial
import torch
from dataclasses import dataclass
from tqdm import tqdm
import struct

ser_header = b"ABC"

# Explanation
# SER_CODE_CHECK_BUFFER 
# - check if buffer ready
# SER_CODE_TEST_PROTO 
# - send test data asking for prorotype return
# SER_CODE_TEST_CLASS 
# - send test data asking for class return
# SER_CODE_TRAIN_FULL_PROTO 
# - send training data asking for class prototype return
# SER_CODE_TRAIN_FULL_NONE 
# - send training data asking acknowledge return
# SER_CODE_RESET_PROTO 
# - reset prototype value to 0
# SER_CODE_GET_PROTO_MEAN 
# - checking cluster mean value of certain class

SLEEP_TIME                  = 0.001
WAIT_TIMEOUT                = 1
SER_RESP_LEN                = 6
SER_CODE_CHECK_BUFFER       = 1
SER_CODE_TEST_PROTO         = 2
SER_CODE_TEST_CLASS         = 3
SER_CODE_TRAIN_FULL_PROTO   = 4
SER_CODE_TRAIN_FULL_NONE    = 5
SER_CODE_TRAIN_ACT_PROTO    = 6
SER_CODE_TRAIN_ACT_NONE     = 7
SER_CODE_RESET_PROTO        = 8
SER_CODE_GET_PROTO_MEAN     = 9
SER_CODE_SET_PROTO_MEAN     = 10
SER_CODE_RECALC_PROTO       = 11
SER_CODE_LAST_LAYER_TRAIN   = 12
SER_CODE_BIPOLARIZE_PROTO   = 13
SER_CODE_RESET_ACT          = 14
SER_CODE_GET_ACT_MEAN       = 15
SER_CODE_SET_ACT_MEAN       = 16
SER_CODE_TEST_ACT           = 17
SER_CODE_TEST_LAST_LAYER    = 18
SER_CODE_SET_EPS            = 19

@dataclass
class serial_data:
    opcode: int
    data: torch.Tensor = torch.tensor(0)
    gt_class: torch.Tensor = None
    byte_sz:int = 1


def wait_with_timeout(serial_handler:serial.Serial, byte_len:int = 1, verbose=True):
    start = time.perf_counter()
    while serial_handler.in_waiting < byte_len:
        time.sleep(SLEEP_TIME)
        end = time.perf_counter()
        if (end-start>=WAIT_TIMEOUT):
            if verbose:
                print("Serial wait timeout")
            break

def send_serial_data(serial_handler:serial.Serial, data:serial_data, sleep_time=0):
    # send data length
    serial_handler.write(ser_header)
    serial_handler.write(int(data.opcode).to_bytes(1,'little'))
    time.sleep(sleep_time)

    if (data.opcode == SER_CODE_TEST_PROTO) or \
      (data.opcode == SER_CODE_TEST_CLASS) or \
      (data.opcode == SER_CODE_TEST_ACT) or \
      (data.opcode == SER_CODE_TEST_LAST_LAYER):
        dat_len = int(data.data.nelement())*data.byte_sz
        serial_handler.write(dat_len.to_bytes(4,'little'))
        time.sleep(sleep_time)
        send_data_body(serial_handler, data.data, data.byte_sz)
        
    elif (data.opcode == SER_CODE_TRAIN_FULL_PROTO) or \
      (data.opcode == SER_CODE_TRAIN_FULL_NONE) or \
      (data.opcode == SER_CODE_TRAIN_ACT_PROTO) or \
      (data.opcode == SER_CODE_TRAIN_ACT_NONE) or \
      (data.opcode == SER_CODE_SET_PROTO_MEAN) or \
      (data.opcode == SER_CODE_SET_ACT_MEAN):
        dat_len = int(data.data.nelement())*data.byte_sz
        serial_handler.write(dat_len.to_bytes(4,'little'))
        serial_handler.write(int(data.gt_class).to_bytes(2,'little'))
        time.sleep(sleep_time)
        send_data_body(serial_handler, data.data, data.byte_sz)
    
    elif (data.opcode == SER_CODE_GET_PROTO_MEAN) or \
      (data.opcode == SER_CODE_GET_ACT_MEAN):
        serial_handler.write(int(data.gt_class).to_bytes(2,'little'))

    elif (data.opcode == SER_CODE_BIPOLARIZE_PROTO):
        serial_handler.write(int(data.data).to_bytes(4,'little'))
    
    elif (data.opcode == SER_CODE_LAST_LAYER_TRAIN):
        serial_handler.write(int(data.data[0]).to_bytes(2,'little')) # number of epoch
        serial_handler.write(int(data.data[1]).to_bytes(4,'little')) # weight grad_divider
        serial_handler.write(int(data.data[2]).to_bytes(4,'little')) # bias grad_divider

    elif (data.opcode == SER_CODE_SET_EPS):
        serial_handler.write(bytearray(struct.pack("f", float(data.data[0])))) # number of epoch
        serial_handler.write(bytearray(struct.pack("f", float(data.data[1])))) # number of epoch
        serial_handler.write(bytearray(struct.pack("f", float(data.data[2])))) # number of epoch
    
    # SER_CODE_RECALC_PROTO, SER_CODE_CHECK_BUFFER, SER_CODE_RESET_PROTO have not body, just need to send the header

def send_data_body(serial_handler:serial.Serial, data:torch.Tensor, byte_sz:int):
    if len(data.shape) == 3:
        send_dat = data.permute(1,2,0).flatten()
    elif (len(data.shape) == 2) or (len(data.shape) == 1):
        send_dat = data.flatten()
    else:
        print("Data dimension hsould be less than 4 and not 0")
        assert False
    for i,x in enumerate(send_dat):
        serial_handler.write(int(x).to_bytes(byte_sz,'little', signed=True)) 

def reset_prototype(serial_handler:serial.Serial):
    send_serial_data(serial_handler, serial_data(opcode=SER_CODE_RESET_PROTO))
    return recv_serial_response(serial_handler)

def get_cls_prototype(serial_handler:serial.Serial, cls):
    send_serial_data(serial_handler, serial_data(opcode=SER_CODE_GET_PROTO_MEAN, gt_class=cls))
    return recv_serial_response(serial_handler)

def set_cls_prototype(serial_handler:serial.Serial, data:torch.Tensor, cls):
    send_serial_data(serial_handler, serial_data(opcode=SER_CODE_SET_PROTO_MEAN, data=data, gt_class=cls, byte_sz=4))
    return recv_serial_response(serial_handler)

def ready_barrier(serial_handler:serial.Serial):
    resp = 0
    while(resp==0):
        send_serial_data(serial_handler, serial_data(opcode=SER_CODE_CHECK_BUFFER, data=torch.zeros(1)))
        wait_with_timeout(serial_handler, byte_len=1, verbose=False)
        if (serial_handler.in_waiting > 0):
           resp = recv_serial_response(serial_handler)

def get_output_batch(serial_handler:serial.Serial, data:serial_data):
    all_output = []
    if data.gt_class is None:
        for i,x in tqdm(enumerate(data.data)):
            one_data = serial_data(opcode=data.opcode, data=x, gt_class=None, byte_sz=data.byte_sz)
            send_serial_data(serial_handler, one_data)
            recv_data = recv_serial_response(serial_handler)
            all_output.append(recv_data.unsqueeze(0))
    else:
        for i,(x,y) in tqdm(enumerate(zip(data.data, data.gt_class))):
            one_data = serial_data(opcode=data.opcode, data=x, gt_class=y, byte_sz=data.byte_sz)
            send_serial_data(serial_handler, one_data)
            recv_data = recv_serial_response(serial_handler)
            all_output.append(recv_data.unsqueeze(0))
    
    return torch.cat(all_output)

def recv_serial_response(serial_handler:serial.Serial, wait_first=True):
    status = 0
    # first while barrier
    if wait_first:
        while serial_handler.in_waiting < 1:
            time.sleep(SLEEP_TIME)
    
    # parse header
    while(status<len(ser_header)):
        wait_with_timeout(serial_handler, 1)
        recv = serial_handler.read(1)
        if ser_header[status:status+1] == recv:
            status += 1
    
    # get data info
    wait_with_timeout(serial_handler, 6)
    data_len = int.from_bytes(serial_handler.read(4), byteorder="little", signed=False)
    byte_len = int.from_bytes(serial_handler.read(1), byteorder="little", signed=False)
    sign_type = int.from_bytes(serial_handler.read(1), byteorder="little", signed=False)
    sign_type = not (sign_type == 0)

    # get data body
    data = torch.zeros(data_len, dtype=torch.int)
    for i in range(data_len):
        wait_with_timeout(serial_handler, byte_len)
        data[i] = int.from_bytes(serial_handler.read(byte_len), byteorder="little", signed=sign_type)

    return data

class model_serial(torch.nn.Module):
    # mode = "meta" or "neurcollapse"
    def __init__(self, eps_in, eps_out, eps_w, dev='/dev/ttyUSB1', baud_rate=115200, max_class= 100, mode="meta", etf_vec=None):
        super().__init__()
        self.serial = serial.Serial(dev, baud_rate, timeout=0.050)
        self.max_class = max_class
        self.on_board = True

        # Initialize important variable for integer backprop
        self.eps_in = eps_in
        self.eps_out = eps_out
        self.eps_w = eps_w
        self.eps_b = eps_out
        self.eps_gb = self.eps_b # epsilon gradient b
        self.eps_gw = eps_in * eps_out # epsilon gradient w
        self.sign_value = torch.round(1/eps_out)

        self.mode = mode
        if (max_class==etf_vec.shape[0]):
            self.etf_vec = etf_vec
        else:
            self.etf_vec = etf_vec.t()

    # arg input is only for compatibility reason
    def reset_prototypes(self, arg):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_RESET_PROTO))
        proto_resp = recv_serial_response(self.serial)
        return 1
    
    # reset memory for last layer activation input
    def reset_activation(self):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_RESET_ACT))
        act_resp = recv_serial_response(self.serial)
        return 1
    
    def init_prototypes(self):
        if (self.mode == "neurcollapse") and (self.etf_vec is not None):
            self.init_etf_vec()
        else:
            self.reset_prototypes()

    # setting epsilon for last layer quantization
    def init_eps(self):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_SET_EPS, data=[self.eps_in, self.eps_out, self.eps_w]))
        return recv_serial_response(self.serial)
    
    # reset etf_vector for neural collapse lase layer training
    def init_etf_vec(self):
        if self.etf_vec.is_floating_point():
            etf_vec_int = torch.round(self.etf_vec/self.eps_out).type(torch.int32)
        else:
            etf_vec_int = self.etf_vec
        ser_data = serial_data(opcode=SER_CODE_SET_PROTO_MEAN, data=etf_vec_int, gt_class=torch.arange(self.max_class), byte_sz=4)
        return get_output_batch(self.serial, ser_data)
    
    # Training with prototype update (not last layer activation update)
    # activation memory will not be updated but the prototype will be updated
    def update_prototypes(self, x, target):
        ser_data = serial_data(opcode=SER_CODE_TRAIN_FULL_NONE, data=x, gt_class=target)
        output = get_output_batch(self.serial, ser_data)
    
    # Training with ast layer activation update
    # activation memory will be updated but the prototype will not be updated
    def update_feat_replay(self, x, target):
        ser_data = serial_data(opcode=SER_CODE_TRAIN_ACT_NONE, data=x, gt_class=target)
        output = get_output_batch(self.serial, ser_data)
    
    # Update the prototype memory according to current activation memory
    def recalculate_prototypes_feat(self):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_RECALC_PROTO))
        return recv_serial_response(self.serial)

    # Bipolarized the prototype memory. this memory is important for last layer training
    def bipolarize_prototypes(self):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_BIPOLARIZE_PROTO, data=self.sign_value))
        return recv_serial_response(self.serial)

    # Do last layer training
    def last_layer_training(self, learn_rate, epoch=1):
        # learning_rate
        self.learn_rate = learn_rate
        self.gb_div = torch.round(self.eps_b/(self.eps_gb*learn_rate))
        self.gw_div = torch.round(self.eps_w/(self.eps_gw*learn_rate))
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_LAST_LAYER_TRAIN, data=[epoch, self.gw_div, self.gb_div]))
        return recv_serial_response(self.serial)
    
    # Get 1 vector of prototype memory
    def check_proto(self, cls):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_GET_PROTO_MEAN, gt_class=cls))
        return recv_serial_response(self.serial)
    
    # Get 1 vector of activation memory
    def check_act(self, cls):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_GET_ACT_MEAN, gt_class=cls))
        return recv_serial_response(self.serial)
    
    # Get multiple vectors of proto memory (classes is 1 dim array (numpy/torch) or List)
    def check_proto_batch(self, classes):
        result = []
        for i in classes:
            result += [self.check_proto(i).unsqueeze(0)]
        return torch.cat(result, dim=0)
    
    # Get multiple vectors of activation memory (classes is 1 dim array (numpy/torch) or List)
    def check_act_batch(self, classes):
        result = []
        for i in classes:
            result += [self.check_act(i).unsqueeze(0)]
        return torch.cat(result, dim=0)

    # Set 1 vector of prototype memory
    def set_proto(self, x, cls):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_SET_PROTO_MEAN, data=x, gt_class=cls, byte_sz=4))
        return recv_serial_response(self.serial)
    
    # Set 1 vector of activation memory
    def set_act(self, x, cls):
        send_serial_data(self.serial, serial_data(opcode=SER_CODE_SET_ACT_MEAN, data=x, gt_class=cls, byte_sz=4))
        return recv_serial_response(self.serial)
    
    # Set multiple vectors of prototype memory 
    # - classes is 1 dim array (numpy/torch) or List
    # - first dimension of x should match number of classses
    def set_proto_batch(self, x, classes):
        for dat,cls in zip(x,classes):
            result = [self.set_proto(dat,cls).unsqueeze(0)]
    
    # Set multiple vectors of activation memory 
    # - classes is 1 dim array (numpy/torch) or List
    # - first dimension of x should match number of classses
    def set_act_batch(self, x, classes):
        for dat,cls in zip(x,classes):
            result = [self.set_act(dat,cls).unsqueeze(0)]
    
    # forward propagation with one hot output
    def forward(self, x):
        self.similarities = torch.zeros(x.shape[0],self.max_class)
        ser_data = serial_data(opcode=SER_CODE_TEST_CLASS, data=x, gt_class=None)
        output = get_output_batch(self.serial, ser_data).to(torch.int64)
        device = x.device
        return torch.nn.functional.one_hot(output, num_classes=100).squeeze(1).double().to(device)
    
    # forward propagation with class id return
    def forward_class(self, x):
        self.similarities = torch.zeros(x.shape[0],self.max_class)
        ser_data = serial_data(opcode=SER_CODE_TEST_CLASS, data=x, gt_class=None)
        output = get_output_batch(self.serial, ser_data).to(torch.int64)
        device = x.device
        return output.view(-1).to(device)
    
    # forward propation with protorype output
    def embedding(self, x):
        ser_data = serial_data(opcode=SER_CODE_TEST_PROTO, data=x, gt_class=None)
        output = get_output_batch(self.serial, ser_data)
        dev = x.device
        return output.to(dev)
    
    # forward propagation with activation output (without last layer)
    def conv_embedding(self, x):
        ser_data = serial_data(opcode=SER_CODE_TEST_ACT, data=x, gt_class=None)
        output = get_output_batch(self.serial, ser_data)
        dev = x.device
        return output.to(dev)
    
    # do forward propagtion for the last layer
    def fc_embedding(self, x):
        ser_data = serial_data(opcode=SER_CODE_TEST_LAST_LAYER, data=x, gt_class=None)
        output = get_output_batch(self.serial, ser_data)
        dev = x.device
        return output.to(dev)
    
    #function for compatibility reason
    def get_feat_replay(self):
        return None, None
    
    #function for compatibility reason
    def update_prototypes_feat(self, feat,label,nways_session):
        self.recalculate_prototypes_feat()

    #function for compatibility reason
    def nudge_prototypes(self, nways_session,writer,session, gpu):
        pass

    #function for compatibility reason
    def hrr_superposition(self, nways_session, compress):
        pass



####################################################################################################
# Older useless test class
####################################################################################################
def uart_test():
    import torchvision.transforms as transforms
    from PIL import Image

    # serial config
    ser = serial.Serial('/dev/ttyUSB1', 230400, timeout=0.050)

    # fetch one data
    img_dat = torch.round(torch.rand(1,32,32)*255)

    # Sending data
    data = serial_data(opcode=SER_CODE_TRAIN_FULL_PROTO, data=img_dat, gt_class=3)
    send_serial_data(ser, data)
    
    # receiving data
    recv_data = recv_serial_response(ser)

    A = img_dat.reshape(-1,256).sum(0)
    B = recv_data
    print((A!=B).sum())

    ser.close()


def mnist_test():
    import os
    from torchvision import datasets, transforms

    def emulate_network(inp:torch.Tensor):
        batch = inp.shape[0]
        return inp.permute(0,2,3,1).reshape(batch,-1,256).sum(1)
    
    # serial config
    ser = serial.Serial('/dev/ttyUSB1', 460800, timeout=0.050)
    device = 'gpu'
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32))
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)),'data'),
                    train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
                    dataset1, batch_size=8, num_workers=1, pin_memory=True, shuffle=False)
    
    response = reset_prototype(ser)
    if response.item() == 1 :
        print("Reset successful")
    
    correct = 0
    ndata = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = torch.round(data*255)-128
            data = data.repeat(1, 3, 1,1)
            # data, target = data.to(device), target.to(device)
            ser_data = serial_data(opcode=SER_CODE_TEST_PROTO, data=data, gt_class=target)
            output = get_output_batch(ser, ser_data, [256])
            checker = emulate_network(data)
            correct += ((checker != output).sum(1) == 0).sum(0)
            ndata += len(checker)
            print(i)
            if (i==20):
                break
    
    print(100. * correct / ndata)

# if __name__=="__main__":
#     mnist_test()