
// Copyright (C) 2022-2024 ETH Zurich
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// SPDX-License-Identifier: Apache-2.0
// ==============================================================================
// 
// Authors: 
// Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
// Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
// Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
// Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
// Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)

#ifndef __UART_CONTOL_H__
#define __UART_CONTOL_H__

/* ENplanation
SER_CODE_CHECK_BUFFER 
- check if buffer ready
SER_CODE_TEST_PROTO 
- send test data asking for prorotype return
SER_CODE_TEST_CLASS 
- send test data asking for class return
SER_CODE_TRAIN_FULL_PROTO 
- send training data asking for class prototype return
SER_CODE_TRAIN_FULL_NONE 
- send training data asking acknowledge return
SER_CODE_RESET_PROTO 
- reset prototype value to 0
SER_CODE_GET_PROTO_MEAN 
- checking cluster mean value of certain class
*/
#define SER_CODE_LEN 1
#define SER_RESP_LEN 6

#define SER_CODE_CHECK_BUFFER 1
#define SER_CODE_TEST_PROTO 2
#define SER_CODE_TEST_CLASS 3
#define SER_CODE_TRAIN_FULL_PROTO 4
#define SER_CODE_TRAIN_FULL_NONE 5
#define SER_CODE_TRAIN_ACT_PROTO 6
#define SER_CODE_TRAIN_ACT_NONE 7
#define SER_CODE_RESET_PROTO 8
#define SER_CODE_GET_PROTO_MEAN 9
#define SER_CODE_SET_PROTO_MEAN 10
#define SER_CODE_RECALC_PROTO 11
#define SER_CODE_LAST_LAYER_TRAIN 12
#define SER_CODE_BIPOLARIZE_PROTO 13
#define SER_CODE_RESET_ACT 14
#define SER_CODE_GET_ACT_MEAN 15
#define SER_CODE_SET_ACT_MEAN 16
#define SER_CODE_TEST_ACT 17
#define SER_CODE_TEST_LAST_LAYER 18
#define SER_CODE_SET_EPS 19

#define SER_INFO_LEN_CHECK_BUFFER 0
#define SER_INFO_LEN_CLASS_RESET 0
#define SER_INFO_LEN_RECALC_PROTO 0
#define SER_INFO_LEN_TEST 4
#define SER_INFO_LEN_TRAIN 6
#define SER_INFO_LEN_CLASS_RESET 0
#define SER_INFO_LEN_GET_PROTO 2
#define SER_INFO_LEN_SET_PROTO 6
#define SER_INFO_LEN_BIPOLARIZE 4
#define SER_INFO_LEN_SET_EPS 12
#define SER_INFO_LEN_LAST_LAYER_TRAIN 10

#include "uart_buffer.h"

static char *ser_header = "ABC";

struct serial_data {
  uint32_t data_len;
  uint8_t *data;
  uint16_t gt_class;  // String
  uint8_t opcode;
};

void recv_data_body(struct uart_duplex_buffer *uart_buf, void *buffer, int dat_len);

void recv_serial_data(struct uart_duplex_buffer *uart_buf, struct serial_data *dat);

void send_serial_response(struct uart_duplex_buffer *uart_buf, uint8_t *data, uint32_t data_len,  uint8_t byte_len, uint8_t sign_type);

// void serial_routine(struct pi_device *device);

#endif  // __UART_CONTOL_H__