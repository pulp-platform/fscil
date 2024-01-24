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

#ifndef __UART_BUFFER_H__
#define __UART_BUFFER_H__

#include "pmsis.h"
#include <stdint.h>
#include <stdbool.h>

struct uart_buffer {
  volatile uint8_t *data; 
  volatile uint16_t head; 
  volatile uint16_t tail; 
  volatile uint16_t n;
  volatile uint16_t max;
  volatile bool start;
};

struct uart_duplex_buffer {
  struct uart_buffer *rx_buf;
  struct uart_buffer *tx_buf;
};

void init_uart_buf(struct uart_duplex_buffer *uart_buf, struct pi_device *uart, int max_len);
void free_uart_buf(struct uart_duplex_buffer *uart_buf);
void start_buffering(struct uart_duplex_buffer *uart_buf);
// void stop_buffering(struct uart_duplex_buffer uart_buf); // wrong implementation
uint16_t read_data(struct uart_duplex_buffer *uart_buf, char *out_buf, uint16_t len);
uint16_t send_data(struct uart_duplex_buffer *uart_buf, char *inp_buf, uint16_t len);
uint16_t read_data_wait(struct uart_duplex_buffer *uart_buf, char *out_buf, uint16_t len);
uint16_t send_data_wait(struct uart_duplex_buffer *uart_buf, char *inp_buf, uint16_t len);
uint16_t check_rx_buffer(struct uart_duplex_buffer *uart_buf);
uint16_t check_tx_buffer(struct uart_duplex_buffer *uart_buf);

uint16_t check_buffer(struct uart_buffer *uart_buf);
uint16_t num_free_buffer(struct uart_buffer *uart_buf);
bool is_buffer_full(struct uart_buffer *uart_buf);
bool is_buffer_empty(struct uart_buffer *uart_buf);
void uart_rx_cb(void *arg);
void uart_tx_cb(void *arg);


#endif  // __UART_BUFFER_H__