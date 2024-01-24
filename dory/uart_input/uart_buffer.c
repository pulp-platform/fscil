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

#include "uart_buffer.h"

static pi_task_t rx_task = {0};
static pi_task_t tx_task = {0};
static struct uart_buffer *rx_buf_global;
static struct uart_buffer *tx_buf_global;
static struct pi_device *uart_global;

void uart_rx_cb(void *arg){
  struct uart_buffer *rx_buf = (struct uart_buffer *) arg;
  if(rx_buf->start){
    if (~is_buffer_full(rx_buf)){
        rx_buf->n += 1;
        rx_buf->head = (rx_buf->head + 1) % rx_buf->max;
    }
    // pi_task_callback(&rx_task, (void *) uart_rx_cb, rx_buf); // I dont think we need this redefinition
    pi_uart_read_async(uart_global, (rx_buf->data) + (rx_buf->head), 1, &rx_task);
  }
}

void uart_tx_cb(void *arg){
  struct uart_buffer *tx_buf = (struct uart_buffer *) arg;
  if(tx_buf->start){
    if (~is_buffer_empty(tx_buf)){
        tx_buf->n -= 1;
        tx_buf->tail = (tx_buf->tail + 1) % tx_buf->max;
        pi_task_callback(&tx_task, (void *) uart_tx_cb, tx_buf); // I dont think we need this redefinition
        pi_uart_write_async(uart_global, (tx_buf->data) + (tx_buf->head), 1, &tx_task);
    }
  }
}

void init_uart_buf(struct uart_duplex_buffer *uart_buf, struct pi_device *uart, int max_len){
  uart_global = uart;
  // // RX
  // uart_buf->rx_buf = &rx_buf_global;
  uart_buf->rx_buf->data = (uint8_t *) pi_l2_malloc(sizeof(uint8_t) * max_len);
  uart_buf->rx_buf->head = 0;
  uart_buf->rx_buf->tail = 0;
  uart_buf->rx_buf->n = 0;
  uart_buf->rx_buf->max = max_len;
  uart_buf->rx_buf->start = false;
  // // TX
  // uart_buf->tx_buf = &tx_buf_global;
  uart_buf->tx_buf->data = (uint8_t *) pi_l2_malloc(sizeof(uint8_t) * max_len);
  uart_buf->tx_buf->head = 0;
  uart_buf->tx_buf->tail = 0;
  uart_buf->tx_buf->n = 0;
  uart_buf->tx_buf->max = max_len;
  uart_buf->tx_buf->start = false;

  start_buffering(uart_buf);
}

void free_uart_buf(struct uart_duplex_buffer *uart_buf){
  // RX
  pi_l2_free(uart_buf->rx_buf->data, uart_buf->rx_buf->max);
  // TX
  pi_l2_free(uart_buf->tx_buf->data, uart_buf->tx_buf->max);
}

void start_buffering(struct uart_duplex_buffer *uart_buf){
  // RX
  struct uart_buffer *rx_buf = uart_buf->rx_buf;
  rx_buf->start = true;
  pi_task_callback(&rx_task, (void *) uart_rx_cb, rx_buf);
  pi_uart_read_async(uart_global, rx_buf->data, 1, &rx_task);
  // TX
  struct uart_buffer *tx_buf = uart_buf->tx_buf;
  tx_buf->start = true;
}

uint16_t read_data(struct uart_duplex_buffer *uart_buf, char *out_buf, uint16_t len){
  struct uart_buffer *rx_buf = uart_buf->rx_buf;
  uint16_t fetch = len;
  if (len>rx_buf->n) fetch = rx_buf->n;
  for(int i=0; i<fetch; i++){
    out_buf[i] = rx_buf->data[(rx_buf->tail+i)%(rx_buf->max)];
  }
  rx_buf->n -= fetch;
  rx_buf->tail = (rx_buf->tail+fetch) % rx_buf->max;
  return fetch;
}

uint16_t send_data(struct uart_duplex_buffer *uart_buf, char *inp_buf, uint16_t len){
  struct uart_buffer *tx_buf = uart_buf->tx_buf;
  uint16_t send = num_free_buffer(tx_buf);
  if (send>len) send = len;
  tx_buf->n += send;
  for(int i=0; i<send; i++){
    tx_buf->data[(tx_buf->head+i)%(tx_buf->max)] = inp_buf[i];
  }
  tx_buf->head = (tx_buf->head+send) % tx_buf->max;
  pi_task_callback(&tx_task, (void *) uart_tx_cb, tx_buf);
  pi_uart_write_async(uart_global, tx_buf->data, 1, &tx_task);
  return send;
}

uint16_t read_data_wait(struct uart_duplex_buffer *uart_buf, char *out_buf, uint16_t len){
  struct uart_buffer *rx_buf = uart_buf->rx_buf;
  for(int i=0; i<len; i++){
    while(rx_buf->n==0){
      pi_time_wait_us(1);
    }
    out_buf[i] = rx_buf->data[rx_buf->tail];
    rx_buf->n -= 1;
    rx_buf->tail = (rx_buf->tail+1) % rx_buf->max;
  }
  return len;
}

uint16_t send_data_wait(struct uart_duplex_buffer *uart_buf, char *inp_buf, uint16_t len){
  struct uart_buffer *tx_buf = uart_buf->tx_buf;
  pi_uart_write(uart_global, inp_buf, len);
  return len;  
}

uint16_t check_rx_buffer(struct uart_duplex_buffer *uart_buf){
  return check_buffer(uart_buf->rx_buf);
}

uint16_t check_tx_buffer(struct uart_duplex_buffer *uart_buf){
  return check_buffer(uart_buf->tx_buf);
}

uint16_t num_free_buffer(struct uart_buffer *uart_buf){
  return uart_buf->max - uart_buf->n;
}
uint16_t check_buffer(struct uart_buffer *uart_buf){
  return uart_buf->n;
}
bool is_buffer_full(struct uart_buffer *uart_buf){
  return uart_buf->n == uart_buf->max;
}
bool is_buffer_empty(struct uart_buffer *uart_buf){
  return uart_buf->n <= 0;
}

