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

/* PMSIS includes */
#include "pmsis.h"
#include "uart_controller.h"

void recv_data_body(struct uart_duplex_buffer *uart_buf, void *buffer, int dat_len){
    char *buf_char = (char *) buffer;
    read_data_wait(uart_buf, buf_char, dat_len);
    
    // // Debug
    // printf("len %d\ndata:\n", dat_len);
    // for (int i=0; i<10; i++)
    //     printf("   %d, %ld\n", i, buf_char[i]);
    // for (int i=dat_len-10; i<dat_len; i++)
    //     printf("   %d, %ld\n", i, buf_char[i]);
}

void recv_serial_data(struct uart_duplex_buffer *uart_buf, struct serial_data *dat){
    // Parsing the header
    int status = 0;
    int head_len = strlen(ser_header);
    char *buf_char = (char *) pi_l2_malloc(SER_CODE_LEN);
    while(status<head_len) {
        read_data_wait(uart_buf, buf_char, 1);
        if(buf_char[0]==ser_header[status])
            status++;
        else
            status=0;
    }

    // Fetch opcode
    read_data_wait(uart_buf, buf_char, SER_CODE_LEN);
    uint8_t opcode = ((uint8_t *) (buf_char))[0];
    dat->opcode = opcode;
    pi_l2_free(buf_char, SER_CODE_LEN);

    // Get mode info based on the opcode
    // Accept Test Data return prototype (SER_CODE_TEST_PROTO)
    // Accept Test Data return class (SER_CODE_TEST_CLASS)
    // Accept Test Data return last layer input (SER_CODE_TEST_ACT)
    // Accept Test Data do last layer calculation (SER_CODE_TEST_LAST_LAYER)
    if ( (opcode==SER_CODE_TEST_PROTO) || 
         (opcode==SER_CODE_TEST_CLASS) ||
         (opcode==SER_CODE_TEST_ACT) ||
         (opcode==SER_CODE_TEST_LAST_LAYER) ){
        int info_len = SER_INFO_LEN_TEST;
        buf_char = (char *) pi_l2_malloc(info_len);
        
        // read data
        read_data_wait(uart_buf, buf_char, info_len);
        dat->data_len = ((uint32_t *) (buf_char))[0];
        read_data_wait(uart_buf, dat->data, dat->data_len);

        //clean
        pi_l2_free(buf_char, info_len);
    }

    // Accept Train Data return prototype (SER_CODE_TRAIN_FULL_PROTO)
    // Accept Train Data return acknowledge (SER_CODE_TRAIN_FULL_NONE)
    else if ( (opcode==SER_CODE_TRAIN_FULL_PROTO) ||
              (opcode==SER_CODE_TRAIN_FULL_NONE) ||
              (opcode==SER_CODE_TRAIN_ACT_PROTO) ||
              (opcode==SER_CODE_TRAIN_ACT_NONE) ){
        int info_len = SER_INFO_LEN_TRAIN;
        buf_char = (char *) pi_l2_malloc(info_len);

        //read data
        // printf("AA, %d\n", opcode);
        read_data_wait(uart_buf, buf_char, info_len);
        dat->data_len = ((uint32_t *) (buf_char))[0];
        dat->gt_class = ((uint16_t *) (buf_char+4))[0];
        // printf("BB %d\n", dat->data_len);
        recv_data_body(uart_buf, dat->data, dat->data_len);
        // printf("CC\n");

        //clean
        pi_l2_free(buf_char, info_len);
    }

    // Accept proto check
    else if ( (opcode==SER_CODE_GET_PROTO_MEAN) ||
              (opcode==SER_CODE_GET_ACT_MEAN) ){
        int info_len = SER_INFO_LEN_GET_PROTO;
        buf_char = (char *) pi_l2_malloc(info_len);

        //read data
        read_data_wait(uart_buf, buf_char, info_len);
        dat->gt_class = ((uint16_t *) buf_char)[0];

        //clean
        pi_l2_free(buf_char, info_len);
    }

    //
    else if ( (opcode==SER_CODE_SET_PROTO_MEAN) ||
              (opcode==SER_CODE_SET_ACT_MEAN) ){
        int info_len = SER_INFO_LEN_SET_PROTO;
        buf_char = (char *) pi_l2_malloc(info_len);

        //read data
        read_data_wait(uart_buf, buf_char, info_len);
        dat->data_len = ((uint32_t *) (buf_char))[0];
        dat->gt_class = ((uint16_t *) (buf_char+4))[0];
        recv_data_body(uart_buf, dat->data, dat->data_len);

        //clean
        pi_l2_free(buf_char, info_len);
    }
    
    // Bipolarize prototype
    else if (opcode==SER_CODE_LAST_LAYER_TRAIN){
        int info_len = SER_INFO_LEN_LAST_LAYER_TRAIN;
        buf_char = (char *) pi_l2_malloc(info_len);

        //read data
        read_data_wait(uart_buf, buf_char, info_len);
        ((uint16_t *) (dat->data+0))[0] = ((uint16_t *) (buf_char+0))[0];
        ((uint32_t *) (dat->data+2))[0] = ((uint32_t *) (buf_char+2))[0];
        ((uint32_t *) (dat->data+6))[0] = ((uint32_t *) (buf_char+6))[0];

        //clean
        pi_l2_free(buf_char, info_len);
    }

    // Bipolarize prototype
    else if (opcode==SER_CODE_BIPOLARIZE_PROTO){
        read_data_wait(uart_buf, dat->data, SER_INFO_LEN_BIPOLARIZE);
    }

    // Bipolarize prototype
    else if (opcode==SER_CODE_SET_EPS){
        read_data_wait(uart_buf, dat->data, SER_INFO_LEN_SET_EPS);
    }

    // SER_CODE_CHECK_BUFFER, SER_CODE_RESET_PROTO, SER_CODE_RESET_ACT, SER_CODE_RECALC_PROTO dont have body
}

void send_serial_response(struct uart_duplex_buffer *uart_buf, uint8_t *data, uint32_t data_len,  uint8_t byte_len, uint8_t sign_type){
    int info_len = sizeof(int);
    int head_len = strlen(ser_header);

    // printf(" 1\n");
    send_data_wait(uart_buf, ser_header,  strlen(ser_header));
    char *sb = (char *) pi_l2_malloc(sizeof(char) * 4);
    sb[0] = ((char*) &data_len)[0];
    sb[1] = ((char*) &data_len)[1];
    sb[2] = ((char*) &data_len)[2];
    sb[3] = ((char*) &data_len)[3];
    // printf(" 2\n");
    send_data_wait(uart_buf, sb,  4);
    sb[0] = byte_len;
    // printf(" 3\n");
    send_data_wait(uart_buf, sb,  1);
    sb[0] = sign_type;
    // printf(" 4\n");
    send_data_wait(uart_buf, sb,  1);
    // printf(" 5\n");
    send_data_wait(uart_buf, data, data_len*byte_len);
    // printf(" 6\n");

    pi_l2_free(sb, sizeof(char) * 4);

}
