/*
 * network.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified by:
 * Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
 * Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
 * Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
 * Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
 * Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)
 */
#define DEFINE_CONSTANTS
#include "net_utils.h"
#include "pmsis.h"
#include "network.h"
#include "directional_allocator.h"
#include "mem.h"
#include <string.h>
#include "BackpropFullyConnected63.h"
#include "pulp_nn_utils.h"
#include "BackpropLoss.h"


#include "bsp/bsp.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"

// #define VERBOSE 1

#define L3_WEIGHTS_SIZE 4000000
#define L3_INPUT_SIZE 1500000
#define L3_OUTPUT_SIZE 1500000
static void *L3_weights = NULL;
static void *L3_input = NULL;
static void *L3_output = NULL;
static void *L3_grad = NULL;
static void *L2_grad = NULL;
static int cycle_network_execution;
static float eps_in = 1.0;
static float eps_out = 1.0;
static float eps_w = 1.0;

void backprop_grad_reset(){
  int size_grad = (activations_out_size[63] * activations_size[63] + activations_out_size[63]);
  int chunck_size = 10000;
  int send_size = 0;
  char *zeros = (char *) pi_l2_malloc(chunck_size);
  for (int i=0; i<chunck_size; i++){
    zeros[i] = 0;
  }
  for (int i=0; i<size_grad; i+=chunck_size){
    send_size = min(size_grad-i, chunck_size);
    ram_write(L3_grad+i, zeros, send_size);
  }
  pi_l2_free(zeros, chunck_size);
}

void backprop_terminate(){
  pi_l2_free(L2_grad, weights_size[63]*4);
}

void backprop_set_eps(float *e_in, float *e_out, float *e_w){
  // printf("V : %f, %f, %f\n", *e_in, *e_out, *e_w);
  eps_in = *e_in;
  eps_out = *e_out;
  eps_w = *e_w;
  // printf("W : %f, %f, %f\n", eps_in, eps_out, eps_w);
}

void backprop_mem_init(unsigned int W, unsigned int I, unsigned int O, unsigned int WS){
  L3_weights = (void *) W;
  L3_input = (void *) I;
  L3_output = (void *) O;
  int *arr_ws = (int *) WS;
  int n_sizes = 0;
  for (int i = 0; i < 64; i++)
    n_sizes += layer_with_weights[i];
  for (int i = 0; i < n_sizes; i++) 
    L3_weights_size[i] = arr_ws[i];
  L3_grad = L3_output; //borrowing L3 output for L3 grad
  L2_grad = pi_l2_malloc(weights_size[63]*4);
  backprop_grad_reset();
}

void loss_l2_grad(int32_t *gt, int32_t *x, int32_t *out, int arr_len, int32_t batch_size){
  for (int i=0; i<arr_len*batch_size; i++){
    out[i] = 2*(x[i]-gt[i]);
  }
}

void loss_cosim_grad(int32_t *gt_i, int32_t *x_i, int32_t *out_i, int arr_len, float eps_x, int32_t batch_size){
  int32_t *gt, *x, *out;
  for (int j=0; j<batch_size; j++){
    gt = gt_i + arr_len*j;
    x = x_i + arr_len*j;
    out = out_i + arr_len*j;
    float mag_x, mag_gt, mag_x_p2, mag_gt_p2, cossim;
    mag_x_p2=0; mag_gt_p2=0; cossim=0;
    for (int i=0; i<arr_len; i++){
      mag_x_p2  += ((float) (x[i]))*x[i];
      mag_gt_p2 += ((float) (gt[i]))*gt[i];
      cossim += ((float) (gt[i]))*x[i];
    }
    // printf("X : ");
    // for (int i=0; i<10; i++){
    //   printf("%d, ",x[i]);
    // }
    // printf("\n");
    // printf("GT : ");
    // for (int i=0; i<10; i++){
    //   printf("%d, ",gt[i]);
    // }
    // printf("\n");
    
    // printf("%e, %e, %e, %e, %e\n", mag_x, mag_gt, cossim, (float) (x[0]), (float) (gt[0]));
    // __builtin_sqrtf
    // mag_x  = sqrt(mag_x_p2); 
    // mag_gt = sqrt(mag_gt_p2);
    plp_sqrt_f32s_xpulpv2(&mag_x_p2 ,&mag_x ); 
    plp_sqrt_f32s_xpulpv2(&mag_gt_p2,&mag_gt);
    cossim = cossim/(mag_x*mag_gt);
    
    float div_gt = mag_x*mag_gt;
    float div_x = mag_x*mag_x/cossim;
    for (int i=0; i<arr_len; i++){
      // the sign of this calculation is reversed because we need to maximize cosine similarity
      // we dont want to minimize cosisne similarity
      out[i] = (int32_t) ((x[i]/div_x - gt[i]/div_gt)/(eps_x*eps_x));
    }
    // printf("%e, %e, %e, %e, %e, %e\n", mag_x, mag_gt, cossim, div_gt, div_x, eps_x);
    // printf("GRAD : ");
    // for (int i=0; i<10; i++){
    //   printf("%d, ",out[i]);
    // }
    // printf("\n");
  }
}

void backprop_weight_update(void *l2_buffer, int l2_buffer_sz, int divider_w, int divider_b){
  int size_weight = (activations_out_size[63] * activations_size[63])/4;
  int chunk_curr = 0;
  int32_t temp = 0;
  
  // move weight pointer to last alyer weight
  void *L3_weights_curr = L3_weights;
  int weight_l_cnt=0;
  for (int i = 0; i < 63; i++)
    if (layer_with_weights[i])
       L3_weights_curr += L3_weights_size[weight_l_cnt++];
  
  // update weight (weight data saved in 8 bit)
  int chunk_size = l2_buffer_sz/5;
  int32_t *grad_arr = (int32_t *) (l2_buffer+chunk_size);
  int8_t *weight_arr = (int8_t *) (l2_buffer);

  // int ln = 10;
  // int off = 0;
  // ram_read(weight_arr, L3_weights_curr+off, ln);
  // ram_read(grad_arr, L3_grad+off*4, ln*4);
  // printf("divider: %d, %d\n", divider_w, divider_b);
  // printf("weight\n");
  // for (int i=0; i<ln; i++){
  //   printf("%d) %d, %d\n", i, weight_arr[i], grad_arr[i]);
  // }

  for (int i=0; i<size_weight; i+=chunk_size){
    chunk_curr = min(size_weight-i, chunk_size);
    ram_read(weight_arr, L3_weights_curr+i, chunk_curr);
    ram_read(grad_arr, L3_grad+i*4, chunk_curr*4);
    for (int i=0; i<chunk_curr; i++){
      temp = (grad_arr[i]+divider_w/2)/divider_w; //mimic rounding not flooring
      // we save the remainider gradient to the old grad array
      // other option is to set it to 0 after weight update
      grad_arr[i] = grad_arr[i] - temp*divider_w;
      // clipping and weight update
      temp = weight_arr[i] - temp;
      temp = temp>127  ?  127 : temp; //clipping
      temp = temp<-128 ? -128 : temp; //clipping
      weight_arr[i] = temp;
    }
    ram_write(L3_weights_curr+i, weight_arr, chunk_curr);
    ram_write(L3_grad+i*4, grad_arr, chunk_curr*4);
  }

  // update bias (bias data saved in 32 bit)
  chunk_size = l2_buffer_sz/8;
  grad_arr = (int32_t *) (l2_buffer+chunk_size*4);
  int32_t *bias_arr = (int32_t *) (l2_buffer);
  int size_bias = activations_out_size[63]/4;
  
  // ram_read(bias_arr, L3_weights_curr+size_weight+off, ln*4);
  // ram_read(grad_arr, L3_grad+size_weight*4+off*4, ln*4);
  // printf("bias\n");
  // for (int i=0; i<ln; i++){
  //   printf("%d) %d, %d\n", i, bias_arr[i], grad_arr[i]);
  // }

  for (int i=0; i<size_bias; i+=chunk_size){
    chunk_curr = min(size_bias-i, chunk_size);
    ram_read(bias_arr, L3_weights_curr+size_weight+i*4, chunk_curr*4);
    ram_read(grad_arr, L3_grad+size_weight*4+i*4, chunk_curr*4);
    for (int i=0; i<chunk_curr; i++){
      temp = (grad_arr[i]+divider_b/2)/divider_b; //mimic rounding not flooring
      // we save the remainider gradient to the old grad array
      // other option is to set it to 0 after bias update
      grad_arr[i] = grad_arr - temp*divider_b;
      // bias update
      bias_arr[i] = bias_arr[i] - temp;
    }
    ram_write(L3_weights_curr+size_weight+i*4, bias_arr, chunk_curr*4);
    ram_write(L3_grad+size_weight*4+i*4, grad_arr, chunk_curr*4);
  }
}

void backprop_run(void *last_layer_out, void *last_layer_in, void *target_label, size_t l2_buffer_size, int batch_size, int exec)
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  unsigned int args[7];
  args[0] = (unsigned int) last_layer_out;
  args[1] = (unsigned int) l2_buffer_size;
  args[2] = (unsigned int) batch_size;
  args[3] = (unsigned int) exec;
  args[4] = (unsigned int) target_label;
  args[5] = (unsigned int) last_layer_in;
  args[6] = (unsigned int) (&eps_out); 
  
  // // Calculate loss function gradient (older method single core)
  // loss_l2_grad((int32_t *)target_label, (int32_t *)last_layer_out, (int32_t *)target_label, activations_out_size[63]/4, batch_size);
  // loss_cosim_grad((int32_t *)target_label, (int32_t *)last_layer_out, (int32_t *)target_label, activations_out_size[63]/4, eps_out, batch_size);
  // printf("F : ");
  // for (int i=0; i<10; i++){
  //   printf("%d, ", ((int32_t *) target_label)[i]);
  // }
  // printf("\n");

  // open cluster...
  pi_cluster_task(&cluster_task, backprop_run_cluster_last, args);
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return;
  // Then offload an entry point, this will get executed on the cluster controller
  cluster_task.stack_size = 3500;
  cluster_task.slave_stack_size = 3400;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  pi_cluster_close(&cluster_dev);
  // print_perf("Final", cycle_network_execution, 146392064);

  // int chunk_size = l2_buffer_size/5;
  // int32_t *grad_arr = (int32_t *) (last_layer_out+chunk_size);
  // int ln = 30;
  // int off = 0;
  // ram_read(grad_arr, L3_grad+off*4, ln*4);
  // printf("GRD : ");
  // for (int i=0; i<ln; i++){
  //   printf("%d, ", grad_arr[i]);
  // }
  // printf("\n");
}

void backprop_run_cluster_last(void *args) {
  unsigned int *real_args = (unsigned int *) args;
  void *last_layer_out = (void *) real_args[0];
  size_t l2_buffer_size = (size_t) real_args[1];
  unsigned int batch_size = (unsigned int) real_args[2];
  int exec = (int) real_args[3];
  void *target_label = (void *) real_args[4];
  void *last_layer_in = (void *) real_args[5];
  void *eps = (void *) real_args[6];

  void *L2_output = last_layer_out;
  void *L2_input = last_layer_in;
  void *L1_buffer;
  // reqirement1 36700 > ALL_OUT*BYTE_OUT*3 + 4*NUM_CORES*3
  // reqirement2 36700 > batch_size*(LEN_IN/2)*BYTE_IN + batch_size*LEN_OUT*BYTE_OUT + LEN_OUT*(LEN_IN/2)*BYTE_W + LEN_OUT*BYTE_B
  // check LEN_OUT, BYTE_OUT, etc in BackpropFullyConnected63_L2.c
  if (pi_core_id() == 0) L1_buffer = pmsis_l1_malloc(36700);

  layer_args_t largs = {
    .L3_input = (unsigned int) NULL,
    .L3_output = (unsigned int) NULL, 
    .L3_after_weights = (unsigned int) NULL, 
    .L2_input = (unsigned int) last_layer_out, //borrow this place holder
    .bypass = (unsigned int) batch_size, //borrow this place holder
    .L2_output = (unsigned int) target_label,  //borrow this place holder
    .L2_weights = (unsigned int) target_label, //borrow this place holder
    .L1_buffer = L1_buffer,
    .ram = (unsigned int) get_ram_ptr(),
    .out_mult = (unsigned int) 1,
    .out_shift = (unsigned int) 0,
    .layer_id = (unsigned int) eps
  };
  cycle_network_execution = 0;
  pi_perf_conf(1<<PI_PERF_CYCLES);
  pi_perf_reset();
  pi_perf_stop();
  pi_perf_start();
  pi_cl_team_fork(NUM_CORES, (void *)BackpropLossCossim, &largs);

  largs.L3_after_weights = (unsigned int) L3_grad; //borrow this place holder
  largs.L2_input = (unsigned int) last_layer_in;
  largs.bypass = (unsigned int) batch_size; //borrow this place holder
  largs.L2_output = (unsigned int) target_label; //borrow this place holder
  largs.L2_weights = (unsigned int) L2_grad; //borrow this place holder
  largs.L1_buffer = L1_buffer;

  // perf measurement begin
  pi_cl_team_fork(NUM_CORES, (void *)BackpropFullyConnected63, &largs);

  // Free L1 Memory
  if (pi_core_id() == 0) pmsis_l1_malloc_free(L1_buffer, 36700);

  // performance measurements: end
  pi_perf_stop();
  cycle_network_execution += pi_perf_read(PI_PERF_CYCLES);
}