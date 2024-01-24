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

#include "pmsis.h"
#include "network.h"


void network_initialize(){}
void network_terminate(){}

void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int type_run)
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  unsigned int args[4];
  args[0] = (unsigned int) l2_buffer;
  args[1] = (unsigned int) l2_buffer_size;
  args[2] = (unsigned int) l2_final_output;
  args[3] = (unsigned int) exec;
  // open cluster...
  pi_cluster_task(&cluster_task, network_run_cluster, args);
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return;
  // // Then offload an entry point, this will get executed on the cluster controller
  // cluster_task.stack_size = 3800;
  // cluster_task.slave_stack_size = 3600;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  pi_cluster_close(&cluster_dev);
}

void network_run_cluster(void *args){
  unsigned int * real_args = (unsigned int *) args;
  void * l2_buffer = (void *) real_args[0];
  size_t l2_buffer_size = (size_t) real_args[1];
  void * l2_final_output = (void *) real_args[2];
  int exec = (int) real_args[3];

  int32_t *L2_output = (int32_t *) pi_l2_malloc(activations_out_size[1]);
  
  // any calculation will just works
  uint8_t *char_input = (uint8_t *) l2_buffer;
  for (int i=0; i<activations_out_size[1]/4;i++){
    L2_output[i] = 0;
    for(int j=i; j<activations_size[0]; j+=activations_out_size[1]/4){
        L2_output[i] += char_input[j];
    }
  }
  
  // copy output
  for (int i=0; i<activations_out_size[1]; i++)
    *((uint8_t*)(l2_final_output)+i) = *((uint8_t*)(L2_output)+i);

  pi_l2_free(L2_output, activations_out_size[1]);
}

void network_run_cluster_conv(void *args){
  unsigned int * real_args = (unsigned int *) args;
  void * l2_buffer = (void *) real_args[0];
  size_t l2_buffer_size = (size_t) real_args[1];
  void * l2_final_output = (void *) real_args[2];
  int exec = (int) real_args[3];

  int32_t *L2_output = (int32_t *) pi_l2_malloc(activations_out_size[1]);
  
  // any calculation will just works
  uint8_t *char_input = (uint8_t *) l2_buffer;
  for (int i=0; i<activations_out_size[1]/4;i++){
    L2_output[i] = 0;
    for(int j=i; j<activations_size[0]; j+=activations_out_size[1]/4){
        L2_output[i] += char_input[j];
    }
  }
  
  // copy output
  for (int i=0; i<activations_out_size[1]; i++)
    *((uint8_t*)(l2_final_output)+i) = *((uint8_t*)(L2_output)+i);
  for (int i=activations_out_size[1]; i<activations_size[1]; i++)
    *((uint8_t*)(l2_final_output)+i) = 0;

  pi_l2_free(L2_output, activations_out_size[1]);
}

// Do nothing
void network_run_cluster_last(void *args){}
void backprop_run_cluster_last(void * args){}
void backprop_mem_init(unsigned int W, unsigned int I, unsigned int O, unsigned int WS){}
void network_get_mem_addr(unsigned int *W, unsigned int *I, unsigned int *O, unsigned int *WS){}
void backprop_run(void *l2_buffer, void *last_layer_out, void *target_label, size_t l2_buffer_size, void *l2_final_output, int exec){}
void backprop_weight_update(void *l2_buffer, int l2_buffer_sz, int divider){}


