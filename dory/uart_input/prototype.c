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

#include "prototype.h"
#include "mem.h"

void init_prototype(struct prototype *prot, int max_len, int protolen){
  prot->sums = ram_malloc(sizeof(int32_t) * max_len * protolen);
  prot->lengths = (uint16_t *) pi_l2_malloc(sizeof(uint16_t) * max_len);
  prot->one_proto = (int32_t *) pi_l2_malloc(sizeof(int32_t) * protolen);
  for (int i=0; i<max_len; i++){
    (prot->lengths)[i] = 0;
  }
  prot->max_proto = max_len;
  prot->len_proto = protolen;
}

void free_prototype(struct prototype *prot){
  ram_free(prot->sums, sizeof(int32_t) * prot->len_proto * prot->max_proto);
  pi_l2_free(prot->lengths, sizeof(uint16_t)*prot->max_proto);
  pi_l2_free(prot->one_proto, sizeof(int32_t)*prot->len_proto);
}

void reset_prototype(struct prototype *prot){
  for (int i=0; i<prot->len_proto; i++){
    (prot->one_proto)[i] = 0;
  }
  int offset = sizeof(int32_t) * prot->len_proto;
  for (int i=0; i<prot->max_proto; i++){
    ram_write(prot->sums+offset*i, (char *) prot->one_proto, offset);
    (prot->lengths)[i] = 0;
  }
}

void update_prototype(struct prototype *prot, void *new_proto, int gt_class){
  int32_t *int_proto = (int32_t *) new_proto;
  int offset = sizeof(int32_t) * prot->len_proto;
  ram_read((char *) prot->one_proto, prot->sums+offset*gt_class, offset);
  if ((prot->lengths)[gt_class]!=0){
    for (int i=0; i<prot->len_proto; i++){
      (prot->one_proto)[i] += int_proto[i];
    }
  }
  //new class
  else{
    for (int i=0; i<prot->len_proto; i++){
      (prot->one_proto)[i] = int_proto[i];
    }
  }
  (prot->lengths)[gt_class] += 1;
  ram_write(prot->sums+offset*gt_class, (char *) prot->one_proto, offset);
}

void update_prototype_8b(struct prototype *prot, void *new_proto, int gt_class){
  uint8_t *int_proto = (uint8_t *) new_proto;
  int offset = sizeof(int32_t) * prot->len_proto;
  ram_read((char *) prot->one_proto, prot->sums+offset*gt_class, offset);
  if ((prot->lengths)[gt_class]!=0){
    for (int i=0; i<prot->len_proto; i++){
      (prot->one_proto)[i] += int_proto[i];
    }
  }
  //new class
  else{
    for (int i=0; i<prot->len_proto; i++){
      (prot->one_proto)[i] = int_proto[i];
    }
  }
  (prot->lengths)[gt_class] += 1;
  ram_write(prot->sums+offset*gt_class, (char *) prot->one_proto, offset);
}


int32_t *get_one_proto(struct prototype *prot, int idx){
  int offset = sizeof(int32_t) * prot->len_proto;
  ram_read((char *) prot->one_proto, prot->sums+offset*idx, offset);
  if (prot->lengths[idx]!=0){
    for (int i=0; i<prot->len_proto; i++) prot->one_proto[i] = (prot->one_proto)[i]/(prot->lengths)[idx];
  }
  else {
    for (int i=0; i<prot->len_proto; i++) prot->one_proto[i] = 0;
  }
  return prot->one_proto;
}

void set_one_proto(struct prototype *prot, char *inp_proto, int idx){
  // assert (gt_class < prot->max_proto);
  int offset = sizeof(int32_t) * prot->len_proto;
  (prot->lengths)[idx] = 1;

  ram_write(prot->sums+offset*idx, (char *) inp_proto, offset);
}

void set_one_proto_8b(struct prototype *prot, char *inp_proto, int idx){
  // assert (gt_class < prot->max_proto);
  uint8_t *int_proto = (uint8_t *) inp_proto;
  int offset = sizeof(int32_t) * prot->len_proto;
  (prot->lengths)[idx] = 1;
  for (int i=0; i<prot->len_proto; i++){
    (prot->one_proto)[i] = int_proto[i];
  }

  ram_write(prot->sums+offset*idx, (char *) prot->one_proto, offset);
}

void bipolarize_proto(struct prototype *prot, int sign_value){
  int offset = sizeof(int32_t) * prot->len_proto;
  
  for (int idx=0; idx<prot->max_proto; idx++){
    if (prot->lengths[idx]!=0){
      ram_read((char *) prot->one_proto, prot->sums+offset*idx, offset);

      for (int i=0; i<prot->len_proto; i++)
        (prot->one_proto)[i] = ((prot->one_proto)[i] < 0) ? -sign_value : sign_value;
      (prot->lengths)[idx] = 1;

      ram_write(prot->sums+offset*idx, (char *) prot->one_proto, offset);
    }
  }
}

int64_t check_proto_distance_l2(struct prototype prot, void *check_proto, int idx){
  int64_t the_max = 0x7FFFFFFFFFFFFFFF;
  if (idx >= prot.max_proto){
    return the_max;
  }
  if ((prot.lengths)[idx]==0){
    return the_max;
  }
  
  int64_t dist = 0, temp;
  int32_t *int_proto = (int32_t *) check_proto;
  int offset = sizeof(int32_t) * prot.len_proto;
  ram_read((char *)prot.one_proto, prot.sums+offset*idx, offset);
  for (int i=0; i<prot.len_proto; i++){
    temp = (prot.one_proto)[i]/(prot.lengths)[idx] - int_proto[i];
    dist += temp*temp;
  }
  return dist;
}

uint16_t get_class_l2(struct prototype prot, void *check_proto){
  int64_t min_dist = 0x7FFFFFFFFFFFFFFF;
  int64_t dist;
  uint16_t best_idx = 0;
  for (int i=0; i<prot.max_proto; i++){
    dist = check_proto_distance_l2(prot, check_proto, i);
    if (dist < min_dist){
        min_dist = dist;
        best_idx = i;
    }
  }
  return best_idx;
}


float check_proto_distance_cos(struct prototype prot, void *check_proto, int idx){
  float the_min = -1;
  if (idx >= prot.max_proto){
    return the_min;
  }
  if ((prot.lengths)[idx]==0){
    return the_min;
  }
  
  float mag_a=0, mag_b=0, corr=0;
  float temp_a, temp_b;
  int32_t *int_proto = (int32_t *) check_proto;
  int offset = sizeof(int32_t) * prot.len_proto;
  ram_read((char *)prot.one_proto, prot.sums+offset*idx, offset);
  for (int i=0; i<prot.len_proto; i++){
    temp_a = ((float) ((prot.one_proto)[i]))/(prot.lengths)[idx];
    temp_b = (float) (int_proto[i]);
    mag_a += temp_a*temp_a;
    mag_b += temp_b*temp_b;
    corr += temp_a*temp_b;
  }
  float corr_sign = (corr<0) ? -1 : 1;
  return corr_sign*(corr*corr)/(mag_a*mag_b);
}

uint16_t get_class_cos(struct prototype prot, void *check_proto){
  float max_dist = -1;
  float dist;
  uint16_t best_idx = 0;
  for (int i=0; i<prot.max_proto; i++){
    dist = check_proto_distance_cos(prot, check_proto, i);
    if (dist > max_dist){
        max_dist = dist;
        best_idx = i;
    }
  }
  return best_idx;
}