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


import os
import re
import sys
import shutil

def is_empty_line(inp):
    temp = inp.replace(" ", "")
    temp = temp.replace("\t", "")
    temp = temp.replace("\r", "")
    temp = temp.replace("\n", "")
    if len(temp) == 0:
        return True
    elif len(temp) >= 2:
        return (temp[0:2] == "//")
    else:
        return False

def parse_int_with_cue(filename, cue):
    with open(filename, "r") as f:
        text = f.read()
        start = text.find(cue)+len(cue)
        end = start
        while (text[end].isdigit()):
            end += 1
    ret = 0
    if text[start:end].isdigit():
        ret = int(text[start:end])
    return ret

def get_files(path):
    for file in os.listdir(path):
        abs_path = os.path.join(path, file)
        if os.path.isfile(abs_path):
            yield abs_path

def remove_duplicate(filename, pattern):
    with open(filename, "r+") as fp:
        lines = fp.readlines() 
        len_lines = len(lines)

        if (pattern[-1]=="\n"):
            p = pattern
        else:
            p = pattern + "\n"
        prev_line_found = False

        for i in range(len_lines-1, -1, -1):
            if is_empty_line(lines[i]):
                continue
            elif p in lines[i]:
                if (prev_line_found):
                    lines.pop(i)
                prev_line_found = True
            else:
                prev_line_found = False
            


        fp.seek(0)
        fp.truncate()
        fp.writelines(lines)

def remove_lines(filename, string_patterns):
    with open(filename, "r+") as fp:
        lines = fp.readlines() 
        len_lines = len(lines)

        for i in range(len_lines-1, -1, -1):
            for s in string_patterns:
                if s in lines[i]:
                    lines.pop(i)
                    break

        fp.seek(0)
        fp.truncate()
        fp.writelines(lines)

def replace_strings(filename, string_tuples):
    with open(filename, "r+") as fp:
        text = fp.read() 

        for s1,s2 in string_tuples:
            text = re.sub(s1, s2, text)

        fp.seek(0)
        fp.truncate()
        fp.write(text)

if __name__=="__main__":
    template_dir     = sys.argv[1] 
    dory_network_dir = sys.argv[2] 

    # Edit files in SRC
    src_loc = os.path.join(dory_network_dir,"src")
    files_to_update = ["Addition", "BNReluConvolution", "Linear", "FullyConnected", "Pooling", "main.c", "BackpropLoss"]
    lines_to_remove = ["dory_dma_channel", "dory_dma_free", "pulp.h", "dma_id"]
    string_to_replace = [
        ("dory_dma.h", "gap9_dma.h"),
        ("dory_get_tile.h", "dory.h"),
        ("dory_dma_memcpy_async\(", "gap9_dma("),
        ("dory_dma_barrier\([&A-Z_a-z0-9]*\);", "pi_cl_team_barrier(0);"),
        #these edit is only for main.c
        (" PMU_set_voltage", " //PMU_set_voltage"),
    ]
    for filename in get_files(src_loc):
        for fu in files_to_update:
            if fu in filename:
                print(filename)
                remove_lines(filename, lines_to_remove)
                replace_strings(filename, string_to_replace)
                remove_duplicate(filename, "pi_cl_team_barrier(0);")
                break
    
    # Edit files in inc
    inc_loc = os.path.join(dory_network_dir,"inc")
    files_to_update = ["pulp_nn_utils.h","network"]
    lines_to_remove = ["pulp.h"]
    # string_to_replace = [
    #     ("#ifdef GAP_SDK\n#include \"pulp.h\"\n#endif", ""),
    # ]
    for filename in get_files(inc_loc):
        for fu in files_to_update:
            if fu in filename:
                print(filename)
                remove_lines(filename, lines_to_remove)
                break

    # Edit src/net_utils.c
    filename = os.path.join(dory_network_dir,"src/net_utils.c")
    string_to_replace = [
        ("#include \"net_utils.h\"\n", "#include \"net_utils.h\" \n#include \"pmsis.h\""),
    ]
    print(filename)
    replace_strings(filename, string_to_replace)

    # Edit src/mem.c
    filename = os.path.join(dory_network_dir,"src/mem.c")
    string_to_replace = [
        ("pi_hyperflash_conf_init", "pi_mx25u51245g_conf_init"),
        ("flash_conf_t flash_conf", "struct pi_mx25u51245g_conf flash_conf"),
        ("ram_conf_t ram_conf", "struct pi_hyper_conf ram_conf"),
        ("pi_hyperram_conf_init", "pi_default_ram_conf_init"),
        ("bsp/flash/hyperflash.h", "bsp/flash/spiflash.h"),
    ]
    print(filename)
    replace_strings(filename, string_to_replace)
    
    # Edit src/network.c
    filename = os.path.join(dory_network_dir,"src/network.c")
    slave_stack = parse_int_with_cue(filename, "cluster_task.slave_stack_size = ")
    stack = parse_int_with_cue(filename, "cluster_task.stack_size = ")
    l1_size = parse_int_with_cue(filename, "pmsis_l1_malloc(")
    if (l1_size==0):
        l1_size = parse_int_with_cue(filename, "0, L1_buffer, ")
    lines_to_remove = [
        "layer_args->L1_buffer",
        "pmsis_l1_malloc",
        "L1_buffer;",
    ]
    string_to_replace = [
        # ("pmsis_l1_malloc\(", "pi_cl_l1_malloc((void *) 0, "),
        # ("pmsis_l1_malloc_free\(", "pi_cl_l1_free((void *) 0, "),

        # comment stack size definition
        (" cluster_task.stack_size", " //cluster_task.stack_size"),
        (" cluster_task.slave_stack_size", " //cluster_task.slave_stack_size"),
        # # comment print
        # (" print_perf", " //print_perf"),
        # ("\n#define VERBOSE", "\n//#define VERBOSE"),
        # define global L1_buffer
        ("static void \*L3_output = NULL;\n", 
            "static void *L3_output = NULL; \n" +
            "static void *L1_buffer = NULL;\n"),
        # multiline does:1. initiate L1 buffer, 2. restructure cluster definition, 3. difining stack in gap9 way
        ("pi_cluster_task\(&cluster_task, network_run_cluster, args\);\n" + 
         "  pi_open_from_conf\(&cluster_dev, &conf\);\n" + 
         "  if \(pi_cluster_open\(&cluster_dev\)\)\n" + 
         "    return;",
            "pi_open_from_conf( &cluster_dev, &conf);\n" +
            "  if (pi_cluster_open(&cluster_dev))\n" +
            "    return;\n" +
            "  L1_buffer = pi_l1_malloc(&cluster_dev, "+str(l1_size)+");\n" +
            "  pi_cluster_task(&cluster_task, network_run_cluster, args);\n" +
            "  int stacks_size = "+ str(slave_stack) +" * pi_cl_cluster_nb_pe_cores();\n" +
            "  void *stacks = pi_cl_l1_scratch_alloc(&cluster_dev, &cluster_task, stacks_size);\n" +
            "  pi_cluster_task_stacks(&cluster_task, stacks, "+ str(slave_stack) +");"
        ),
        # Same thing as above but it is for backprop supported network.c
        ("if\(type_run==0\){ pi_cluster_task\(&cluster_task, network_run_cluster, args\);}\n" +
         "  else if \(type_run==1\) {pi_cluster_task\(&cluster_task, network_run_cluster_conv, args\);}\n" +
         "  else if \(type_run==2\) {pi_cluster_task\(&cluster_task, network_run_cluster_last, args\);}\n" +
         "  pi_open_from_conf\(&cluster_dev, &conf\);\n" + 
         "  if \(pi_cluster_open\(&cluster_dev\)\)\n" + 
         "    return;",
            "pi_open_from_conf( &cluster_dev, &conf);\n" +
            "  if (pi_cluster_open(&cluster_dev))\n" +
            "    return;\n" +
            "  L1_buffer = pi_l1_malloc(&cluster_dev, "+str(l1_size)+");\n" +
            "  if(type_run==0){ pi_cluster_task(&cluster_task, network_run_cluster, args);}\n" +
            "  else if (type_run==1) {pi_cluster_task(&cluster_task, network_run_cluster_conv, args);}\n" +
            "  else if (type_run==2) {pi_cluster_task(&cluster_task, network_run_cluster_last, args);}\n" +
            "  int stacks_size = "+ str(slave_stack) +" * pi_cl_cluster_nb_pe_cores();\n" +
            "  void *stacks = pi_cl_l1_scratch_alloc(&cluster_dev, &cluster_task, stacks_size);\n" +
            "  pi_cluster_task_stacks(&cluster_task, stacks, "+ str(slave_stack) +");"
        ),
        # Same thing as above but it is for backprop.c
        ("pi_cluster_task\(&cluster_task, backprop_run_cluster_last, args\);\n" +
         "  pi_open_from_conf\(&cluster_dev, &conf\);\n" + 
         "  if \(pi_cluster_open\(&cluster_dev\)\)\n" + 
         "    return;",
            "pi_open_from_conf( &cluster_dev, &conf);\n" +
            "  if (pi_cluster_open(&cluster_dev))\n" +
            "    return;\n" +
            "  L1_buffer = pi_l1_malloc(&cluster_dev, "+str(l1_size)+");\n" +
            "  pi_cluster_task(&cluster_task, backprop_run_cluster_last, args);\n" +
            "  int stacks_size = "+ str(slave_stack) +" * pi_cl_cluster_nb_pe_cores();\n" +
            "  void *stacks = pi_cl_l1_scratch_alloc(&cluster_dev, &cluster_task, stacks_size);\n" +
            "  pi_cluster_task_stacks(&cluster_task, stacks, "+ str(slave_stack) +");"
        ),
        # add L1 free
        ("pi_cluster_send_task_to_cl\(&cluster_dev, &cluster_task\);\n", 
            "pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task); \n" +
            "  pi_cl_l1_free((void *) 0, L1_buffer, " +str(l1_size) + ");\n"
        ),
        # update cluster config
        ("pi_cluster_conf_init\(&conf\);\n", 
            "pi_cluster_conf_init(&conf); \n" +
            "  conf.cc_stack_size = " + str(stack) + ";\n"
        ),
        ("conf.id=0;\n", 
            "conf.id=0; \n" +
            "  conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE | PI_CLUSTER_ICACHE_PREFETCH_ENABLE | PI_CLUSTER_ICACHE_ENABLE;\n"
        ),
        # assign L1_buffer to execute layer
        (".L1_buffer = 0,", ".L1_buffer = L1_buffer,"),
    ]
    print(filename)
    replace_strings(filename, string_to_replace)
    remove_lines(filename, lines_to_remove)

    
    filename = os.path.join(dory_network_dir,"src/backprop.c")
    if(os.path.isfile(filename)):
        print(filename)
        replace_strings(filename, string_to_replace)
        remove_lines(filename, lines_to_remove)


    shutil.copy(os.path.join(template_dir,"dory.c"), src_loc)
    shutil.copy(os.path.join(template_dir,"gap9_dma.c"), src_loc)
    shutil.copy(os.path.join(template_dir,"dory.h"), inc_loc)
    shutil.copy(os.path.join(template_dir,"gap9_dma.h"), inc_loc)

    if(os.path.isfile(os.path.join(src_loc, "dory_dma.c"))):
        os.remove(os.path.join(src_loc, "dory_dma.c"))