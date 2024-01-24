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


import sys
import os
import shutil

def parse_int_with_cue(filename, cue):
    with open(filename, "r") as f:
        text = f.read()
        start = text.find(cue)+len(cue)
        end = start
        while (text[end].isdigit()):
            end += 1
    return int(text[start:end])


def main():
    fscil_template_dir = sys.argv[1]
    dory_network_dir = sys.argv[2]

    #checking
    if not os.path.isdir(dory_network_dir):
        print("checking fail")
        return
    if not os.path.isdir(os.path.join(dory_network_dir, "src")):
        print("checking fail")
        return
    if not os.path.isdir(os.path.join(dory_network_dir, "inc")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(dory_network_dir, "src", "main.c")):
        print("checking fail")
        return
    print("Correct dory_network_dir = " + dory_network_dir)

    if not os.path.isfile(os.path.join(fscil_template_dir,"uart_controller.h")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(fscil_template_dir,"prototype.h")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(fscil_template_dir,"uart_buffer.h")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(fscil_template_dir,"uart_controller.c")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(fscil_template_dir,"prototype.c")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(fscil_template_dir,"uart_buffer.c")):
        print("checking fail")
        return
    if not os.path.isfile(os.path.join(fscil_template_dir,"main.c")):
        print("checking fail")
        return
    print("Correct fscil_template_dir = " + fscil_template_dir)
    
    # copy library    
    dst = os.path.join(dory_network_dir, "inc")
    shutil.copy(os.path.join(fscil_template_dir,"uart_controller.h"), dst)
    shutil.copy(os.path.join(fscil_template_dir,"prototype.h"), dst)
    shutil.copy(os.path.join(fscil_template_dir,"uart_buffer.h"), dst)
    dst = os.path.join(dory_network_dir, "src")
    shutil.copy(os.path.join(fscil_template_dir,"uart_controller.c"), dst)
    shutil.copy(os.path.join(fscil_template_dir,"prototype.c"), dst)
    shutil.copy(os.path.join(fscil_template_dir,"uart_buffer.c"), dst)
    
    # copy main
    dst = os.path.join(dory_network_dir, "inc", "main_old.c")
    if not os.path.isfile(dst):
        shutil.copy(
            os.path.join(dory_network_dir, "src", "main.c"), 
            dst
        )
    shutil.copy(
        os.path.join(fscil_template_dir, "main.c"), 
        os.path.join(dory_network_dir, "src")
    )

    l2_size = parse_int_with_cue(
        os.path.join(dory_network_dir, "inc", "main_old.c"),
        "void *l2_buffer = pi_l2_malloc("
    )
    l2_size_old = parse_int_with_cue(
        os.path.join(dory_network_dir, "src", "main.c"),
        "define L2_INPUT_SIZE "
    )
        

    inp_size = parse_int_with_cue(
        os.path.join(dory_network_dir, "inc", "main_old.c"),
        "size_t l2_input_size = "
    )
    inp_size_old = parse_int_with_cue(
        os.path.join(dory_network_dir, "src", "main.c"),
        "define NET_INPUT_SIZE "
    )
    
    with open(os.path.join(dory_network_dir, "src", "main.c"), "r") as f:
        text = f.read()
    with open(os.path.join(dory_network_dir, "src", "main.c"), "w") as f:
        cue = "define L2_INPUT_SIZE "
        text = text.replace(cue + str(l2_size_old), cue + str(l2_size))
        print("Replace : \"" + cue + str(l2_size_old) + "\"  ==>  \"" + cue + str(l2_size) + "\"")
        cue = "define NET_INPUT_SIZE "
        text = text.replace(cue + str(inp_size_old), cue + str(inp_size))
        print("Replace : \"" + cue + str(inp_size_old) + "\"  ==>  \"" + cue + str(inp_size) + "\"")
        f.write(text)



if __name__=="__main__":
    main()
