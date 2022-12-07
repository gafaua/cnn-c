#pragma once

#include <stdio.h>  
#include "network.h"

void save_newtork(Network* net, char* filename);
Network* read_newtork(char* filename, int with_gradient);