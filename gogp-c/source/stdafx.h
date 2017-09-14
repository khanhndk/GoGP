// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define FLOAT_MODE
#ifdef FLOAT_MODE
#define mydouble float
#define MYMAXDBL FLT_MAX
#endif

#define SHOWERROR(name) {std::cout << name; std::getchar();}
#define SHOWVAR(name) {std::cout << #name << ": " << name << std::endl;}

#define SAVEVAR(f, name) f << #name << ": " << name << "\t";
#define SAVEVAR(f, name, expr) f << #name << ": " << expr << "\t";

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

#include <iostream>
#include <fstream>

#include <vector>
#include <map>

#include <cstdio>
#include <ctime>
#include <random>

#include "mysimd.h"
#include "util.h"
#include "cache.h"
#include "q_matrix.h"

#include "svm_node.h"
#include "svm_parameter.h"
#include "svm_problem.h"

#include "alg_type.h"
#include "kernel_type.h"

#include "kernel.h"
#include "nolabel_q.h"

#include "svm_model.h"

#include "report_predict.h"
#include "report_regression.h"
#include "report.h"

#include "learner.h"
#include "gogp.h"


// TODO: reference additional headers your program requires here
