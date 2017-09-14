#pragma once

enum {
	C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR, LSVM, LOCSVM, CSVM, HMS_SVDD, MMEB1, MMEB2, SVDD = 12, MEB1 = 14, MEB2,
	AMMNorm1, MSVDD, MSVDD_SGD, SGDBoost, SGDLMOCSVM, ApproxOCSVM, SApproxOCSVM, ApproxLMOCSVM1, ApproxLMOCSVM2, ApproxLMOCSVM3,
	gUS3VM,
	C_SVC_BIN = 100,
	SAMMEX1 = 27,
	SAMMEX10 = 270,
	SAMMEX11 = 271,
	SAMMEX12 = 272,
	LMOCSVM = 230, /*large marchine one-class svm */
	MOCSVM = 28, /*modified ocsvm to deal with negative data points */
	ApproxMOCSVM1 = 281,
	ApproxMOCSVM2 = 282,
	ApproxMOCSVM3 = 283,
	ApproxMOCSVM4 = 284,
	ApproxMOCSVM5 = 285,
	ApproxMOCSVM6 = 286,
	MOCSVM_SGD = 29,
	ApproxMOCSVM_SGD1 = 291,
	ApproxMOCSVM_SGD2 = 292,
	ApproxMOCSVM_SGD3 = 293,
	ApproxMOCSVM_SGD4 = 294,
	ApproxMOCSVM_SGD5 = 295,
	ApproxMOCSVM_SGD6 = 296,
	ApproxMOCSVM_SGD7 = 297, //online 
	MCSVM_CS = 30,
	MCMPSVM_CS = 301,
	MCMPSVM_CS_DA = 302, //SAMM use liblinear
	OFOC = 31,
	OFOC_H = 310,
	OFOC_SH = 311,
	OFOC_L = 312,
	OFOC_FF = 32, //OFOC FastFood
	OFOC_FF_H = 320,
	OFOC_FF_SH = 321,
	OFOC_FF_L = 322,
	OFOC_ONLINE = 33,
	OFOC_ONLINE_H = 330,
	OFOC_ONLINE_SH = 331,
	OFOC_ONLINE_L = 332,
	GOGP = 36,
};	/* svm_type */
