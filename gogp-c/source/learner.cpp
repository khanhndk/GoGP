#include "stdafx.h"

report* svm_train(svm_problem * prob, const svm_parameter * param)
{
	switch (param->svm_type)
	{
	case GOGP:
		return solve_gogp(prob, param);
	default:
		break;
	}
	SHOWERROR("invalid svm_type")
	return NULL;
}

void svm_predict(report * report, const svm_problem * test_prob, mydouble*& predict)
{
	switch (report->model->param.svm_type)
	{
	case GOGP:
		gogp_predict(report, test_prob, predict);
		break;
	default:
		SHOWERROR("predict not defined");
	}
}
