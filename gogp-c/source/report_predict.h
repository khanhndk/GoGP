#pragma once

struct report_predict
{
public:
	svm_model* model;
	int** cf_matrix;
	int* true_pop;
	int* pred_pop;
	double* accuracy;
	double* precision;
	double* recall;
	double* npv;
	double* f1;
	double accuracy_total; //balance acc
	double accuracy_avg; //unbalance acc
	double precision_avg;
	double recall_avg;
	double npv_avg;
	double f1_avg;

	report_predict();

	virtual double report(mydouble* y_test, mydouble* y_pred, int N);
};