#pragma once

struct report_regression : report_predict
{
public:
	svm_model* model;
	double rmse;
	double mse;
	report_regression();
	report_regression(report_predict& parent);

	double report(mydouble* y_test, mydouble* y_pred, int N) override;
};