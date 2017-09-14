#include "stdafx.h"

report_regression::report_regression()
{
}

report_regression::report_regression(report_predict & parent)
{
	model = parent.model;
}

double report_regression::report(mydouble * y_test, mydouble * y_pred, int N)
{
	mydouble sum_square_error = 0;
	for (int n = 0; n < N; n++)
	{
		mydouble error = y_test[n] - y_pred[n];
		sum_square_error += error * error;
	}
	mse = sum_square_error / N;
	rmse = sqrt(mse);

	return sum_square_error / N;
}
