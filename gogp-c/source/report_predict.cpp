#include "stdafx.h"

report_predict::report_predict()
{

}

double report_predict::report(mydouble * y_test, mydouble * y_pred, int N)
{
	std::vector<int>* index_label = model->index_label;
	std::map<int, int>* label_index = model->label_index;

	int M = model->index_label->size();
	int** matrix = new int*[M];
	for (int m = 0; m < M; m++)
	{
		matrix[m] = new int[M];
		for (int j = 0; j < M; j++)
			matrix[m][j] = 0;
	}

	for (int n = 0; n < N; n++)
	{
		int i = (int)(label_index->find(y_test[n])->second); //true
		int j = (int)(label_index->find(y_pred[n])->second); //predict
		matrix[i][j]++;
	}

	cf_matrix = matrix;
	true_pop = new int[M];
	pred_pop = new int[M];
	accuracy = new double[M];
	precision = new double[M];
	recall = new double[M];
	npv = new double[M];
	f1 = new double[M];
	accuracy_total = 0;
	accuracy_avg = 0;
	precision_avg = 0;
	recall_avg = 0;
	npv_avg = 0;
	f1_avg = 0;

	for (int m = 0; m < M; m++)
	{
		true_pop[m] = 0;
		pred_pop[m] = 0;
		for (int j = 0; j < M; j++)
		{
			true_pop[m] += matrix[m][j];
			pred_pop[m] += matrix[j][m];
		}
	}

	for (int m = 0; m < M; m++)
	{
		accuracy[m] = (double)matrix[m][m] / true_pop[m];
		precision[m] = (double)matrix[m][m] / pred_pop[m];
		recall[m] = (double)matrix[m][m] / true_pop[m]; // same as accuracy
		f1[m] = 2 * precision[m] * recall[m] / (precision[m] + recall[m]);
		accuracy_total += matrix[m][m];
		accuracy_avg += accuracy[m];
		precision_avg += precision[m];
		recall_avg += recall[m];
		f1_avg += f1[m];
	}
	accuracy_total = accuracy_total / N;
	accuracy_avg = accuracy_avg / M;
	precision_avg = precision_avg / M;
	recall_avg = recall_avg / M;
	f1_avg = f1_avg / M;

	return accuracy_total;
}
