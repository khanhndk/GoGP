#include "stdafx.h"

void report::start()
{
	timer = clock();
}

double report::stop()
{
	seconds += (clock() - timer) / (double)CLOCKS_PER_SEC;
	return get_elapse();
}

void report::reset()
{
	seconds = 0;
}

double report::get_elapse()
{
	return seconds;
}

void report::report_predict(mydouble * y_test, mydouble * y_pred, int N)
{
	r_predict->model = model;
	r_predict->report(y_test, y_pred, N);
}

void report::begin_write_report(char* filename, std::ofstream & fp)
{
	fp = std::ofstream(filename, std::ofstream::out | std::ofstream::app);
}

void report::end_write_report(std::ofstream & fp)
{
	fp << std::endl;
	fp.close();
}

void report::write_reportid(std::ofstream & fp)
{
	svm_parameter param = model->param;
	if (strlen(param.testid) > 0)
		fp << "testid: " << param.testid << "\tcrossid: " << param.crossid << "\trunid: " << param.runid << "\t";
}

void report::write_indicators(std::ofstream & fp)
{
	fp << "acc_oc: " << (r_predict->accuracy_avg) << "\t";
	fp << "acc_detail: ";
	for (int m = 0; m < model->index_label->size(); m++)
		fp << r_predict->accuracy[m] << "|";
	fp << "\t";
	fp << "train: " << train_time << "\ttest: " << predict_time;
	fp << "\t";

	SAVEVAR(fp, fscore, r_predict->f1_avg);
	SAVEVAR(fp, precision, r_predict->precision_avg);
	SAVEVAR(fp, npv, r_predict->precision[0]);
}

void report::print_report()
{
	printf("Acc Total: %f\n", r_predict->accuracy_total);
	printf("Acc: %f; Prec: %f\n", r_predict->accuracy_avg, r_predict->precision_avg);
	printf("Cls in detail: ");
	for (int m = 0; m < model->index_label->size(); m++)
		printf("%d ", r_predict->true_pop[m]);
	printf("\n");
	printf("Acc in detail: ");
	for (int m = 0; m < model->index_label->size(); m++)
		printf("%f ", r_predict->accuracy[m]);
	printf("\n");
	printf("Train: %f; Test: %f\n", train_time, predict_time);
}
