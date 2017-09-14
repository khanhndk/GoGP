#pragma once

struct report
{
private:
	clock_t timer;
	double seconds;

public:
	svm_model* model;
	report_predict* r_predict;

	double train_time;
	double predict_time;

	std::vector<double> mistake_rate;
	std::vector<double>* mistake_class_rate; //detail for each class

	void start();
	double stop();
	void reset();
	double get_elapse();
	void report_predict(mydouble* y_test, mydouble* y_pred, int N);
	virtual void begin_write_report(char* filename, std::ofstream& fp);
	virtual void end_write_report(std::ofstream& fp);
	virtual void write_reportid(std::ofstream& fp);
	virtual void write_indicators(std::ofstream& fp);
	virtual void write_report(char* filename) 
	{
		std::ofstream fp;
		begin_write_report(filename, fp);
		write_reportid(fp);
		write_indicators(fp);
		model->write_report(fp);
		fp << *(model->param.arg);
		end_write_report(fp);
	};
	virtual void write_report_online(char* filename)
	{
		svm_parameter param = model->param;
		int n_class = model->index_label->size();

		std::ofstream fp(filename, std::ofstream::out | std::ofstream::app);
		if (strlen(param.testid) > 0)
			fp << "testid: " << param.testid << "\tcrossid: " << param.crossid << "\trunid: " << param.runid << "\t";

		if ((n_class == 2) && (mistake_class_rate != NULL))
		{
			mydouble neg_rate = mistake_class_rate[0][mistake_class_rate[0].size() - 1];
			mydouble pos_rate = mistake_class_rate[1][mistake_class_rate[1].size() - 1];
			fp << "mistake_rate_oc: " << (pos_rate + neg_rate) / 2 << "\t";
		}
		fp << "mistake_rate: " << mistake_rate[mistake_rate.size() - 1] << "\t";

		if (mistake_class_rate != NULL)
		{
			for (int c = 0; c < n_class; c++)
				fp << "mistake_rate" + std::to_string(c) + ": " << mistake_class_rate[c][mistake_class_rate[c].size() - 1] << "\t";
		}
		fp << "train: " << train_time << "\t";

		model->write_report(fp);

		fp << (*param.arg);

		fp << "mistake_rate_detail: ";
		for (int m = 0; m < mistake_rate.size(); m++)
			fp << mistake_rate[m] << "|";
		fp << "\t";
		if (mistake_class_rate != NULL)
		{
			for (int c = 0; c < n_class; c++)
			{
				fp << "mistake_rate" + std::to_string(c) + ": ";
				for (int m = 0; m < mistake_class_rate[c].size(); m++)
					fp << mistake_class_rate[c][m] << "|";
				fp << "\t";
			}
		}
		fp << "\n";

		fp.close();
	}
	virtual void print_report();
};