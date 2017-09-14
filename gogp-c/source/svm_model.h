#pragma once

struct svm_model
{
	//report report;

	svm_parameter param;	// parameter
	svm_problem* prob;

	std::vector<int>* index_label;
	std::map<int, int>* label_index;

	virtual void write_report(std::ofstream& fp) {};
	virtual void write_report_online(char* filename) {};

	mydouble objective;
	mydouble g_error;
	mydouble e_error;
};

struct svm_plane_model : svm_model
{
	mydouble* a;
	int l_a;
};