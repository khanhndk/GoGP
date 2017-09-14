#pragma once

#include <Eigen/Dense>

typedef Eigen::Matrix<mydouble, -1, -1, Eigen::RowMajor> MyMatrixR;
typedef Eigen::Matrix<mydouble, -1, 1> MyVector; //column vector
typedef Eigen::Matrix<mydouble, 1, -1> MyVectorR; //row vector

struct svm_problem
{
	int l;
	mydouble *y;
	struct svm_node **x;
	int max_index;

	svm_problem() {}
	svm_problem(int size, mydouble* _y, svm_node** _x, int _maxindex)
	{
		l = size; y = _y; x = _x; max_index = _maxindex;
	}

	void copy(svm_problem& prob)
	{
		prob.l = this->l;
		prob.max_index = this->max_index;
		prob.y = this->y;
		prob.x = this->x;
	}
};

struct svm_problem_eigen : svm_problem
{
	MyVector* mat_x;
	MyVector* vec_y;
};

struct svm_problem_matrix : svm_problem
{
	mydouble** mat_x;
	mydouble* vec_y;
};

void read_problem(const char *filename, svm_parameter& param, svm_problem& prob);
void write_problem(const char *filename, svm_problem& prob);

// big class -> 1 & small -> -1 //index 0:-1 & 1:1
svm_problem* prob_formalise_unbal_bin(const svm_problem& prob, std::vector<int>* &class_name, std::map<int, int>* &class_index,
	int& switch_label);

svm_problem_eigen* prob_full_eigen(svm_problem& prob, bool extend_dim);

svm_problem_matrix* prob_full_matrix(svm_problem& prob, bool extend_dim, int padding);

void prob_swap(svm_problem& prob, int i, int j);