
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "svm_problem.h"

static int max_line_len = 1024;
static char* line;

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

static char* readline(FILE *input)
{
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

void read_problem(const char *filename, svm_parameter& param, svm_problem& prob)
{
	svm_node *x_space;

	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		getchar();
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	while (readline(fp) != NULL)
	{
		char *p = strtok(line, " \t"); // label

									   // features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(mydouble, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = 0;
	j = 0;
	for (i = 0; i<prob.l; i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t");
		if (strcmp(label, "?") == 0)
		{
			prob.y[i] = INT_MAX;
		}
		else
		{
			prob.y[i] = strtod(label, &endptr);
			if (endptr == label)
				exit_input_error(i + 1);
		}

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

	prob.max_index = max_index;

	if (param.kernel_type == PRECOMPUTED)
		for (i = 0; i<prob.l; i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}


	fclose(fp);
}

void write_problem(const char *filename, svm_problem& prob)
{
	FILE *fp = fopen(filename, "w");
	for (int i = 0; i < prob.l; i++)
	{
		fprintf(fp, "%f", prob.y[i]);
		int j = 0;
		while (prob.x[i][j].index != -1)
		{
			fprintf(fp, " %d:%f", prob.x[i][j].index, prob.x[i][j].value);
			j++;
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

svm_problem * prob_formalise_unbal_bin(const svm_problem& prob, std::vector<int>*& class_name, std::map<int, int>*& class_index, int& swith_label)
{
	swith_label = 0;
	int N = prob.l;
	class_name = new std::vector<int>();
	for (int i = 0; i < N; i++)
		if (std::find(class_name->begin(), class_name->end(), (int)prob.y[i]) == class_name->end())
			class_name->push_back((int)prob.y[i]);

	if (class_name->size() > 2)
	{
		SHOWERROR("not binary problem");
		return NULL;
	}

	std::sort(class_name->begin(), class_name->end());
	int count[2]; count[0] = count[1] = 0;
	for (int i = 0; i < N; i++)
	{
		if (prob.y[i] == (*class_name)[0])
			count[0]++;
		else
			count[1]++;
	}


	class_index = new std::map<int, int>();
	if (count[0] > count[1])
	{
		class_index->insert(std::pair<int, int>((*class_name)[0], 1));
		class_index->insert(std::pair<int, int>((*class_name)[1], 0));
		std::swap((*class_name)[0], (*class_name)[1]);
		swith_label = 1;
	}
	else
	{
		class_index->insert(std::pair<int, int>((*class_name)[0], 0));
		class_index->insert(std::pair<int, int>((*class_name)[1], 1));
	}

	mydouble* y = new mydouble[N];
	for (int i = 0; i < N; i++)
		y[i] = (class_index->find((int)prob.y[i])->second == 0) ? -1 : 1;

	svm_problem* problem = new svm_problem();
	problem->l = N;
	problem->max_index = prob.max_index;
	problem->x = prob.x;
	problem->y = y;

	return problem;
}

svm_problem_eigen * prob_full_eigen(svm_problem & prob, bool extend_dim)
{
	svm_problem_eigen* result = new svm_problem_eigen();
	prob.copy(*result);
	int D = prob.max_index + ((extend_dim) ? 1 : 0);
	int N = prob.l;
	result->mat_x = new MyVector[N];
	for (int n = 0; n < N; n++)
		result->mat_x[n] = MyVector(D);
	if (result->mat_x[0].rows() != D) SHOWERROR("Error init mat rows");

	for (int n = 0; n < N; n++)
		for (int d = 0; d < D; d++)
		{
			if ((d == D - 1) && (extend_dim))
				result->mat_x[n](d) = 1;
			else
				result->mat_x[n](d) = 0;
		}

	for (int n = 0; n < N; n++)
	{
		svm_node* x = prob.x[n];
		for (int i = 0; x[i].index != -1; i++)
			result->mat_x[n](x[i].index - 1) = x[i].value;
	}

	result->vec_y = new MyVector(N);
	for (int n = 0; n < N; n++)
		(*(result->vec_y))(n) = prob.y[n];

	return result;
}

svm_problem_matrix * prob_full_matrix(svm_problem & prob, bool extend_dim, int padding)
{
	svm_problem_matrix* result = new svm_problem_matrix();
	prob.copy(*result);
	int max_index = prob.max_index;
	int D = max_index + ((extend_dim) ? 1 : 0) + padding;
	int N = prob.l;
	result->mat_x = new mydouble*[N];
	for (int n = 0; n < N; n++)
		result->mat_x[n] = new mydouble[D];

	for (int n = 0; n < N; n++)
		for (int d = 0; d < D; d++)
		{
			if ((d == max_index) && (extend_dim))
				result->mat_x[n][d] = 1;
			else
				result->mat_x[n][d] = 0;
		}

	for (int n = 0; n < N; n++)
	{
		svm_node* x = prob.x[n];
		for (int i = 0; x[i].index != -1; i++)
			result->mat_x[n][x[i].index - 1] = x[i].value;
	}

	result->vec_y = new mydouble[N];
	for (int n = 0; n < N; n++)
		result->vec_y[n] = prob.y[n];

	return result;
}

void prob_swap(svm_problem& prob, int i, int j)
{
	svm_node* tmpNode; double tmpLabel;
	tmpNode = prob.x[i]; tmpLabel = prob.y[i];
	prob.x[i] = prob.x[j]; prob.y[i] = prob.y[j];
	prob.x[j] = tmpNode; prob.y[j] = tmpLabel;
}
