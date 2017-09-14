#pragma once

#include "stdafx.h"

kernel::kernel(int l, svm_node * const * x_, const svm_parameter& param)
	:kernel_type(param.kernel_type), degree(param.degree),
	gamma(param.gamma), coef0(param.coef0), rbf_eps(20 / param.gamma)
{

	switch (kernel_type)
	{
	case LINEAR:
		kernel_function = &kernel::kernel_linear;
		break;
	case POLY:
		kernel_function = &kernel::kernel_poly;
		break;
	case RBF:
		kernel_function = &kernel::kernel_rbf;
		break;
	case SIGMOID:
		kernel_function = &kernel::kernel_sigmoid;
		break;
	case PRECOMPUTED:
		kernel_function = &kernel::kernel_precomputed;
		break;
	}

	clone(x, x_, l);

	if ((kernel_type == RBF))
	{
		x_square = new double[l];
		for (int i = 0; i<l; i++)
			x_square[i] = dot(x[i], x[i]);
	}
	else
		x_square = 0;
}

kernel::~kernel()
{
	delete[] x;
	delete[] x_square;
}

double kernel::dot(const svm_node *px, const svm_node *py)
{
	//const svm_node* x = px;
	//const svm_node* y = py;
	//int max_index = 2;
	////double* tmp = new double[max_index];
	////memset(tmp, 0, max_index*sizeof(*tmp));
	////int cx = 0;
	////while(px->index != -1) { cx++; px++; }
	////int cy = 0;
	////while(py->index != -1) { cy++; py++; }

	////#pragma omp for
	////for(int i = 0; i < cx; i++)
	////	tmp[x[i].index-1] = x[i].value;

	//double sum = 0;
	////#pragma omp for
	////for(int i = 0; i < cy; i++)
	////{
	////	sum += tmp[y[i].index-1] * y[i].value;
	////}

	//return sum;

	double sum = 0;
	while (px->index != -1 && py->index != -1)
	{
		if (px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if (px->index > py->index)
				++py;
			else
				++px;
		}
	}
	return sum;
}

double kernel::k_function(const svm_node *x, const svm_node *y,
	const svm_parameter& param, int kernel_extend_dimension, bool approx_calc, mydouble aprox_rbf_value)
{
	switch (param.kernel_type)
	{
	case LINEAR:
		return dot(x, y) + kernel_extend_dimension;
	case POLY:
		return powi(param.gamma*dot(x, y) + param.coef0, param.degree) + kernel_extend_dimension;
	case RBF:
	{
		double sum = 0;
		while (x->index != -1 && y->index != -1)
		{
			if (x->index == y->index)
			{
				double d = x->value - y->value;
				sum += d*d;
				++x;
				++y;
			}
			else
			{
				if (x->index > y->index)
				{
					sum += y->value * y->value;
					++y;
				}
				else
				{
					sum += x->value * x->value;
					++x;
				}
			}
		}

		while (x->index != -1)
		{
			sum += x->value * x->value;
			++x;
		}

		while (y->index != -1)
		{
			sum += y->value * y->value;
			++y;
		}

		double value_exp = -param.gamma*sum;
		if ((approx_calc) && (value_exp < aprox_rbf_value))
			return kernel_extend_dimension;

		return exp(value_exp) + kernel_extend_dimension;
	}
	case SIGMOID:
		return tanh(param.gamma*dot(x, y) + param.coef0) + kernel_extend_dimension;
	case PRECOMPUTED:  //x: test (validation), y: SV
		return x[(int)(y->value)].value;
	default:
		return 0;  // Unreachable 
	}
}