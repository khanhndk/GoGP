#pragma once

template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst, (void *)src, sizeof(T)*n);
}

class kernel : public q_matrix {
public:
	bool approx_calc;
	double aprox_error;
	double aprox_rbf_value;

	int kernel_extend_dimension;
	kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~kernel();

	static double k_function(const svm_node *x, const svm_node *y, const svm_parameter& param,
		int kernel_extend_dimension = 0, bool approx_calc = false, mydouble aprox_rbf_value = 0);
	virtual mydouble *get_q(int column, int len) const = 0;
	virtual mydouble *get_qd() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		std::swap(x[i], x[j]);
		if (x_square) std::swap(x_square[i], x_square[j]);
	}


	double (kernel::*kernel_function)(int i, int j) const;
	const int kernel_type;
	double *x_square;
	const svm_node **x;

	static double dot(const svm_node *px, const svm_node *py);

	const double rbf_eps;

	// svm_parameter

	const int degree;
	const double gamma;
	const double coef0;

	virtual double kernel_linear(int i, int j) const
	{
		return dot(x[i], x[j]) + kernel_extend_dimension;
	}
	virtual double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i], x[j]) + coef0, degree) + kernel_extend_dimension;
	}
	virtual double kernel_rbf(int i, int j) const
	{
		double dsquare = (x_square[i] + x_square[j] - 2 * dot(x[i], x[j]));
		double value_exp = -gamma*dsquare;
		if ((approx_calc) && (value_exp < aprox_rbf_value))
			return kernel_extend_dimension;
		return exp(value_exp) + kernel_extend_dimension;
	}
	virtual double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i], x[j]) + coef0) + kernel_extend_dimension;
	}
	virtual double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};
