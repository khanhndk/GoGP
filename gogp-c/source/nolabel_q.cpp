#include "stdafx.h"

nolabel_q::nolabel_q(const svm_problem & prob, const svm_parameter & param)
	:kernel(prob.l, prob.x, param)
{
	mycache = new cache(prob.l, (long int)(param.cache_size*(1 << 20)));
	QD = new mydouble[prob.l];
	for (int i = 0; i < prob.l; i++)
		QD[i] = (mydouble)(this->*kernel_function)(i, i);
}

mydouble* nolabel_q::get_q(int i, int len) const
{
	mydouble *data = NULL;
	int start, j;
	if (i < 0)
	{
		j = 2;
	}
	if ((start = mycache->get_data(i, &data, len)) < len)
	{
		//int N = len - start;
		//#pragma omp parallel firstprivate(data)
		//{
		//	int ithread = omp_get_thread_num();
		//	int nthreads = omp_get_num_threads();
		//	int omp_start = ithread * N/nthreads;
		//	int omp_finish = (ithread + 1) * N/nthreads;
		//	double* tmp = new double[omp_finish - omp_start];

		//	for (int j = omp_start; j < omp_finish; j++) {
		//		tmp[j-omp_start] = (mydouble)((this->*kernel_function)(i, start+j));
		//	}
		//	for (int j = omp_start; j < omp_finish; j++) {
		//		data[start+j] = tmp[j-omp_start];
		//	}
		//}
		//#pragma omp parallel for firstprivate(data)
		//#pragma loop(hint_parallel(8))
		for (j = start; j < len; j++)
			data[j] = (mydouble)((this->*kernel_function)(i, j));
	}
	return data;
}

mydouble* nolabel_q::get_qd() const
{
	return QD;
}

void nolabel_q::swap_index(int i, int j) const
{
	mycache->swap_index(i, j);
	kernel::swap_index(i, j);
	std::swap(QD[i], QD[j]);
}
nolabel_q::~nolabel_q()
{
	delete mycache;
	delete[] QD;
}