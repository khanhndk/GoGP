#include "stdafx.h"


float accumulate(const std::vector<float>& v)
{
	// copy the length of v and a pointer to the data onto the local stack
	const size_t N = v.size();
	const float* p = (N > 0) ? &v.front() : NULL;

	__m128 mmSum = _mm_setzero_ps();
	size_t i = 0;

	// unrolled loop that adds up 4 elements at a time
	for (; i < ROUND_DOWN(N, 4); i += 4)
	{
		mmSum = _mm_add_ps(mmSum, _mm_loadu_ps(p + i));
	}

	// add up single values until all elements are covered
	for (; i < N; i++)
	{
		mmSum = _mm_add_ss(mmSum, _mm_load_ss(p + i));
	}

	// add up the four float values from mmSum into a single value and return
	mmSum = _mm_hadd_ps(mmSum, mmSum);
	mmSum = _mm_hadd_ps(mmSum, mmSum);
	return _mm_cvtss_f32(mmSum);
}

double accumulate(double* a, double* b, size_t size)
{
	// copy the length of v and a pointer to the data onto the local stack
	const size_t N = size;
	const double* p1 = a;
	const double* p2 = b;

	__m128d mmSum = _mm_setzero_pd();

	size_t i = 0;

	// unrolled loop that adds up 4 elements at a time
	for (; i < ROUND_DOWN(N, 2); i += 2)
	{
		mmSum = _mm_add_pd(mmSum, _mm_mul_pd(_mm_loadu_pd(p1 + i), _mm_loadu_pd(p2 + i)));
	}

	//// add up single values until all elements are covered
	//for(; i < N; i++)
	//{
	// mmSum = _mm_add_sd(mmSum, _mm_load_sd(p + i));
	//}

	// add up the four float values from mmSum into a single value and return
	mmSum = _mm_hadd_pd(mmSum, mmSum);
	//mmSum = _mm_hadd_pd(mmSum, mmSum);
	return _mm_cvtsd_f64(mmSum);
	//return _mm_cvtss_f32(mmSum);

	
}

