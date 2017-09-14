#pragma once

//just for float; I will develop for double
inline mydouble dot_vector(mydouble * x, mydouble * y, size_t size)
{
	const size_t N = size;
	const mydouble* p1 = x;
	const mydouble* p2 = y;

	__m128 mmSum = _mm_setzero_ps();
	size_t i = 0;
	size_t maxloop = ROUND_DOWN(N, 4);

	for (; i < maxloop; i += 4)
		mmSum = _mm_add_ps(mmSum, _mm_mul_ps(_mm_loadu_ps(p1 + i), _mm_loadu_ps(p2 + i)));

	// add up single values until all elements are covered
	for (; i < N; i++)
		mmSum = _mm_add_ss(mmSum, _mm_mul_ss(_mm_load_ss(p1 + i), _mm_load_ss(p2 + i)));

	mmSum = _mm_hadd_ps(mmSum, mmSum);
	mmSum = _mm_hadd_ps(mmSum, mmSum);
	return _mm_cvtss_f32(mmSum);
}
