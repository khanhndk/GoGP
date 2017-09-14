#include "stdafx.h"
#include <chrono>
#include <queue>

report * solve_gogp_knn(svm_problem * prob, const svm_parameter * param)
{
	//INPUT: gamma, lbd, theta, k
	gogp_report* report = new gogp_report();
	report->r_predict = new report_regression();
	gogp_model* model = new gogp_model();
	report->model = model;
	model->prob = prob;

	int N = prob->l;
	int D = prob->max_index; //are you sure?
	mydouble gamma = param->gamma;
	mydouble lbd = param->lbd;
	mydouble theta = param->theta;
	int k = param->k;
	mydouble sigma2 = N * lbd / 2.0;
	mydouble ymax = -MYMAXDBL;
	for (int n = 0; n < N; n++)
		if (ymax < prob->y[n])
			ymax = prob->y[n];
	int info_step = param->info_step;


	printf("N=%d, D=%d; k=%d, lbd=%f; gamma=%f; theta:%f\n", N, D, k, lbd, gamma, theta);
	SHOWVAR(ymax);

	//just in case dataset is not random
	std::vector<int> n_index;
	for (int n = 0; n < N; n++)
		n_index.push_back(n);
	#ifdef _DEBUG
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(n_index.begin(), n_index.end(), std::default_random_engine(seed));
	#endif

	report->reset();
	report->start();

	mydouble scale_ball = ymax / sqrt(lbd);
	svm_node** X = prob->x;

	std::vector<std::vector<mydouble>*> K_sigma2;
	std::vector<mydouble>* K_tmp = new std::vector<mydouble>();
	K_tmp->push_back(1.0 + sigma2); //assume RBF
	K_sigma2.push_back(K_tmp);

	int w_l = 1;
	int* w_index = new int[N]; //just in case ...
	w_index[0] = n_index[0];
	mydouble* w = new mydouble[N]; //just in case ...
	w[0] = -2.0 * (0.0 - prob->y[n_index[0]]) / (lbd * 1);

	mydouble* wbar = new mydouble[N]; //just in case ...
	wbar[0] = w[0];

	mydouble* Kn = new mydouble[N]; //just in case ...
	//no need init

	mydouble* dist2 = new mydouble[N]; //just in case ...
	//no need init

	int* k_index_knn = new int[k]; //no need init
	MyMatrixR K_knn(k, k); assert(K_knn.cols() == k); assert(K_knn.rows() == k);
	MyVector kn_knn(k); assert(kn_knn.rows() == k);
	MyVector d_project(k); assert(d_project.rows());

	mydouble sum_square_error = 0;
	mydouble cur_rmse;

	for (int n = 1; n < N; n++)
	{
		int t = n + 1;
		int nt = n_index[n];
		svm_node* xnt = prob->x[nt];
		mydouble ynt = prob->y[nt];

		#pragma region CalcDist
		//are you sure we need to calc all?
		for (int wi = 0; wi < w_l; wi++)
		{
			mydouble dist_tmp = 0;
			const svm_node* xnt_tmp = xnt;
			const svm_node* x_tmp = X[w_index[wi]];
			for (; xnt_tmp->index != -1 && x_tmp->index != -1;)
			{
				if (xnt_tmp->index == x_tmp->index)
				{
					mydouble tmp = xnt_tmp->value - x_tmp->value;
					dist_tmp += tmp * tmp;
					xnt_tmp++;
					x_tmp++;
				}
				else
				{
					if (xnt_tmp->index < x_tmp->index)
					{
						dist_tmp += xnt_tmp->value * xnt_tmp->value;
						xnt_tmp++;
					}
					else
					{
						dist_tmp += x_tmp->value * x_tmp->value;
						x_tmp++;
					}
				}
			}
			while (xnt_tmp->index != -1)
			{
				dist_tmp += xnt_tmp->value * xnt_tmp->value;
				++xnt_tmp;
			}
			while (x_tmp->index != -1)
			{
				dist_tmp += x_tmp->value * x_tmp->value;
				++x_tmp;
			}
			//mydouble tmp1 = calc_dist2(xnt, prob->x[w0_index[wi]]);
			//mydouble tmp2 = calc_dist2(xnt, X[w0_index[wi]]);
			//assert(tmp1 == dist_tmp);
			assert(dist_tmp >= 0);
			dist2[wi] = dist_tmp;
		}
		#pragma endregion

		for (int wi = 0; wi < w_l; wi++)
			Kn[wi] = exp(-gamma * dist2[wi]);

		//predict
		mydouble y_pred_t = 0;
		for(int wi = 0; wi < w_l; wi++)
			y_pred_t += w[wi] * Kn[wi];
		mydouble alpha_t = y_pred_t - ynt;
		sum_square_error += alpha_t * alpha_t;
		cur_rmse = sqrt(sum_square_error / n);
		if (!(n % info_step))
			printf("n:%d; core:%d; rmse: %f\n", n, w_l, cur_rmse);

		//calc projection
		std::priority_queue<std::pair<mydouble, int>> knn_queue;
		for (int wi = 0; wi < w_l; wi++)
		{
			if (knn_queue.size() < k)
				knn_queue.push(std::pair<mydouble, int>(dist2[wi], wi));
			else if (knn_queue.top().first > dist2[wi])
			{
				knn_queue.push(std::pair<mydouble, int>(dist2[wi], wi));
				knn_queue.pop();
			}
		}
		int knn_queue_size = knn_queue.size();
		for (int ki = 0; ki < knn_queue_size; ki++)
		{
			k_index_knn[ki] = knn_queue.top().second;
			knn_queue.pop();
		}
		for (int kj1 = 0; kj1 < knn_queue_size; kj1++)
		{
			int k_index = k_index_knn[kj1];
			kn_knn(kj1) = exp(-gamma * dist2[k_index]);
			std::vector<mydouble>* K_tmp = K_sigma2[k_index];
			for (int kj2 = 0; kj2 < knn_queue_size; kj2++)
			{
				int kj2_index = k_index_knn[kj2];
				//if ((*K_tmp)[kj2_index] < -1) //kernel >= 0
				//	(*K_tmp)[kj2_index] = exp(-gamma *
				//		calc_dist2(prob->x[w0_index[k_index]], prob->x[w0_index[kj2_index]]));
				//mydouble tmp1 = exp(-gamma *
				//	calc_dist2(prob->x[w0_index[k_index]], prob->x[w0_index[kj2_index]]));
				//assert(tmp1 == (*K_tmp)[kj2_index]);
				K_knn(kj1, kj2) = (*K_tmp)[kj2_index];
			}
		}
		if (knn_queue_size < k)
		{
			K_knn.conservativeResize(knn_queue_size, knn_queue_size);
			kn_knn.conservativeResize(knn_queue_size);
			d_project.resize(knn_queue_size); //no need to keep unchange
		}

		MyVector d_project = (K_knn.inverse() * kn_knn);

		//calc dist
		mydouble dist2 = 1.0 - kn_knn.dot(d_project);

		if (knn_queue_size < k)
		{
			K_knn.conservativeResize(k, k);
			kn_knn.conservativeResize(k);
			d_project.resize(k); //no need to keep unchange
		}

		mydouble scale_w = (t - 1.0) / t;
		//scale
		for(int wi = 0; wi < w_l; wi++)
			w[wi] *= scale_w;

		//update
		if (dist2 > theta)
		{
			w_index[w_l] = nt;
			w[w_l] = -2.0 * alpha_t / (lbd * t);
			wbar[w_l] = 0;
			for (int wi = 0; wi < w_l; wi++)
				K_sigma2[wi]->push_back(Kn[wi]);
			K_tmp = new std::vector<mydouble>();
			for (int wi = 0; wi < w_l; wi++)
				K_tmp->push_back(Kn[wi]);
			K_tmp->push_back(1.0 + sigma2);
			K_sigma2.push_back(K_tmp);
			w_l++;
		}
		else
		{
			for (int ki = 0; ki < knn_queue_size; ki++)
			{
				int k_index = k_index_knn[ki];
				w[k_index] -= (2.0 * alpha_t / (lbd * t)) * w[k_index] * d_project(ki);
			}
		}
		if (lbd < 2)
		{
			mydouble wnorm2 = dot_vector(w, w, w_l);
			//mydouble wnorm2 = 0;
			//for (int wi = 0; wi < w_l; wi++)
			//	wnorm2 += w[wi] * w[wi];

			mydouble scale_project = scale_ball / sqrt(wnorm2);
			if (scale_project < 1)
				for(int wi = 0; wi < w_l; wi++)
					w[wi] *= scale_project;
		}
		for(int wi = 0; wi < w_l; wi++)
			wbar[wi] = wbar[wi] * (t - 1) / t + w[wi] / t;
	}

	report->train_time = report->stop();
	SHOWVAR(w_l);
	SHOWVAR(N);
	model->w = w;
	model->wbar = wbar;
	model->w_index = w_index;
	model->w_l = w_l;
	model->param = *param;
	return report;
}

//speed up version
report * solve_gogp_s(svm_problem * train_prob, const svm_parameter * param)
{
	//std::cout << "Speepup version" << std::endl;
	//INPUT: gamma, lbd, theta, p1 - percent of batch, epoch
	gogp_report* report = new gogp_report();
	report->r_predict = new report_regression();
	gogp_model* model = new gogp_model();
	report->model = model;
	model->prob = train_prob;
	svm_problem_eigen* prob = prob_full_eigen(*train_prob, false);

	int N = prob->l;
	int D = prob->max_index; //are you sure?
	mydouble gamma = param->gamma;
	mydouble lbd = param->lbd;
	mydouble theta = param->theta;
	mydouble sigma2 = N * lbd / 2.0;
	mydouble ymax = -MYMAXDBL;
	for (int n = 0; n < N; n++)
		if (ymax < prob->y[n])
			ymax = prob->y[n];
	int info_step = param->info_step;
	mydouble p1 = param->p1;
	int N_batch = p1 * N;
	int T = param->epoch * N_batch;;

	printf("N=%d, D=%d; lbd=%f; gamma=%f; theta:%f\n", N, D, lbd, gamma, theta);
	//SHOWVAR(ymax);

	//just in case dataset is not random
	std::vector<int> n_index;
	for (int n = 0; n < N; n++)
		n_index.push_back(n);
#ifdef _DEBUG
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(n_index.begin(), n_index.end(), std::default_random_engine(seed));
#endif


	mydouble scale_ball = ymax / sqrt(lbd);
	svm_node** X = prob->x;

	MyMatrixR K_sigma2(1, 1);
	assert(K_sigma2.cols() == 1);
	assert(K_sigma2.rows() == 1);
	K_sigma2(0, 0) = 1.0 + sigma2; //assume RBF
	MyMatrixR Kinv_sigma2(1, 1);
	Kinv_sigma2(0, 0) = 1.0 / K_sigma2(0, 0);

	int w_l = 1;
	int* w_index = new int[N]; //just in case ...
	w_index[0] = n_index[0];
	MyVector w(1);
	w(0) = -2.0 * (0.0 - prob->y[n_index[0]]) / (lbd * 1);
	mydouble wnorm2 = w(0) * w(0); //kernel func = 1

	MyVector wbar(1);
	assert(wbar.rows() == 1);
	wbar(0) = w(0);

	MyVector Kn(1);
	assert(Kn.rows() == 1);
	//no need init

	MyVector d_project(1);
	assert(d_project.rows() == 1);
	//no need init

	std::default_random_engine ugen(time(NULL));
	std::uniform_int_distribution<int> udist(0, N_batch - 1);
	int* pos_core = new int[N];
	for (int n = 0; n < N; n++)
		pos_core[n] = -1;
	for (int t = 1; t <= T; t++)
	{
		int nt = udist(ugen);
		svm_node* xnt = prob->x[nt];
		int ynt = prob->y[nt];

		//predict
		for (int wi = 0; wi < w_l; wi++)
			Kn(wi) = kernel::k_function(xnt, X[w_index[wi]], *param);
		
		mydouble y_pred_t = Kn.dot(w);
		mydouble alpha_t = y_pred_t - ynt;

		//calc projection
		d_project = Kinv_sigma2 * Kn;

		//calc dist
		mydouble dist2 = 1.0 - Kn.dot(d_project); //assume RBF

		mydouble scale_w = (t - 1.0) / t;
		//scale
		w *= scale_w;

		//update
		if (dist2 > theta)
		{
			//printf("dist2:%f;\tt:%d\n", dist2, t);

			int i_new_w = w_l;
			if (pos_core[nt] >= 0)
			{
				i_new_w = pos_core[nt];
				w(i_new_w) += -2.0 * alpha_t / (lbd * t);
			}
			else
			{
				pos_core[nt] = w_l;
				w_index[i_new_w] = nt;
				w.conservativeResize(w_l + 1);
				w(w_l) = -2.0 * alpha_t / (lbd * t);
				wbar.conservativeResize(w_l + 1);
				wbar(w_l) = 0;
				K_sigma2.conservativeResize(w_l + 1, w_l + 1);
				K_sigma2.block(0, w_l, w_l, 1) = Kn;
				K_sigma2.block(w_l, 0, 1, w_l) = Kn.transpose();
				//assert(K_sigma2(w_l, 0) = Kn(0));
				//assert(K_sigma2(0, w_l) = Kn(0));
				K_sigma2(w_l, w_l) = 1.0 + sigma2;
				Kinv_sigma2 = K_sigma2.inverse();
				Kn.resize(w_l + 1); //no need keep unchange
				d_project.resize(w_l + 1); //no need keep unchange
				w_l++;
			}
		}
		else
		{
			w -= (2.0 * alpha_t / (lbd * t)) * w.cwiseProduct(d_project);
		}
		if (lbd < 2)
		{
			MyMatrixR ww = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
			wnorm2 = (ww.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array()).sum();
			if (!isfinite(wnorm2))
				SHOWERROR(ww);
			if (wnorm2 != 0)
			{
				mydouble scale_project = scale_ball / sqrt(wnorm2);
				if (scale_project < 1)
					w *= scale_project;
			}

		}
		wbar = wbar * (t - 1) / t + w / t;
	}

	//just record online time
	report->reset();
	report->start();

	MyMatrixR mat_x(w_l, D);
	for (int wi = 0; wi < w_l; wi++)
		mat_x.row(wi) = (*(prob->mat_x + w_index[wi]));
	MyMatrixR K_mat = K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2;

	MyMatrixR ww_K_sigma2 = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
	ww_K_sigma2 = ww_K_sigma2.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array();

	wnorm2 = ww_K_sigma2.sum();

	mydouble sum_square_error = 0;
	mydouble cur_rmse;
	int n_rmse = 0;
	for (int n = N_batch; n < N; n++)
	{
		int t = n + 1;
		int nt = n_index[n];
		MyVector* xnt = prob->mat_x + nt;
		mydouble ynt = prob->y[nt];

		//predict
		//for (int wi = 0; wi < w_l; wi++)
		//	Kn(wi) = kernel::k_function(xnt, X[w_index[wi]], *param);
		//Kn = (xnt->transpose().replicate(w_l, 1) - mat_x).array().pow(2).rowwise().sum();
		Kn = (mat_x.rowwise() - xnt->transpose()).array().pow(2).rowwise().sum();
		Kn *= -param->gamma;
		Kn = Kn.array().exp();

		mydouble y_pred_t = Kn.dot(w);
		mydouble alpha_t = y_pred_t - ynt;
		sum_square_error += alpha_t * alpha_t;
		cur_rmse = sqrt(sum_square_error / (++n_rmse));
		if (!(n % info_step))
			printf("[%0.2f%%]; core:%d; rmse: %f\n", (float)n / N * 100, w_l, cur_rmse);

		//calc projection
		d_project = Kinv_sigma2 * Kn;

		//calc dist
		mydouble dist2 = 1.0 - Kn.dot(d_project); 
		//assume RBF
		//if (dist2 == 1.0)
		//{
		//	SHOWVAR(d_project);
		//	SHOWVAR(Kn);
		//	SHOWERROR("dist2 == 1");
		//}

		mydouble scale_w = (t - 1.0) / t;
		//scale
		w *= scale_w;
		wnorm2 *= scale_w * scale_w;
		ww_K_sigma2 *= scale_w * scale_w;

		//update
		if (dist2 > theta)
		{
			mydouble modifier = -2.0 * alpha_t / (lbd * t);
			//std::cout << "dist2: " << dist2 << std::endl;
			w_index[w_l] = nt;
			w.conservativeResize(w_l + 1);
			w(w_l) = modifier;
			wbar.conservativeResize(w_l + 1);
			wbar(w_l) = 0;
			K_sigma2.conservativeResize(w_l + 1, w_l + 1);
			K_sigma2.block(0, w_l, w_l, 1) = Kn;
			K_sigma2.block(w_l, 0, 1, w_l) = Kn.transpose();
			//assert(K_sigma2(w_l, 0) = Kn(0));
			//assert(K_sigma2(0, w_l) = Kn(0));
			K_sigma2(w_l, w_l) = 1.0 + sigma2;
			Kinv_sigma2 = K_sigma2.inverse();
			Kn.resize(w_l + 1); //no need keep unchange
			d_project.resize(w_l + 1); //no need keep unchange

			mat_x.conservativeResize(w_l + 1, Eigen::NoChange);
			mat_x.row(w_l) = (*(prob->mat_x + nt));
			w_l++;

			wnorm2 += 2 * modifier * y_pred_t * scale_w + (modifier * modifier);
			ww_K_sigma2 = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
			ww_K_sigma2 = ww_K_sigma2.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array();
			K_mat = K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2;
		}
		else
		{
			MyVector proj_mod = (2.0 * alpha_t / (lbd * t)) * d_project;
			//MyMatrixR proj_mod_rep = proj_mod.replicate(1, w_l);
			//MyMatrixR proj_mod_rep = proj_mod.transpose().replicate(w_l, 1);

			MyMatrixR kron_proj_mod = proj_mod * proj_mod.transpose();
			MyMatrixR kron_proj_c = proj_mod * w.transpose(); 

			ww_K_sigma2.array() += (-kron_proj_c - kron_proj_c.transpose() + kron_proj_mod).array()
									* K_mat.array();
			//ww_K_sigma2 = ww_K_sigma2.array() + (-kron_proj_c - kron_proj_c.transpose() + kron_proj_mod).array()
				//* K_mat.array();
			w -= proj_mod; //w.cwiseProduct(d_project);
			wnorm2 = ww_K_sigma2.sum();
		}
		if (lbd < 2)
		{
			//MyMatrixR ww = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
			//wnorm2 = (ww.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array()).sum();
			if (!isfinite(wnorm2))
				SHOWERROR(ww_K_sigma2);
			//SHOWVAR(wnorm2);
			mydouble scale_project = scale_ball / sqrt(wnorm2);
			if (scale_project < 1)
				w *= scale_project;
		}
		wbar = wbar * (t - 1) / t + w / t;
	}

	report->train_time = report->stop();
	report->rmse_rate = cur_rmse;
	//SHOWVAR(w_l);
	//SHOWVAR(N);
	model->w = new mydouble[w_l];
	model->wbar = new mydouble[w_l];
	for (int wi = 0; wi < w_l; wi++)
	{
		model->w[wi] = w(wi);
		model->wbar[wi] = wbar(wi);
	}
	model->w_index = w_index;
	model->w_l = w_l;
	model->param = *param;
	return report;
}

report * solve_gogp(svm_problem * prob, const svm_parameter * param)
{
	if (param->svm_variant == 1)
		return solve_gogp_knn(prob, param);
	if (param->svm_variant == 2)
		return solve_gogp_s(prob, param);

	//INPUT: gamma, lbd, theta, p1 - percent of batch, epoch
	gogp_report* report = new gogp_report();
	report->r_predict = new report_regression();
	gogp_model* model = new gogp_model();
	report->model = model;
	model->prob = prob;

	int N = prob->l;
	int D = prob->max_index; //are you sure?
	mydouble gamma = param->gamma;
	mydouble lbd = param->lbd;
	mydouble theta = param->theta;
	mydouble sigma2 = N * lbd / 2.0;
	mydouble ymax = -MYMAXDBL;
	for (int n = 0; n < N; n++)
		if (ymax < prob->y[n])
			ymax = prob->y[n];
	int info_step = param->info_step;
	mydouble p1 = param->p1;
	int N_batch = p1 * N;
	int T = param->epoch * N_batch;;

	printf("N=%d, D=%d; lbd=%f; gamma=%f; theta:%f\n", N, D, lbd, gamma, theta);
	SHOWVAR(ymax);

	//just in case dataset is not random
	std::vector<int> n_index;
	for (int n = 0; n < N; n++)
		n_index.push_back(n);
#ifdef _DEBUG
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(n_index.begin(), n_index.end(), std::default_random_engine(seed));
#endif


	mydouble scale_ball = ymax / sqrt(lbd);
	svm_node** X = prob->x;

	MyMatrixR K_sigma2(1, 1);
	assert(K_sigma2.cols() == 1);
	assert(K_sigma2.rows() == 1);
	K_sigma2(0, 0) = 1.0 + sigma2; //assume RBF
	MyMatrixR Kinv_sigma2(1, 1);
	Kinv_sigma2(0, 0) = 1.0 / K_sigma2(0, 0);

	int w_l = 1;
	int* w_index = new int[N]; //just in case ...
	w_index[0] = n_index[0];
	MyVector w(1);
	w(0) = -2.0 * (0.0 - prob->y[n_index[0]]) / (lbd * 1);
	mydouble wnorm2 = w(0) * w(0); //kernel func = 1

	MyVector wbar(1);
	assert(wbar.rows() == 1);
	wbar(0) = w(0);

	MyVector Kn(1);
	assert(Kn.rows() == 1);
	//no need init

	MyVector d_project(1);
	assert(d_project.rows() == 1);
	//no need init

	std::default_random_engine ugen(time(NULL));
	std::uniform_int_distribution<int> udist(0, N_batch - 1);
	int* pos_core = new int[N];
	for (int n = 0; n < N; n++)
		pos_core[n] = -1;
	for (int t = 1; t <= T; t++)
	{
		int nt = udist(ugen);
		svm_node* xnt = prob->x[nt];
		int ynt = prob->y[nt];

		//predict
		for (int wi = 0; wi < w_l; wi++)
			Kn(wi) = kernel::k_function(xnt, X[w_index[wi]], *param);
		mydouble y_pred_t = Kn.dot(w);
		mydouble alpha_t = y_pred_t - ynt;

		//calc projection
		d_project = Kinv_sigma2 * Kn;

		//calc dist
		mydouble dist2 = 1.0 - Kn.dot(d_project); //assume RBF

		mydouble scale_w = (t - 1.0) / t;
		//scale
		w *= scale_w;

		//update
		if (dist2 > theta)
		{
			printf("dist2:%f;\tt:%d\n", dist2, t);

			int i_new_w = w_l;
			if (pos_core[nt] >= 0)
			{
				i_new_w = pos_core[nt];
				w(i_new_w) += -2.0 * alpha_t / (lbd * t);
			}	
			else
			{
				pos_core[nt] = w_l;
				w_index[i_new_w] = nt;
				w.conservativeResize(w_l + 1);
				w(w_l) = -2.0 * alpha_t / (lbd * t);
				wbar.conservativeResize(w_l + 1);
				wbar(w_l) = 0;
				K_sigma2.conservativeResize(w_l + 1, w_l + 1);
				K_sigma2.block(0, w_l, w_l, 1) = Kn;
				K_sigma2.block(w_l, 0, 1, w_l) = Kn.transpose();
				//assert(K_sigma2(w_l, 0) = Kn(0));
				//assert(K_sigma2(0, w_l) = Kn(0));
				K_sigma2(w_l, w_l) = 1.0 + sigma2;
				Kinv_sigma2 = K_sigma2.inverse();
				Kn.resize(w_l + 1); //no need keep unchange
				d_project.resize(w_l + 1); //no need keep unchange
				w_l++;
			}
		}
		else
		{
			w -= (2.0 * alpha_t / (lbd * t)) * w.cwiseProduct(d_project);
		}
		if (lbd < 2)
		{
			MyMatrixR ww = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
			wnorm2 = (ww.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array()).sum();
			if (!isfinite(wnorm2))
				SHOWERROR(ww);
			if (wnorm2 != 0)
			{
				mydouble scale_project = scale_ball / sqrt(wnorm2);
				if (scale_project < 1)
					w *= scale_project;
			}

		}
		wbar = wbar * (t - 1) / t + w / t;
	}

	//just record online time
	report->reset();
	report->start();

	MyMatrixR ww_K_sigma2 = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
	ww_K_sigma2 = ww_K_sigma2.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array();

	wnorm2 = ww_K_sigma2.sum();

	mydouble sum_square_error = 0;
	mydouble cur_rmse;
	int n_rmse = 0;
	for (int n = N_batch; n < N; n++)
	{
		int t = n + 1;
		int nt = n_index[n];
		svm_node* xnt = prob->x[nt];
		mydouble ynt = prob->y[nt];

		//predict
		for (int wi = 0; wi < w_l; wi++)
			Kn(wi) = kernel::k_function(xnt, X[w_index[wi]], *param);
		mydouble y_pred_t = Kn.dot(w);
		mydouble alpha_t = y_pred_t - ynt;
		sum_square_error += alpha_t * alpha_t;
		cur_rmse = sqrt(sum_square_error / (++n_rmse));
		if (!(n % info_step))
			printf("[%0.2f%%]; core:%d; rmse: %f\n", (float)n/N*100, w_l, cur_rmse);

		//calc projection
		d_project = Kinv_sigma2 * Kn;

		//calc dist
		mydouble dist2 = 1.0 - Kn.dot(d_project); //assume RBF
		//if (dist2 == 1.0)
		//{
		//	SHOWVAR(d_project);
		//	SHOWVAR(Kn);
		//	SHOWERROR("dist2 == 1");
		//}

		mydouble scale_w = (t - 1.0) / t;
		//scale
		w *= scale_w;
		wnorm2 *= scale_w * scale_w;
		ww_K_sigma2 *= scale_w * scale_w;

		//update
		if (dist2 > theta)
		{
			mydouble modifier = -2.0 * alpha_t / (lbd * t);
			std::cout << "dist2: " << dist2 << std::endl;
			w_index[w_l] = nt;
			w.conservativeResize(w_l + 1);
			w(w_l) = modifier;
			wbar.conservativeResize(w_l + 1);
			wbar(w_l) = 0;
			K_sigma2.conservativeResize(w_l + 1, w_l + 1);
			K_sigma2.block(0, w_l, w_l, 1) = Kn;
			K_sigma2.block(w_l, 0, 1, w_l) = Kn.transpose();
			//assert(K_sigma2(w_l, 0) = Kn(0));
			//assert(K_sigma2(0, w_l) = Kn(0));
			K_sigma2(w_l, w_l) = 1.0 + sigma2;
			Kinv_sigma2 = K_sigma2.inverse();
			Kn.resize(w_l + 1); //no need keep unchange
			d_project.resize(w_l + 1); //no need keep unchange
			w_l++;

			wnorm2 += 2 * modifier * y_pred_t * scale_w + (modifier * modifier);
			ww_K_sigma2 = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
			ww_K_sigma2 = ww_K_sigma2.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array();
		}
		else
		{
			MyVector proj_mod = (2.0 * alpha_t / (lbd * t)) * d_project;
			MyMatrixR proj_mod_rep = proj_mod.replicate(1, w_l);

			MyMatrixR kron_proj_mod = proj_mod_rep.array() * proj_mod_rep.transpose().array();
			MyMatrixR kron_proj_c = proj_mod_rep.array() * w.transpose().replicate(w_l, 1).array();

			ww_K_sigma2 = ww_K_sigma2.array() + (-kron_proj_c - kron_proj_c.transpose() + kron_proj_mod).array()
				* (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array();
			w -= proj_mod; //w.cwiseProduct(d_project);
			wnorm2 = ww_K_sigma2.sum();
		}
		if (lbd < 2)
		{
			//MyMatrixR ww = w.replicate(1, w_l).array() * w.transpose().replicate(w_l, 1).array();
			//wnorm2 = (ww.array() * (K_sigma2 - MyMatrixR::Identity(w_l, w_l) * sigma2).array()).sum();
			if (!isfinite(wnorm2))
				SHOWERROR(ww_K_sigma2);
			//SHOWVAR(wnorm2);
			mydouble scale_project = scale_ball / sqrt(wnorm2);
			if (scale_project < 1)
				w *= scale_project;
		}
		wbar = wbar * (t - 1) / t + w / t;
	}

	report->train_time = report->stop();
	report->rmse_rate = cur_rmse;
	SHOWVAR(w_l);
	SHOWVAR(N);
	model->w = new mydouble[w_l];
	model->wbar = new mydouble[w_l];
	for (int wi = 0; wi < w_l; wi++)
	{
		model->w[wi] = w(wi);
		model->wbar[wi] = wbar(wi);
	}
	model->w_index = w_index;
	model->w_l = w_l;
	model->param = *param;
	return report;
}


void gogp_predict(report * report, const svm_problem * test_prob, mydouble *& predict)
{
	gogp_model* model = (gogp_model*)report->model;
	svm_parameter param = model->param;
	report->reset();
	report->start();

	mydouble gamma = model->param.gamma;
	mydouble* w = model->w;
	int* w_index = model->w_index;
	int w_l = model->w_l;

	mydouble* Kn = new mydouble[w_l];
	svm_problem* train_prob = model->prob;
	predict = new mydouble[test_prob->l];

	for (int n = 0; n < test_prob->l; n++)
	{
		svm_node* xn = test_prob->x[n];

		for (int wi = 0; wi < w_l; wi++)
			Kn[wi] = kernel::k_function(xn, train_prob->x[w_index[wi]], param);
		mydouble wxn = 0;
		for (int wi = 0; wi < w_l; wi++)
			wxn += w[wi] * Kn[wi];

		predict[n] = wxn;
	}
	report->predict_time = report->stop();
}

