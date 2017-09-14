// OFOC.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

void parse_command_line(int argc, char **argv, svm_parameter& param, char *input_file_name, char *model_file_name, char* test_file_name);

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	char test_file_name[1024];

	svm_parameter param;

	parse_command_line(argc, argv, param, input_file_name, model_file_name, test_file_name);

	svm_problem train_prob;
	read_problem(input_file_name, param, train_prob);
	if (param.max_index > 0)
		train_prob.max_index = param.max_index;

	if (param.mode == 0)
	{
		report* report = svm_train(&train_prob, &param);

		svm_problem test_prob;
		read_problem(test_file_name, param, test_prob);
		if (param.max_index > 0)
			test_prob.max_index = param.max_index;
		if (train_prob.max_index < test_prob.max_index) SHOWERROR("max_index train < test");
		if (train_prob.max_index > test_prob.max_index)
			test_prob.max_index = train_prob.max_index;

		mydouble* y_pred = NULL;
		svm_predict(report, &test_prob, y_pred);

		report->report_predict(test_prob.y, y_pred, test_prob.l);

		report->write_report(model_file_name);

		report->print_report();

		if (param.demo == 1)
		{
			svm_problem predict_prob;
			read_problem("nolabel.txt", param, predict_prob);
			svm_predict(report, &predict_prob, predict_prob.y);
			write_problem("predict.txt", predict_prob);
		}
		if (param.pause > 0)
			system("PAUSE");

#ifdef _DEBUG
		system("PAUSE");
#endif		
		
	}
	else if (param.mode == 1)
	{
		report* report = svm_train(&train_prob, &param);
		report->write_report_online(model_file_name);
		printf("Mistake rate: %f\n", report->mistake_rate[report->mistake_rate.size() - 1]);
		int n_class = report->model->index_label->size();
		for (int c = 0; c < n_class; c++)
			printf("Mistake rate %d: %f\n", c, report->mistake_class_rate[c][report->mistake_class_rate[c].size() - 1]);
		printf("Train: %f\n", report->train_time);
	}
	else if (param.mode == 5)
	{
		int N = train_prob.l;

		srand(time(NULL));
		std::default_random_engine ugen(rand());
		std::uniform_int_distribution<int> udist(0, N - 1);
		
		for (int n = 0; n < N; n++)
		{
			int r = udist(ugen);
			prob_swap(train_prob, n, r);
		}
		write_problem(test_file_name, train_prob);
	}
    return 0;
}

void parse_command_line(int argc, char **argv, svm_parameter& param, char *input_file_name, char *model_file_name, char* test_file_name)
{
	std::vector<str_parameter> param_list;
	int i;

	#pragma region default values
	strcpy(param.testid, "");
	param.balance = 1;
	param.core_size = 0;
	param.demo = 0;
	strcpy(param.file_omega, "");
	param.full_matrix = 0;
	param.iota = 2;
	param.info_step = 1000;
	param.max_index = -1;
	param.mode = 0;
	param.par = 0; //parallel mode
	param.param_scale = 1;
	param.p1 = 0;
	//need to be sorted
	param.report_type = 0;
	param.svm_type = LSVM;
	param.svm_variant = 0;
	param.kernel_type = ERBF;
	param.degree = 3;
	param.gamma = 10;	// 1/k
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 300;
	param.C = 30;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 0; //KK 2014.11.04
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.T = 2;
	param.sample_size = 50;
	param.max_loop = 2000;
	param.min_vector_sphere = 10;
	param.clustertype = 1;
	param.nr_train = 50;
	param.part = 1;
	param.lbd = 100;
	param.trust = 0.9;
	param.train_one_class = 0;
	param.prune = 10;
	param.prune_threhold = 0.001;
	param.num_weight_init = 3;
	param.num_weight_per_class = 5;
	param.theta = 0;
	param.scale = 20;
	param.pmin = 0.8;
	param.cont = 1;
	param.contdesc = 20;
	param.K = 10;
	param.batch_mode = 0;
	param.batch_size = 5;
	param.pause = 0;


	#pragma endregion

	// parse options
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-') break;
		if (++i >= argc)
			SHOWERROR("input error");
		param_list.push_back(str_parameter(argv[i - 1], argv[i]));

		if (!strcmp(argv[i - 1], "-testid"))
		{
			strcpy(param.testid, argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-crossid"))
		{
			strcpy(param.crossid, argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-runid"))
		{
			strcpy(param.runid, argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-balance"))
		{
			param.balance = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-coresize"))
		{
			param.core_size = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-mode"))
		{
			param.mode = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-fullmatrix"))
		{
			param.full_matrix = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-delta"))
		{
			param.delta = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-iota"))
		{
			param.iota = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-infostep"))
		{
			param.info_step = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-k"))
		{
			param.k = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-p1"))
		{
			param.p1 = atof(argv[i]);;
			continue;
		}
		//need to be sorted
		else if (!strcmp(argv[i - 1], "-report"))
		{
			param.report_type = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-ss"))
		{
			param.svm_variant = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-nS"))
		{
			param.nr_sphere = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-tCl"))
		{
			param.clustertype = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-nTr"))
		{
			param.nr_train = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-minVSp"))
		{
			param.min_vector_sphere = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-part"))
		{
			param.part = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-pause"))
		{
			param.pause = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-lbd"))
		{
			param.lbd = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-tru"))
		{
			param.trust = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-trOC"))
		{
			param.train_one_class = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-alpha"))
		{
			param.alpha = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-beta"))
		{
			param.beta = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-kappa"))
		{
			param.kappa = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-mu"))
		{
			param.mu = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-tau"))
		{
			param.tau = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-theta"))
		{
			param.theta = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-sigma"))
		{
			param.sigma = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-prun"))
		{
			param.prune = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-prunthr"))
		{
			param.prune_threhold = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-nWPC"))
		{
			param.num_weight_per_class = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-nWInit"))
		{
			param.num_weight_init = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-nrCls"))
		{
			param.nr_class = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-scale"))
		{
			param.scale = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-paramscale"))
		{
			param.scale = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-pmin"))
		{
			param.pmin = atof(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-cont"))
		{
			param.cont = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-contdesc"))
		{
			param.contdesc = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-demo"))
		{
			param.demo = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-floss"))
		{
			param.floss = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-ploss"))
		{
			param.ploss = atoi(argv[i]);;
			continue;
		}
		else if (!strcmp(argv[i - 1], "-inc"))
		{
			param.increment = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-incstep"))
		{
			param.increment_step = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-maxloop"))
		{
			param.max_loop = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-maxindex"))
		{
			param.max_index = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-par"))
		{
			param.par = atoi(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-fileomega"))
		{
			strcpy(param.file_omega, argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-epoch"))
		{
			param.epoch = atof(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-batch"))
		{
			param.batch_mode = atof(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-batchsize"))
		{
			param.batch_size = atof(argv[i]);
			continue;
		}
		else if (!strcmp(argv[i - 1], "-dim_rf"))
		{
			param.dim_rf = atoi(argv[i]);
			continue;
		}
		switch (argv[i - 1][1])
		{
		case 's':
			param.svm_type = atoi(argv[i]);
			break;
		case 't':
			param.kernel_type = atoi(argv[i]);
			break;
		case 'T':
			param.T = atoi(argv[i]);
			break;
		case 'K':
			param.K = atoi(argv[i]);
			break;
		case 'M':
			param.sample_size = atoi(argv[i]);
			break;
		case 'd':
			param.degree = atoi(argv[i]);
			break;
		case 'g':
			param.gamma = atof(argv[i]);
			break;
		case 'r':
			param.coef0 = atof(argv[i]);
			break;
		case 'n':
			param.nu = atof(argv[i]);
			break;
		case 'm':
			param.cache_size = atof(argv[i]);
			break;
		case 'c':
			param.C = atof(argv[i]);
			param.nu = 1 / param.C;
			break;
		case 'e':
			param.eps = atof(argv[i]);
			break;
		case 'p':
			param.p = atof(argv[i]);
			break;
		case 'h':
			param.shrinking = atoi(argv[i]);
			break;
		case 'b':
			param.probability = atoi(argv[i]);
			break;
		case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
			param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
			param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
			param.weight[param.nr_weight - 1] = atof(argv[i]);
			break;
		default:
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
			SHOWERROR("input error");
		}
	}

	std::sort(param_list.begin(), param_list.end());
	param.arg = new std::string("");
	for (int ip = 0; ip < param_list.size(); ip++)
	{
		param.arg->append(param_list[ip].name.substr(1));
		param.arg->append(": ");
		param.arg->append(param_list[ip].value);
		param.arg->append("\t");
	}

	// determine filenames

	if (i >= argc)
		SHOWERROR("input error");

	strcpy(input_file_name, argv[i]);

	if (i < argc - 2)
	{
		strcpy(test_file_name, argv[i + 1]);
		strcpy(model_file_name, argv[i + 2]);
	}
	else if (i < argc - 1)
	{
		strcpy(model_file_name, argv[i + 1]);
	}
	else
	{
		SHOWERROR("input error");
	}
}