#pragma once

struct svm_parameter
{
	//for cvs
	char testid[100];
	char crossid[100];
	char runid[100];

	int demo; //0 - normal, 1 - save predict to 2D visualize

	int mode; //0 - batch with one setting, 1 - online with one setting
			  //2 - export sub problem, 3 - cross validation
			  //4 - incremential testing
			  //-1 - just for testing

	int balance; //run with balance data or not
	int par; //parallel mode
	int full_matrix; //run in full maxtrix mode
	int info_step; //num step to report or info
	double p1;

	int report_type;
	int svm_type;
	int svm_variant;
	int kernel_type;
	int nr_class; //for AMMNorm1
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

					/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */

	int T; /*for LSVM  or AMMNorm1*/
	int sample_size; /*for CSVM, SApproxOCSVM */
	int nr_sphere; /*for multisphere model*/
	int clustertype; /*for multisphere model*/

	int max_loop;
	int nr_train;
	int min_vector_sphere; //for HMS_SVDD
	double part; //for MMEB;
	double lbd; //for MMEB & AMMNorm1
	double trust; //for MMEB

	int prune; //for AMMNorm1
	double prune_threhold; //for AMMNorm1, MSVDD
	int num_weight_per_class; //for AMMNorm1
	int num_weight_init; //for AMMNorm1

	double scale; //for MSVDD
	double pmin; //for MSVDD
	int cont; //for MSVDD
	int contdesc; //for MSVDD

	int K; //for SGDBoost, for OFOC_FF dim_rf = K*D
	double k; //for k-NN

	int floss; //using which loss funcion
	int ploss; //using which p-normalization

	int train_one_class; //just use only one class
	int have_unlabel; //for unlabel problem

	int chunk_size; //for load large data
	int max_index;
	int increment; //for testing increment. increment = 0: disable
	int increment_step; //for teseting increment. indicate how many step from start to end

	int param_scale;

	double epoch;

	double sigma;
	double alpha;
	double beta;
	double delta;
	double kappa;
	double iota;
	double mu;
	double tau;
	double theta;

	int batch_mode; //mini batch
	int batch_size;
	int core_size;
	int dim_rf; //dim of random feature space

	char file_omega[256];

	std::string* arg;

	int pause;
};

struct str_parameter
{
	std::string name;
	std::string value;
	str_parameter(char* _name, char* _value)
	{
		name = std::string(_name);
		value = std::string(_value);
	}
	bool operator<(const str_parameter& rhs) const
	{
		return name.compare(rhs.name) < 0;
	}
};