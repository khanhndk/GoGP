#pragma once

struct gogp_model : svm_model
{
	int* w_index;
	int w_l;
	mydouble* w;
	mydouble* wbar;

	void write_report(std::ofstream& fp) override
	{
		SAVEVAR(fp, coresize, w_l);
	}
};

struct gogp_report : report
{
	mydouble rmse_rate;
	virtual void write_indicators(std::ofstream& fp) override
	{
		report_regression* r_regression = (report_regression*)r_predict;
		SAVEVAR(fp, rmse_rate, rmse_rate);
		SAVEVAR(fp, rmse_offline, r_regression->rmse);
		SAVEVAR(fp, mse_offline, r_regression->mse);
		SAVEVAR(fp, train, train_time);
		SAVEVAR(fp, test, predict_time);
	}

	virtual void print_report() override
	{
		report_regression* r_regression = (report_regression*)r_predict;
		printf("RMSE Rate (Online): %f\n", rmse_rate);
		printf("RMSE (Offline Prediction): %f\n", r_regression->rmse);
		printf("Train Time: %f\n", train_time);
	}
};

report* solve_gogp(svm_problem *prob, const svm_parameter *param);
void gogp_predict(report * report, const svm_problem * test_prob, mydouble *& predict);