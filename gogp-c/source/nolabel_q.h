#pragma once

class nolabel_q : public kernel
{
private:
	cache* mycache;
	mydouble* QD;
public:
	nolabel_q(const svm_problem& prob, const svm_parameter& param);

	virtual mydouble *get_q(int i, int len) const override;

	virtual mydouble *get_qd() const override;

	virtual void swap_index(int i, int j) const override;

	virtual ~nolabel_q();
};
