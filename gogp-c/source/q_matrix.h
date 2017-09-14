#pragma once

class q_matrix {
public:
	virtual mydouble *get_q(int column, int len) const = 0;
	virtual mydouble *get_qd() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~q_matrix() {}
};