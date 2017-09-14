#pragma once

class cache
{
public:
	cache(int l, long int size);
	~cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, mydouble **data, int len);
	void swap_index(int i, int j);

	//KK 2014.07.09
	bool check_q(int i, int j, mydouble& r);

private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		mydouble *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};