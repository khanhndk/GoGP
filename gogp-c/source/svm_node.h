#pragma once

//index = 1
struct svm_node
{
	int index;
	mydouble value;

	svm_node()
	{
		index = 1;
		value = 0;
	}

	svm_node(int i, mydouble v)
	{
		index = i;
		value = v;
	}
};

inline mydouble calc_dist2(const svm_node* x, const svm_node* y)
{
	mydouble dist = 0;
	for (; x->index != -1 && y->index != -1;)
	{
		if (x->index == y->index)
		{
			mydouble tmp = x->value - y->value;
			dist += tmp * tmp;
			x++;
			y++;
		}
		else
		{
			if (x->index < y->index)
			{
				dist += x->value * x->value;
				x++;
			}
			else
			{
				dist += y->value * y->value;
				y++;
			}
		}
	}
	while (x->index != -1)
	{
		dist += x->value * x->value;
		++x;
	}
	while (y->index != -1)
	{
		dist += y->value * y->value;
		++y;
	}
	return dist;
}