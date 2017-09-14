#include "stdafx.h"

cache::cache(int l_, long int size_) :l(l_), size(size_)
{
	head = (head_t *)calloc(l, sizeof(head_t));	// initialized to 0
	size /= sizeof(mydouble);
	size -= l * sizeof(head_t) / sizeof(mydouble);
	size = std::max(size, 2 * (long int)l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

cache::~cache()
{
	for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
		_aligned_free(h->data);
	free(head);
}

void cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int cache::get_data(const int index, mydouble **data, int len)
{
	head_t *h = &head[index];
	if (h->len) lru_delete(h);
	int more = len - h->len;

	if (more > 0)
	{
		// free old space
		while (size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			//KK 2016.01.04
			_aligned_free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (mydouble *)_aligned_realloc(h->data, sizeof(mydouble)*len, 64);
		size -= more;
		std::swap(h->len, len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void cache::swap_index(int i, int j)
{
	if (i == j) return;

	if (head[i].len) lru_delete(&head[i]);
	if (head[j].len) lru_delete(&head[j]);
	std::swap(head[i].data, head[j].data);
	std::swap(head[i].len, head[j].len);
	if (head[i].len) lru_insert(&head[i]);
	if (head[j].len) lru_insert(&head[j]);

	if (i>j) std::swap(i, j);
	for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
	{
		if (h->len > i)
		{
			if (h->len > j)
				std::swap(h->data[i], h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				_aligned_free(h->data); //KK 2016.06.17 align free
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

bool cache::check_q(int i, int j, mydouble& r)
{
	if ((head[i].data != NULL) && (head[i].len > j))
	{
		r = head[i].data[j];
		return true;
	}
	return false;
}