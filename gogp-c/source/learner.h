#pragma once

report *svm_train(svm_problem *prob, const svm_parameter *param);

void svm_predict(report* report, const svm_problem *test_prob, mydouble*& predict);