import pickle

dataname = 'slice.tu.sh2_170315_110122'

report = pickle.load(open('log/'+dataname+'.log.p', "rb"))

print 'lbd=', report.lbd
print 'theta=', report.theta
print 'online_time', report.online_time

print 'final_rmse', report.final_rmse

print 'core_limit', report.core_limit