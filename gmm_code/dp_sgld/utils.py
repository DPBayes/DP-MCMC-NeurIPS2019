import pickle
import datetime

def fnamer(fname):
	date = datetime.date.today()
	fname += str(date.day)+'_'+str(date.month)+'.p'
	fname_extension = 0
	while True:
		try: 
			f = open(fname, 'rb')
			f.close()
			if fname_extension==0:
				fname = fname[:-2]+'({}).p'.format(fname_extension)
			else:
				fname = fname[:-len('({}).p'.format(fname_extension))]+'({}).p'.format(fname_extension)
			fname_extension += 1
		except:
			return fname
