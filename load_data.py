import numpy as np
import scipy.io as spio
import os



def load(file_name, folder, length = 12, gap = 2):
    load_name = os.path.join(folder, file_name)
    try:
        mat = spio.loadmat(load_name, squeeze_me=True)['ms_all']
    except KeyError:
        mat = spio.loadmat(load_name, squeeze_me=True)['ms']
    samples = int( (mat.shape[0] - length + 1)/gap )
    data = np.zeros( (samples, length, mat.shape[1]) )
    for i in range(samples):
        data[i,:,:] = mat[i*gap:i*gap+length,:]
    return data

def load_undamage_data(args):
    filepath = f'data/data-undamaged-{args.experiment}'
    filenames = os.listdir(filepath)
    # ====== load data
    if len(args.sensor_index) == args.all_sensors and args.experiment in ['crack', 'bc', 'simu']: # all sensors, different setting
        data1 = np.array([load( files, folder=filepath, length=24) for files in filenames]).astype(np.float32)
    else:
        data1 = np.array([load( files, folder=filepath ) for files in filenames]).astype(np.float32)
    # ====== reshape data
    if args.experiment in ['simu', 'simu_temp', 'simu_noise', 'beam_column', 'crack']: #(1, samples, 12, 2*sensors) or (n_test, samples, 12, 2*sensors)
        data = data1.reshape(-1,12,args.all_sensors*2) #reduce 1st dim or comebine 1st & 2nd dim
    elif args.experiment == 'bc': #(sensors, samples, 12, 2)
        data = data1.transpose(1,2,0,3).reshape(-1,12,args.all_sensors*2)[20:,:,:] # remove the initial data varitions
    # ====== transform data
    if args.experiment == 'simu_temp':
        from sklearn.preprocessing import StandardScaler
        args.scaler1, args.scaler2 = StandardScaler(), StandardScaler()
        data1 = args.scaler1.fit_transform(data[:,:,:args.all_sensors].reshape(-1, args.all_sensors)).reshape(-1,12,args.all_sensors)
        data2 = args.scaler2.fit_transform(data[:,:,args.all_sensors:].reshape(-1, args.all_sensors)).reshape(-1,12,args.all_sensors)
        data = np.concatenate( (data1,data2), axis=2)
    if args.experiment == 'beam_column':
        pass
    return data

def load_damage_data(args, index=np.arange(60,80)):
    filepath = f'data/data-damaged-{args.experiment}'
    filenames = os.listdir(filepath)
    if args.experiment in ['simu', 'crack', 'bc']:
        damage_data = np.array([load( files, folder=filepath, length=24) for files in filenames]).astype(np.float32)
    else:
        damage_data = np.array([load( files, folder=filepath ) for files in filenames]).astype(np.float32)
    if args.experiment == 'crack':
        return damage_data.reshape(-1,12,args.all_sensors*2)[20:40]
    elif args.experiment == 'bc':
        return damage_data.transpose(1,2,0,3).reshape(-1,12,args.all_sensors*2)[index]