

def target_binning(target='age'):
    bins = [5*i for i in range(17)]
    labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    y = pd.cut(targets[target], bins=bins, labels=labels)
    
    return y


def split_data(X,y,test_size=0.2,random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train,dtype="float32")
    X_test =np.array(X_test,dtype="float32")
    y_train = np.array(y_train,dtype="int8")
    y_test =np.array(y_test,dtype="int8")

    X_train = X_train/255 - 0.5
    X_test = X_test/255 - 0.5
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    del X,y

    return X_train, X_test, y_train, y_test,y_train_cat,y_test_cat