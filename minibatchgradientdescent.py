import pandas
import sys
import random as rnd

#Written by Mert KAYA - 161101061 - 11 June 2020
#Iteration count can be changed via a parameter in main() method func.
#It iterates 10000 times. So it might take couple of minutes.
#There is a code part commented in main method to print weights


#Global Variables
train_data = []
test_data = []
model = []
batch_sum_arr = []
target_name = ""


#Implementation for mini-batch gradient descent
def gradient_descent(iteration):

    global model

    #initial settings for global model
    old_model = model
    new_model = model

    #loop for iteration times.
    for i in range(iteration):

        #Setting the last changed model as old_model
        old_model = new_model

        #Calculating mse of old_model
        mse_old_value = mse(old_model)

        #creating a new_model
        new_model = minibatch_lin_reg(old_model)

        #Calculating the value of the new_model
        mse_new_value = mse(new_model)

        #stop if the model is getting overfitted   
        #This if will make it stop in a local min     

        #if MSE(new_model) is smaller than 10^27 than it is fitted enough
        #mse_old - mse_new can be implemented too for a more precise result
        if(mse_new_value < pow(10,-27)):
            break
        
        #if the Mse is not changing much anymore it means it is a fix point
        if(abs(mse_old_value)-abs(mse_new_value) < pow(10,-27)):
            break
    
    print('Final test data MSE : ',mse_old_value)
    #global model is changed.
    model = old_model
    return old_model


#To test the model for this scope
def mse(model):

    #obtaining test_data
    global test_data

    result = 1

    sum = 0

    #sum of residual^2 from 1 to test data
    for i in range(len(test_data.values)):
        yi = float(test_data[target_name].values[i])
        yi_est = float(run_model(i,"test"))
        residual_pow = float(pow((float(yi)-float(yi_est)),2))
        sum = sum + residual_pow

    result = result * sum
    #(1/n) * ... 
    result = result * (float(1)/float(len(test_data.values)))
    return result


#To calc (2/n)*lr*sum(residuals*partial_der)
def calc_val(learning_rate,part,batch_size,index):

    batch_range = int(batch_size)
    batch_size = float(batch_size)
    result = 1.0
    result = float(float(learning_rate)*result)

    #2/n from partial derivative
    
    result = float(result*float(2/batch_size))

    #to obtain the train_data's target value
    global target_name

    sum_of_y_yest = 0.0
    for i in range(batch_range):

        #train_data's value for that sample
        yi = float(train_data[target_name].values[index+i])

        #running our model for that sample
        y_estimate = float(run_model(index+i,"train"))
        
        residual = float(yi-y_estimate)
        #value of the feature
        if(part["feature"] != ""):
            val = float(train_data[part["feature"]].values[index+i])
        else:
            #If the weight is featureless.
            val = 1.0

        #pow(val,part[exp]) is obtained from the models derivative
        residual_sum = float(residual*pow(float(val),float(part["exponent"])))   
        sum_of_y_yest = float(sum_of_y_yest + residual_sum)
    
    # lr * 2/n * sum_value
    result = float(float(result) * float(sum_of_y_yest))
    return result


#To run our model for given index of sample on train_data or test_data
def run_model(index,train_or_test):
    global model
    sum = 0.0
    val = 0.0
    header = list(train_data.columns.values)
    i = 1
    for part in model:

        #To obtain value of given feature on data
        name = header[i]
        #value of the feature
        if(train_or_test == "train"):
            val = float(train_data[name].values[index])
        else:
            val = float(test_data[name].values[index])
        res = 1.0

        #val^exponent of given feature
        if(part["exponent"] != ""):
            res = float(res*pow(val,part["exponent"]))
            
        #weight * result
        res = res*part["weight"]
        sum = sum + res
        i = i+1
    return sum

#Default learning rate is 0.1 because it is a common value used find min
#If learning rate is small there must be more iterations.
#It was a trial and error thing.
#Batch_size is 32 because it is commonly used. Can be changed.
def minibatch_lin_reg(model,learning_rate = 0.1,batch_size = 32):
    last_batch = False
    train_data_size = len(train_data.values)
    #Checking if batch_size is smaller or not than the train_data
    if train_data_size < batch_size:
        batch_size = train_data_size

    #To control the batch index on data
    batch_start_count = 0

    #To hold weight's that are fitted
    weight_array = []

    #default weights
    for part in model:
        weight_array.append(part["weight"])

    #from the batch_starting_index to length_of_training_data
    #adds batch_size to batch_starting_count each iteration
    #updates weights after the batch is fitted.
    for i in range(len(train_data.values)):
        
        if batch_start_count == len(train_data.values) :
            break

        #index of the weight (Which features weight it is or is it featureless)
        weight_count = 0

        #For each weight : 
        for part in model:       
            #Holding the updated weight change in an array
            weight_array[weight_count] = calc_val(learning_rate,part,batch_size,batch_start_count)
            #Which weight is updated
            weight_count = weight_count+1
        
        #Updating weights and incrementing the batch_start_index
        #update_weights takes the value to be added to the weights in an array
        update_weights(weight_array)
        batch_start_count = batch_start_count + batch_size
        
        #to break after doing the last(smaller) batch
        if(last_batch):
            return model

        #In case of a batch size problem : 
        # Batch size is updated according to how much data is left
        if ((len(train_data.values) - batch_start_count) < batch_size):
            batch_size = (len(train_data.values) -(batch_start_count))  
            last_batch = True  

    return model


#To update the values of weights after each mini-batch
def update_weights(weight_array):
    global model
    weight_num = 0
    for part in model:
        #Setting each weight of our model with fitted ones
        part["weight"] = float(float(part["weight"]) + float(weight_array[weight_num]))
        weight_num = weight_num + 1
    return model

#To shuffle the dataset to later split for train-test
def shuffle_df(df):
    randomized_df = df.sample(frac=1)
    return randomized_df

#Splitting data for train and test 
#%80 of the dataset is train_data
#%20 of the dataset is test_data
def split_data_train_test(dataset):
    global train_data
    global test_data
    row_count = dataset.shape[0]
    split_point = int(row_count*4/5)
    test_data, train_data = dataset[split_point:], dataset[:split_point]


#setting default values for weights and exponents
#Start weights can be choosed anything.
def set_def_weights(parts):
    new_arr = parts[0]
    for part in new_arr:
        part["weight"] = float(1)
        #If a polynom looks like 3x+5 that means x has exponent of 1.
        if part["exponent"] == "":
            part["exponent"] = float(1)
    return new_arr

    
#After splitting the model_txt
#We need to gather all the parts of it
def create_element_from_part(part):
    part = part.replace(" ","")
    weight = "" 
    feature = ""
    exponent = ""

    #feature is after '*'
    feature_index = part.find("*")
    if(feature_index != -1):
        weight = part[0:feature_index]

    #exponent is after '^'
    exponent_index = part.find("^")
    if (feature_index != -1):
        if  exponent_index != -1:
            feature = part[feature_index+1:exponent_index]
        else:
            feature = part[feature_index+1:len(part)]

    if exponent_index != -1:
        exponent = float(part[exponent_index+1:len(part)])
    
    #If there is no feature which means B0 case.
    feature = feature.replace(" ","")
    
    #Returning them as a dict.
    return{
        'feature' : feature,
        'weight' : weight,
        'exponent' : exponent    
    }

#Splitting the model_txt
def create_parts_from_model_text(model_text):

    part_array = model_text.split("+")
    parts_of_model = []
    for part in part_array:
        parts_of_model.append(create_element_from_part(part))
    return parts_of_model
        


def main():

    #Model and dataset are gathered from the arguments
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    model_txt = open(model_name,"r")
    model_text = model_txt.read()

    #Reading dataset with pandas
    df = pandas.read_csv(dataset_name)

    #Splitting model    
    parts_of_model = []
    parts_of_model.append(create_parts_from_model_text(model_text))
    parts_of_model = set_def_weights(parts_of_model)

    #Setting model to a global variable
    global model
    model = parts_of_model

    #Shuffling dataset (can be suffled by this method)
    df = shuffle_df(df)

    #For better performance target_name is gathered once.
    global target_name
    header = list(df.columns.values)
    end_num = len(header)-1
    target_name = header[end_num]

    #Splitting data as train and test
    split_data_train_test(df)
    #Fitting weights with Gradient descent for 10000 iterations
    #if it takes a long time run it less than 10000 times.
    gradient_descent(10000)
    

    #to print final weights

    #for part in model:
    #    print(part["weight"])


if __name__ == "__main__":
    main()
 
        