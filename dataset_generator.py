import sys
import random
from datetime import datetime

class Generator:
    def __init__(self,num_of_samples):
        self.num_of_samples = int(num_of_samples)
        self.features = ["F1","F2","F3","F4","F5"]
        self.id = datetime.today().strftime('%Y-%m-%d-%H.%M.%S')
        self.weights = {}
        self.powers = {}
        for f in self.features: 
            self.powers[f] = random.randint(1,4)
            self.weights[f] = random.randint(0,20)
        

    def evaluate(self, feature_values):
        return sum([self.weights.get(f)*pow(feature_values.get(f), self.powers.get(f)) for f in self.features])

    def save_model(self):
        model_file = open("model.txt", "w")
        model_arr = []
        i=1
        for f in self.features:
            w = self.weights.get(f)
            p = self.powers.get(f)
            if w == 0:
                continue
            m = "a_"+str(i)
            i+=1
            if p > 0:
                m += "*"+ f
            if p > 1:
                m += "^" + str(p)
            model_arr.append(m)
        model_file.write(" + ".join(model_arr))
        
    def save_data(self):
        w = open("data.csv",'w')
        data = []
        for i in range(self.num_of_samples):
            arr = [i+1]
            feature_values = {}
            for f in self.features:
                feature_values[f] = random.random()
                arr.append(feature_values.get(f))
            arr.append(self.evaluate(feature_values))
            data.append(arr)
        str_data = ","+",".join(self.features)+",target\n"
        for a in data:
            str_data += ",".join([str(x) for x in a]) + "\n"
        w.write(str_data)
            
    def save(self):
        self.save_model()
        self.save_data()
        i=0
        for f in self.features:
            print("a_"+str(i+1)+": "+str(self.weights.get(f)))
            i+=1
if len(sys.argv) != 2:
    print("Usage: python dataset_generator.py <num_of_samples>")
else:
    Generator(sys.argv[1]).save()