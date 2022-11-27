import ast
from flask import *
import os
import pandas as pd
from models.ml_algos import ML_ALGOS
from flask_cors import CORS
app=Flask(__name__)
cors = CORS(app)
from flask_restful import Resource
import pandas as pd

UPLOAD_FOLDER='static/datasets'
dataset=None
def load_csv(path):
    global dataset
    dataset=pd.read_csv(path)
    return [dataset.to_html(classes='data', header="true")]
def getoutput(do_list):
    algo_obj=ML_ALGOS(dataset)
    result_list=[]
    for i in do_list:
        if i==1:
            result_list.append(algo_obj.FIND_S())
            continue
        elif i==2:
            result_list.append(algo_obj.CANDIDATE_ALGO())
            continue
        elif i==3:
            result_list.append(algo_obj.ID3())
            continue
        elif i==4:
            result_list.append(algo_obj.BackPropagation())
            continue
        elif i==5:
            result_list.append(algo_obj.naive_bayes_GNB())
            continue
        elif i==6:
            result_list.append(algo_obj.naive_bayes_MNB())
            continue
        elif i==7:
            result_list.append(algo_obj.naive_bayes_BNB())
            continue
        elif i==8:
            result_list.append(algo_obj.bayesian_network())
            continue
        elif i==9:
            result_list.append(algo_obj.EM_GMM())
            continue
        elif i==10:
            result_list.append(algo_obj.K_Means())
            continue
        else:
            pass

    return result_list


@app.route('/')
def index():# put application's code here
    return render_template('index.html')
@app.route('/UpdateDateSet/', methods=['GET','POST'])
def UpdateDateSet():
    if request.method=='POST':
        file=request.files['file']
        filename=file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        dat=load_csv(UPLOAD_FOLDER+'/'+filename)
        return render_template('index.html',tables=dat,filename=filename)
    else:
        return 's'
@app.route('/getresult/',methods=['POST','GET'])
def getresult():
    if request.method=='POST':
        do_algo=[]
        filename=request.form['filename']
        data=request.form['dataset']
        data=ast.literal_eval(data)
        for i in range(1,11):
            if request.form.get(str(i)):
                do_algo.append(i)
        print(do_algo)
        result=getoutput(do_algo)
        print(result)
        return render_template('index.html',tables=data,filename=filename,result=result)
    else:
        return abort(500)
@app.route('/getcsvFile/',methods=['POST'])
def getcsvFile():
    if request.method=='POST':
        file =request.data
        data = pd.read_csv(file)
        print('sucess')
    return data



if __name__ == '__main__':
    app.run()
