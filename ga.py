import multiprocessing
import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from tqdm import tqdm


def generator(n):
    return np.random.uniform(size=n).tolist()

def model_prediction(result_df, theta):
    
    result_df['p0/t0'] = result_df['p0']/theta[0]
    result_df['p1/t1'] = result_df['p1']/theta[1]
    result_df['p2/t2'] = result_df['p2']/theta[2]
    result_df['p3/t3'] = result_df['p3']/theta[3]

    result_df['predict/t'] = 0
    result_df['max_proba'] = 0

    for j, col in enumerate(['p0/t0', 'p1/t1', 'p2/t2', 'p3/t3']):
        result_df.loc[result_df[col]>result_df['max_proba'], 'predict/t'] = j
        result_df.loc[result_df[col]>result_df['max_proba'], 'max_proba'] = result_df[col]

    result_df['model0_pred/t'] = 0
    result_df.loc[result_df['predict/t']>0,'model0_pred/t'] = 1
    result_df['model1_pred/t'] = 0
    result_df.loc[result_df['predict/t']>1,'model1_pred/t'] = 1
    result_df['model2_pred/t'] = 0
    result_df.loc[result_df['predict/t']>2,'model2_pred/t'] = 1

    fp_t = [0, 0, 0]
    fp_negatives = [0, 0, 0]
    fpr_t = [0, 0, 0]

    tp_t = [0, 0, 0]
    tp_positives = [0, 0, 0]
    tpr_t = [0, 0, 0]

    for idx, col in enumerate([['model0_pred/t', 'model0_truth'], ['model1_pred/t', 'model1_truth'], ['model2_pred/t', 'model2_truth']]):
        fp_t[idx]=1
        fp_negatives[idx]=1
        if ((result_df[col[0]]==1) & (result_df[col[1]]==0)).value_counts()[False]!=result_df.shape[0]:
            fp_t[idx] = ((result_df[col[0]]==1) & (result_df[col[1]]==0)).value_counts()[True]
        if (result_df[col[1]]==0).value_counts()[False]!=result_df.shape[0]:
            fp_negatives[idx] = (result_df[col[1]]==0).value_counts()[True]
        fpr_t[idx] = fp_t[idx]/fp_negatives[idx]
    
    return [(fpr_t[0]-0.1)**2 + (fpr_t[1]-0.1)**2 + (fpr_t[2]-0.1)**2]

def get_params(theta, result_df):
    result_df['p0/t0'] = result_df['p0']/theta[0]
    result_df['p1/t1'] = result_df['p1']/theta[1]
    result_df['p2/t2'] = result_df['p2']/theta[2]
    result_df['p3/t3'] = result_df['p3']/theta[3]

    result_df['predict/t'] = 0
    result_df['max_proba'] = 0

    for j, col in enumerate(['p0/t0', 'p1/t1', 'p2/t2', 'p3/t3']):
        result_df.loc[result_df[col]>result_df['max_proba'], 'predict/t'] = j
        result_df.loc[result_df[col]>result_df['max_proba'], 'max_proba'] = result_df[col]

    result_df['model0_pred/t'] = 0
    result_df.loc[result_df['predict/t']>0,'model0_pred/t'] = 1
    result_df['model1_pred/t'] = 0
    result_df.loc[result_df['predict/t']>1,'model1_pred/t'] = 1
    result_df['model2_pred/t'] = 0
    result_df.loc[result_df['predict/t']>2,'model2_pred/t'] = 1

    fp_t = [0, 0, 0]
    fp_negatives = [0, 0, 0]
    fpr_t = [0, 0, 0]

    tp_t = [0, 0, 0]
    tp_positives = [0, 0, 0]
    tpr_t = [0, 0, 0]

    for idx, col in enumerate([['model0_pred/t', 'model0_truth'], ['model1_pred/t', 'model1_truth'], ['model2_pred/t', 'model2_truth']]):
        fp_t[idx]=1
        fp_negatives[idx]=1
        if ((result_df[col[0]]==1) & (result_df[col[1]]==0)).value_counts()[False]!=result_df.shape[0]:
            fp_t[idx] = ((result_df[col[0]]==1) & (result_df[col[1]]==0)).value_counts()[True]
        if (result_df[col[1]]==0).value_counts()[False]!=result_df.shape[0]:
            fp_negatives[idx] = (result_df[col[1]]==0).value_counts()[True]
        fpr_t[idx] = fp_t[idx]/fp_negatives[idx]
        
    loss = (fpr_t[0]-0.1)**2 + (fpr_t[1]-0.1)**2 + (fpr_t[2]-0.1)**2
    
    best_params = {}
    for idx, col in enumerate([['model0_pred/t', 'model0_truth'], ['model1_pred/t', 'model1_truth'], ['model2_pred/t', 'model2_truth']]):
        tp_t[idx]=0
        tp_positives[idx]=1
        if ((result_df[col[0]]==1) & (result_df[col[1]]==1)).value_counts()[False]!=result_df.shape[0]:
            tp_t[idx] = ((result_df[col[0]]==1) & (result_df[col[1]]==1)).value_counts()[True]
        if (result_df[col[1]]==1).value_counts()[False]!=result_df.shape[0]:
            tp_positives[idx] = (result_df[col[1]]==1).value_counts()[True]
        tpr_t[idx] = tp_t[idx]/tp_positives[idx]

    best_params['loss'] = loss
    best_params['theta_0'] = theta[0]
    best_params['theta_1'] = theta[1]
    best_params['theta_2'] = theta[2]
    best_params['theta_3'] = theta[3]
    best_params['fpr0'] = fpr_t[0]
    best_params['fpr1'] = fpr_t[1]
    best_params['fpr2'] = fpr_t[2]
    best_params['tpr0'] = tpr_t[0]
    best_params['tpr1'] = tpr_t[1]
    best_params['tpr2'] = tpr_t[2]
    return best_params



def ga(df, generation=100, population_size=300, offspring_size=300, tourn_size=280, change_rate=(0.8, 0.4), pool_size=20, features=4):
    '''
    change_rate = (crossover rate, mutation rate)
    '''
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    pool = multiprocessing.Pool(pool_size)
    toolbox.register("map", pool.map)
    toolbox.register("attr_bool", generator, features)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", model_prediction, df)
    toolbox.register("mate", tools.cxUniform, indpb=change_rate[0])
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=change_rate[1])
    toolbox.register("select", tools.selTournament, tournsize=offspring_size)

    pop = toolbox.population(n=population_size)
    for gen in tqdm(range(generation)):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=change_rate[0], mutpb=change_rate[1])
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=offspring_size)
    top = tools.selBest(pop, k=1)
    print('loss: ', model_prediction(df, top[0]), '\n')
    print(get_params(top[0], df))
    return top